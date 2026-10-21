"""Unitree G1 (unitree_hg) low-level messages without any DDS library.

* LowCmd packing + CRC exactly as unitree_sdk2py/utils/crc.py (__packFmtHGLowCmd, crc32_core), checked against
  the SDK's own crc_amd64.so in tests. The firmware drops a LowCmd whose crc does not match.
* LowState decoding straight from the raw CDR bytes of a ROS 2 subscription (raw=True): building 35 Python
  MotorState objects 500 times a second would cost a large share of a Jetson core and the GIL.
* Wireless-remote buttons as in unitree_sdk2py/utils/joystick.py (bytes 2-3 of wireless_remote).
"""
import ctypes
import os
import platform
import math
import struct
from pathlib import Path

import numpy as np

NUM_MOTORS = 35            # message arrays (unitree_hg)
NUM_BODY = 29              # G1 29-DoF body joints = policy joint order (G1JointIndex 0..28)
MODE_PR = 0                # ankle pitch/roll command mode (unitree_hg mode_pr); the policy acts on pitch/roll joints

LOWCMD_FORMAT = '<2B2x' + 'B3x5fI' * NUM_MOTORS + '5I'          # 1004 bytes, C layout used for the CRC
LOWCMD_SIZE = struct.calcsize(LOWCMD_FORMAT)
assert LOWCMD_SIZE == 1004


# ------------------------------------------------------------------------------------------------ CRC
def _table():
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
            crc &= 0xFFFFFFFF
        table.append(crc)
    return table


_TABLE = _table()


def crc32_core(words):
    """Unitree crc32_core: MSB-first CRC-32 (poly 0x04C11DB7, init 0xFFFFFFFF, no reflection, no final xor) over
    32-bit words, each word fed most-significant bit first. Equivalent to CRC-32/MPEG-2 over the bytes of every
    word in big-endian order; table-driven (bit-wise Python would take ~5 ms per 1004-byte command)."""
    crc = 0xFFFFFFFF
    table = _TABLE
    for word in words:
        for shift in (24, 16, 8, 0):
            crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ (word >> shift)) & 0xFF]
    return crc


def _load_sdk_crc():
    """The SDK's crc32_core (unitree_sdk2py/utils/lib/crc_<arch>.so): ~0.01 ms instead of ~0.4 ms in Python.
    Used only if it reproduces the Python reference on a test vector; G1_SDK_CRC=python forces the fallback."""
    if os.environ.get('G1_SDK_CRC') == 'python':
        return None
    name = {'x86_64': 'crc_amd64.so', 'aarch64': 'crc_aarch64.so'}.get(platform.machine())
    dirs = [os.environ.get('G1_SDK_CRC_DIR'),
            str(Path(__file__).resolve().parents[3] / 'third_party/unitree_sdk2_python/unitree_sdk2py/utils/lib')]
    for d in dirs:
        if not d or not name or not (Path(d) / name).exists():
            continue
        try:
            lib = ctypes.CDLL(str(Path(d) / name))
            fn = lib.crc32_core; fn.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32); fn.restype = ctypes.c_uint32
            words = [(i * 2654435761) & 0xFFFFFFFF for i in range(251)]
            if fn((ctypes.c_uint32 * len(words))(*words), len(words)) == crc32_core(words):
                return fn
        except OSError:
            pass
    return None


_SDK_CRC = _load_sdk_crc()
CRC_IMPLEMENTATION = 'sdk' if _SDK_CRC is not None else 'python'


def crc_of_packed(packed):
    """CRC over all little-endian 32-bit words except the last one (the crc field itself), as the SDK does."""
    count = len(packed) // 4 - 1
    if _SDK_CRC is not None:
        return _SDK_CRC((ctypes.c_uint32 * count).from_buffer_copy(packed, 0), count)
    return crc32_core(struct.unpack_from('<%dI' % count, packed))


# ------------------------------------------------------------------------------------------------ LowCmd
class LowCmd:
    """One G1 low-level command: 35 motors (only the 29 body joints are enabled), PD targets + feedforward."""

    def __init__(self, mode_machine):
        self.mode_pr = MODE_PR
        self.mode_machine = int(mode_machine)
        self.mode = np.zeros(NUM_MOTORS, np.uint8)
        self.q = np.zeros(NUM_MOTORS); self.dq = np.zeros(NUM_MOTORS); self.tau = np.zeros(NUM_MOTORS)
        self.kp = np.zeros(NUM_MOTORS); self.kd = np.zeros(NUM_MOTORS)
        self.crc = 0

    @classmethod
    def body(cls, mode_machine, q, kp, kd, tau=None, dq=None, enable=True):
        cmd = cls(mode_machine)
        n = NUM_BODY
        cmd.mode[:n] = 1 if enable else 0
        cmd.q[:n] = q; cmd.kp[:n] = kp; cmd.kd[:n] = kd
        if tau is not None:
            cmd.tau[:n] = tau
        if dq is not None:
            cmd.dq[:n] = dq
        for name in ('q', 'dq', 'tau', 'kp', 'kd'):
            if not np.isfinite(getattr(cmd, name)).all():
                raise ValueError("non-finite LowCmd " + name)
        cmd.finalize()
        return cmd

    def pack(self, crc=None):
        values = [self.mode_pr, self.mode_machine]
        for i in range(NUM_MOTORS):
            values += [int(self.mode[i]), float(self.q[i]), float(self.dq[i]), float(self.tau[i]),
                       float(self.kp[i]), float(self.kd[i]), 0]
        values += [0, 0, 0, 0, self.crc if crc is None else crc]
        return struct.pack(LOWCMD_FORMAT, *values)

    def finalize(self):
        self.crc = crc_of_packed(self.pack(crc=0))
        return self

    def as_float32(self, name):
        """Values exactly as they go on the wire (float32), for logs and the simulated robot."""
        return getattr(self, name).astype(np.float32).astype(float)


# ------------------------------------------------------------------------------------------------ LowState (CDR)
def _cdr_layout():
    """Field offsets of unitree_hg/msg/LowState in a little-endian CDR1 payload (after the 4-byte encapsulation).
    XCDR1 aligns every primitive to its own size, relative to the payload start; nested structs add no padding of
    their own. (This differs from the C struct for motor 0, whose uint8 mode follows the int16 IMU temperature.)"""
    off = 0
    layout = {}
    def take(name, size, count=1):
        nonlocal off
        off = (off + size - 1) // size * size
        layout.setdefault(name, []).append(off)
        off += size * count
    take('version', 4, 2); take('mode_pr', 1); take('mode_machine', 1); take('tick', 4)
    take('imu.quaternion', 4, 4); take('imu.gyroscope', 4, 3); take('imu.accelerometer', 4, 3); take('imu.rpy', 4, 3)
    take('imu.temperature', 2)
    for _ in range(NUM_MOTORS):
        take('m.mode', 1); take('m.q', 4); take('m.dq', 4); take('m.ddq', 4); take('m.tau_est', 4)
        take('m.temperature', 2, 2); take('m.vol', 4); take('m.sensor', 4, 2); take('m.motorstate', 4); take('m.reserve', 4, 4)
    take('wireless_remote', 1, 40); take('reserve', 4, 4); take('crc', 4)
    return layout, off


LOWSTATE_LAYOUT, LOWSTATE_CDR_SIZE = _cdr_layout()
_Q = np.array(LOWSTATE_LAYOUT['m.q'][:NUM_BODY]) // 4
_DQ = np.array(LOWSTATE_LAYOUT['m.dq'][:NUM_BODY]) // 4
_TAU = np.array(LOWSTATE_LAYOUT['m.tau_est'][:NUM_BODY]) // 4
_MODE = np.array(LOWSTATE_LAYOUT['m.mode'][:NUM_BODY])
_TEMP = np.array(LOWSTATE_LAYOUT['m.temperature'][:NUM_BODY]) // 2


class LowState:
    __slots__ = ('tick', 'mode_pr', 'mode_machine', 'q', 'dq', 'tau', 'motor_mode', 'temperature',
                 'quat_wxyz', 'gyro', 'acc', 'rpy', 'remote', 'received')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def decode_lowstate(data, received=None):
    """Raw CDR (bytes incl. the 4-byte encapsulation) -> LowState. Little-endian only (the robot's encoding)."""
    data = bytes(data)
    if len(data) < 4 + LOWSTATE_CDR_SIZE - 3 or data[:2] != b'\x00\x01':
        raise ValueError("not a little-endian CDR1 unitree_hg LowState (%d bytes)" % len(data))
    payload = data[4:]
    if len(payload) < LOWSTATE_CDR_SIZE:
        payload = payload + b'\x00' * (LOWSTATE_CDR_SIZE - len(payload))   # CDR may drop final alignment bytes
    floats = np.frombuffer(payload, '<f4', count=LOWSTATE_CDR_SIZE // 4)
    shorts = np.frombuffer(payload, '<i2', count=LOWSTATE_CDR_SIZE // 2)
    raw = np.frombuffer(payload, np.uint8)
    L = LOWSTATE_LAYOUT
    imu = lambda name, n: floats[L[name][0] // 4: L[name][0] // 4 + n].astype(float)
    remote_at = L['wireless_remote'][0]
    return LowState(tick=struct.unpack_from('<I', payload, L['tick'][0])[0],
                    mode_pr=int(raw[L['mode_pr'][0]]), mode_machine=int(raw[L['mode_machine'][0]]),
                    q=floats[_Q].astype(float), dq=floats[_DQ].astype(float), tau=floats[_TAU].astype(float),
                    motor_mode=raw[_MODE].copy(), temperature=shorts[_TEMP].astype(int),
                    quat_wxyz=imu('imu.quaternion', 4), gyro=imu('imu.gyroscope', 3), acc=imu('imu.accelerometer', 3),
                    rpy=imu('imu.rpy', 3), remote=bytes(raw[remote_at:remote_at + 40]), received=received)


def encode_lowstate_cdr(tick, mode_machine, q, dq, tau, quat_wxyz, gyro, acc, remote=bytes(40), mode_pr=MODE_PR):
    """Inverse of decode_lowstate (tests and the simulated robot's raw path)."""
    payload = bytearray(LOWSTATE_CDR_SIZE)
    L = LOWSTATE_LAYOUT
    put = lambda fmt, off, *v: struct.pack_into(fmt, payload, off, *v)
    put('<I', L['tick'][0], int(tick) & 0xFFFFFFFF); payload[L['mode_pr'][0]] = mode_pr; payload[L['mode_machine'][0]] = mode_machine
    put('<4f', L['imu.quaternion'][0], *quat_wxyz); put('<3f', L['imu.gyroscope'][0], *gyro); put('<3f', L['imu.accelerometer'][0], *acc)
    for i in range(NUM_BODY):
        payload[L['m.mode'][i]] = 1
        put('<f', L['m.q'][i], q[i]); put('<f', L['m.dq'][i], dq[i]); put('<f', L['m.tau_est'][i], tau[i])
    payload[L['wireless_remote'][0]:L['wireless_remote'][0] + 40] = bytes(remote)[:40].ljust(40, b'\x00')
    return b'\x00\x01\x00\x00' + bytes(payload)


# ------------------------------------------------------------------------------------------------ remote
# joystick.py: button1 = f'{remote[2]:08b}', button2 = f'{remote[3]:08b}' (string index 0 = bit 7).
_BUTTONS = {'R1': (2, 0), 'L1': (2, 1), 'start': (2, 2), 'select': (2, 3), 'R2': (2, 4), 'L2': (2, 5),
            'A': (3, 0), 'B': (3, 1), 'X': (3, 2), 'Y': (3, 3), 'up': (3, 4), 'right': (3, 5), 'down': (3, 6), 'left': (3, 7)}


def remote_buttons(remote):
    """Set of pressed button names (Unitree remote: L1/R1 = shoulder, L2/R2 = triggers)."""
    if remote is None or len(remote) < 4:
        return frozenset()
    return frozenset(name for name, (byte, bit) in _BUTTONS.items() if remote[byte] >> bit & 1)


def remote_sticks(remote):
    """(lx, ly, rx, ry) of the Unitree remote, each in [-1, 1]: lx right +, ly forward + (unitree_sdk2py Joystick)."""
    if remote is None or len(remote) < 24:
        return 0., 0., 0., 0.
    lx, rx, ry = struct.unpack_from('<3f', bytes(remote), 4); (ly,) = struct.unpack_from('<f', bytes(remote), 20)
    return tuple(float(v) if math.isfinite(v) else 0. for v in (lx, ly, rx, ry))


def remote_bytes(*pressed, sticks=(0., 0., 0., 0.)):
    """Inverse of remote_buttons / remote_sticks (tests, simulated robot); sticks = (lx, ly, rx, ry)."""
    out = bytearray(40)
    for name in pressed:
        byte, bit = _BUTTONS[name]; out[byte] |= 1 << bit
    lx, ly, rx, ry = sticks
    struct.pack_into('<3f', out, 4, lx, rx, ry); struct.pack_into('<f', out, 20, ly)
    return bytes(out)
