import ctypes
import os
import platform
import struct
import unittest
from pathlib import Path

import numpy as np

from g1_wholebody import lowlevel as ll

SDK_LIB = Path(os.environ.get('G1_SDK_CRC_DIR') or Path(__file__).resolve().parents[3] / 'third_party/unitree_sdk2_python/unitree_sdk2py/utils/lib')


class LowLevelTest(unittest.TestCase):
    def test_crc_matches_unitree_sdk_library(self):
        lib = {'x86_64': 'crc_amd64.so', 'aarch64': 'crc_aarch64.so'}.get(platform.machine())
        if lib is None or not (SDK_LIB / lib).exists():
            self.skipTest('Unitree SDK CRC library not available here (checked on the laptop checkout)')
        crc_lib = ctypes.CDLL(str(SDK_LIB / lib))
        crc_lib.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
        crc_lib.crc32_core.restype = ctypes.c_uint32
        rng = np.random.default_rng(0)
        for trial in range(50):
            cmd = ll.LowCmd.body(rng.integers(0, 255), rng.normal(size=29), rng.uniform(0, 300, 29), rng.uniform(0, 10, 29),
                                 tau=rng.normal(size=29))
            packed = cmd.pack(crc=0)
            words = struct.unpack('<%dI' % (len(packed) // 4 - 1), packed[:-4])
            reference = crc_lib.crc32_core((ctypes.c_uint32 * len(words))(*words), len(words))
            self.assertEqual(cmd.crc, reference)
        self.assertEqual(len(cmd.pack()), 1004)

    def test_fast_and_python_crc_agree(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            cmd = ll.LowCmd.body(5, rng.normal(size=29), rng.uniform(0, 300, 29), rng.uniform(0, 10, 29), tau=rng.normal(size=29))
            packed = cmd.pack(crc=0)
            words = struct.unpack('<%dI' % (len(packed) // 4 - 1), packed[:-4])
            self.assertEqual(ll.crc_of_packed(packed), ll.crc32_core(words))

    def test_crc_matches_sdk_bitwise_reference(self):
        # SDK utils/crc.py _crc_py, copied: the bit-wise definition.
        def crc_py(data):
            crc = 0xFFFFFFFF
            for current in data:
                bit = 1 << 31
                for _ in range(32):
                    crc = ((crc << 1) & 0xFFFFFFFF) ^ 0x04c11db7 if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
                    if current & bit:
                        crc ^= 0x04c11db7
                    bit >>= 1
            return crc
        words = list(np.random.default_rng(1).integers(0, 2**32, 40, dtype=np.uint64))
        self.assertEqual(ll.crc32_core([int(w) for w in words]), crc_py([int(w) for w in words]))

    def test_only_body_motors_enabled_and_nonfinite_rejected(self):
        cmd = ll.LowCmd.body(5, np.zeros(29), np.ones(29), np.ones(29))
        self.assertTrue((cmd.mode[:29] == 1).all() and (cmd.mode[29:] == 0).all())
        self.assertTrue((cmd.kp[29:] == 0).all())
        q = np.zeros(29); q[3] = np.nan
        with self.assertRaises(ValueError):
            ll.LowCmd.body(5, q, np.ones(29), np.ones(29))

    def test_lowstate_cdr_round_trip(self):
        rng = np.random.default_rng(2)
        q, dq, tau = rng.normal(size=29), rng.normal(size=29), rng.normal(size=29)
        quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
        raw = ll.encode_lowstate_cdr(123456, 5, q, dq, tau, quat, [0.1, 0.2, 0.3], [0, 0, 9.8], ll.remote_bytes('L1', 'A'))
        s = ll.decode_lowstate(raw)
        self.assertEqual((s.tick, s.mode_machine), (123456, 5))
        np.testing.assert_allclose(s.q, q.astype(np.float32)); np.testing.assert_allclose(s.tau, tau.astype(np.float32))
        np.testing.assert_allclose(s.quat_wxyz, quat.astype(np.float32))
        self.assertEqual(ll.remote_buttons(s.remote), {'L1', 'A'})

    def test_cdr_layout_motor0_follows_int16(self):
        L = ll.LOWSTATE_LAYOUT
        self.assertEqual(L['imu.temperature'][0], 68)
        self.assertEqual(L['m.mode'][0], 70)          # CDR: no struct padding (C struct would put it at 72)
        self.assertEqual(L['m.q'][0], 72)
        self.assertEqual(L['m.mode'][1] - L['m.mode'][0], 54)
        self.assertEqual(L['m.mode'][2] - L['m.mode'][1], 56)

    def test_remote_layout_matches_sdk_joystick(self):
        # joystick.py: LT = bin(byte2)[2], RT = [3], back = [4], start = [5], LB = [6], RB = [7]; byte3: left..A
        def sdk(remote):
            b1 = [int(c) for c in f'{remote[2]:08b}']; b2 = [int(c) for c in f'{remote[3]:08b}']
            names = dict(L2=b1[2], R2=b1[3], select=b1[4], start=b1[5], L1=b1[6], R1=b1[7],
                         left=b2[0], down=b2[1], right=b2[2], up=b2[3], Y=b2[4], X=b2[5], B=b2[6], A=b2[7])
            return {k for k, v in names.items() if v}
        for byte2 in range(256):
            for byte3 in (0, 1, 2, 4, 128, 255):
                remote = bytes([0, 0, byte2, byte3]) + bytes(36)
                self.assertEqual(ll.remote_buttons(remote), sdk(remote))


if __name__ == '__main__':
    unittest.main()
