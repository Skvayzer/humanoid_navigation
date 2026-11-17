#!/usr/bin/env python3
"""FAST_LIO map and matching debug viewer: the bridge between the browser page and fastlio_mapping.

  scripts/lio_viewer/server.py --replay <record.g1replay> [--scan 200] [--config config/dataset_mid360.yaml]
        [--binary <fastlio_mapping>] [--port 8765] [--trajectory <odom.tum>] [-p map_type:=ikdtree ...]

It starts `fastlio_mapping --replay ... --debug-port <p>` (src/debug_session.hpp), which runs the record
to the requested scan and stops before that scan's update, and serves the page at http://localhost:<port>.
The page reads the maps and the scan, runs the update from a perturbed prior as often as it likes, changes
the map thresholds, and steps forward through the record. Going back restarts the replay (seconds).

Stdlib only. Run it in the ROS 2 environment of the workspace that built fast_lio (its install/setup.bash
sourced: fastlio_mapping is found there), or give --binary.
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import webbrowser
from bisect import bisect_right
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent          # the fast_lio source package
STATIC = HERE / "static"


def find_binary():
    """fastlio_mapping in the sourced workspaces (AMENT_PREFIX_PATH), None if none has it."""
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        binary = Path(prefix) / "lib/fast_lio/fastlio_mapping"
        if prefix and binary.exists():
            return binary
    return None


def scan_table(replay):
    """Header stamps [s] of every cloud record of a .g1replay, in file order (scan 1 = index 0)."""
    stamps = []
    with open(replay, "rb") as f:
        magic = f.read(4)
        if magic != b"G1RP":
            raise SystemExit(f"{replay}: not a .g1replay file")
        version, meta_len = struct.unpack("<II", f.read(8))
        meta = json.loads(f.read(meta_len))
        while True:
            head = f.read(9)
            if len(head) < 9:
                break
            kind, stamp_ns = struct.unpack("<Bq", head)
            if kind == 1:
                f.seek(48, 1)
                continue
            end_ns, n, clen = struct.unpack("<qII", f.read(16))
            f.seek(clen, 1)
            stamps.append(stamp_ns * 1e-9)
    return meta, stamps


def read_tum(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            poses.append([float(x) for x in v[:8]])
    return poses


class EngineError(Exception):
    pass


class Engine:
    """One fastlio_mapping debug session (process + socket)."""

    def __init__(self, args, workdir):
        self.args = args
        self.workdir = workdir
        self.proc = None
        self.sock = None
        self.rfile = None
        self.lock = threading.RLock()
        self.scan = None          # the last "scan" header
        self.ended = None         # the "end" message when the record ran out
        self.restarts = 0
        self.log_path = workdir / "fastlio.log"
        self.last_map = None      # header of the last map reply (for voxel_flags)

    # --- process -------------------------------------------------------------------------------------
    def spawn(self, target):
        self.close()
        port = free_port()
        a = self.args
        binary = Path(a.binary).expanduser() if a.binary else find_binary()
        if binary is None or not binary.exists():
            raise EngineError("fastlio_mapping not found: source the workspace's install/setup.bash or give --binary")
        cmd = [str(binary), "--replay", str(a.replay), str(self.workdir / "odom.tum"),
               "--debug-port", str(port), "--ros-args", "-r", "__node:=rv_laser_mapping", "--params-file", str(a.config),
               "-p", f"runtime_pos_log_path:={self.workdir / 'fast_lio_time_log.csv'}"]
        for p in a.param:
            cmd += ["-p", p]
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, cwd=str(self.workdir))
        self.restarts += 1
        deadline = time.time() + 60
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                if self.proc.poll() is not None:
                    raise EngineError("fastlio_mapping exited before the session started:\n" + self.tail())
                if time.time() > deadline:
                    raise EngineError("fastlio_mapping did not open its debug port:\n" + self.tail())
                time.sleep(0.1)
        self.sock.settimeout(None)
        self.rfile = self.sock.makefile("rb")
        hello = self.read_header()
        if hello.get("type") != "hello":
            raise EngineError(f"unexpected greeting {hello}")
        self.scan = None
        self.ended = None
        self.send(f"goto {target}")
        ack = self.read_header()
        if ack.get("type") != "running":
            raise EngineError(f"goto refused: {ack}")
        self.wait_scan()

    def close(self):
        if self.sock is not None:
            try:
                self.send("quit")
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
            self.rfile = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if getattr(self, "log", None):
            self.log.close()

    def tail(self, n=30):
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # --- protocol ------------------------------------------------------------------------------------
    def send(self, line):
        self.sock.sendall((line + "\n").encode())

    def read_header(self):
        line = self.rfile.readline()
        if not line:
            self.sock = None
            raise EngineError("the debug session ended unexpectedly:\n" + self.tail())
        return json.loads(line)

    def read_reply(self):
        head = self.read_header()
        payload = b""
        if "bytes" in head:
            payload = self.rfile.read(head["bytes"])
        if head.get("type") == "error":
            raise EngineError(head.get("message", "error"))
        return head, payload

    def request(self, line):
        with self.lock:
            if self.sock is None:
                raise EngineError("no session (the record ended or the process died); choose a scan to restart")
            self.send(line)
            return self.read_reply()

    def wait_scan(self):
        """After a goto: the next reply is the scan header, or "end" if the record ran out."""
        head = self.read_header()
        if head.get("type") == "scan":
            self.scan = head
            self.last_map = None
        elif head.get("type") == "end":
            self.ended = head
            self.scan = None
            self.sock = None
            self.proc.wait(timeout=30)
        else:
            raise EngineError(f"unexpected reply {head}")

    # --- commands ------------------------------------------------------------------------------------
    def goto(self, target):
        with self.lock:
            target = max(1, int(target))
            if self.sock is not None and self.scan is not None and target > self.scan["index"]:
                self.send(f"goto {target}")
                ack = self.read_header()
                if ack.get("type") != "running":
                    raise EngineError(f"goto refused: {ack}")
                self.wait_scan()
            else:
                self.spawn(target)
            return self.status()

    def status(self):
        return {"scan": self.scan, "ended": self.ended, "restarts": self.restarts, "alive": self.sock is not None}

    def map(self, radius):
        head, payload = self.request(f"map {radius:g}")
        self.last_map = head
        return head, payload

    def update(self, perturb):
        p = [float(x) for x in perturb] + [0.0] * 6
        return self.request("update " + " ".join(f"{x:.12g}" for x in p[:6]))

    def set(self, key, value):
        head, _ = self.request(f"set {key} {value}")
        if self.scan is not None:
            self.scan["down_points"] = head["down_points"]
            self.scan["patches"] = head["patches"]
            self.scan["params"] = head["params"]
        return head


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class App:
    def __init__(self, args):
        self.args = args
        self.workdir = Path(args.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.meta, self.stamps = scan_table(args.replay)
        self.trajectory = []
        if args.trajectory:
            for t, x, y, z, *_ in read_tum(args.trajectory):
                self.trajectory.append([bisect_right(self.stamps, t), x, y, z])
        self.engine = Engine(args, self.workdir)

    def info(self):
        return {
            "replay": str(self.args.replay),
            "config": str(self.args.config),
            "params": self.args.param,
            "dataset": {k: v for k, v in self.meta.items() if k in ("name", "platform", "mount", "duration_s", "n_scans", "n_imu")},
            "n_scans": len(self.stamps),
            "stamps": self.stamps,
            "trajectory": self.trajectory,
            "status": self.engine.status(),
        }


class Handler(BaseHTTPRequestHandler):
    app: App = None

    def log_message(self, fmt, *args):
        if self.app.args.verbose:
            super().log_message(fmt, *args)

    # --- helpers -------------------------------------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_framed(self, head, payload):
        # <u32 header length><JSON header, space-padded so the payload starts 8-byte aligned><payload>
        header = json.dumps(head).encode()
        header += b" " * (-(4 + len(header)) % 8)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(4 + len(header) + len(payload)))
        self.end_headers()
        self.wfile.write(struct.pack("<I", len(header)))
        self.wfile.write(header)
        self.wfile.write(payload)

    def send_file(self, path, ctype):
        try:
            body = path.read_bytes()
        except OSError:
            self.send_json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def body_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def handle_errors(self, fn):
        try:
            fn()
        except EngineError as e:
            self.send_json({"error": str(e)}, 500)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    # --- routes --------------------------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        app, eng = self.app, self.app.engine
        if url.path == "/":
            return self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
        if url.path.startswith("/static/"):
            name = url.path[len("/static/"):]
            ctype = {"js": "text/javascript", "css": "text/css", "html": "text/html"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
            return self.send_file(STATIC / Path(name).name, ctype)
        if url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if url.path == "/api/info":
            return self.handle_errors(lambda: self.send_json(app.info()))
        if url.path == "/api/status":
            return self.handle_errors(lambda: self.send_json(eng.status()))
        if url.path == "/api/log":
            return self.send_json({"log": eng.tail(80)})
        if url.path == "/api/map":
            return self.handle_errors(lambda: self.send_framed(*eng.map(float(q.get("radius", 30)))))
        if url.path == "/api/scan":
            return self.handle_errors(lambda: self.send_framed(*eng.request("scan")))
        if url.path == "/api/voxel_flags":
            return self.handle_errors(lambda: self.send_framed(*eng.request("voxel_flags")))
        if url.path == "/api/poses":
            return self.handle_errors(lambda: self.send_framed(*eng.request("poses")))
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        eng = self.app.engine
        if url.path == "/api/goto":
            body = self.body_json()
            return self.handle_errors(lambda: self.send_json(eng.goto(body.get("scan", 1))))
        if url.path == "/api/update":
            body = self.body_json()
            return self.handle_errors(lambda: self.send_framed(*eng.update(body.get("perturb", []))))
        if url.path == "/api/set":
            body = self.body_json()
            return self.handle_errors(lambda: self.send_json(eng.set(body["key"], body["value"])))
        self.send_json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True, type=Path, help=".g1replay record (scripts/make_replay.py)")
    ap.add_argument("--config", type=Path, default=PACKAGE / "config/dataset_mid360.yaml", help="FAST_LIO params file")
    ap.add_argument("--binary", help="fastlio_mapping (default: found in the sourced workspace)")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port of the viewer")
    ap.add_argument("--scan", type=int, default=100, help="scan to start at (1-based, replay order)")
    ap.add_argument("--trajectory", type=Path, help="odom.tum of a full run of this record, shown on the map")
    ap.add_argument("--workdir", default="lio_viewer_session", help="where the session's log and odometry go")
    ap.add_argument("-p", "--param", action="append", default=[], help="FAST_LIO parameter, e.g. map_type:=ikdtree (repeatable)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--no-start", action="store_true", help="do not start the replay until the page asks for a scan")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    args.replay = args.replay.resolve()
    args.config = args.config.resolve()
    if not args.replay.exists():
        raise SystemExit(f"{args.replay} not found")

    app = App(args)
    print(f"record: {app.meta.get('name')} ({len(app.stamps)} scans); config: {args.config.name}; params: {args.param or '-'}")
    if not args.no_start:
        print(f"starting the replay to scan {args.scan} ...", flush=True)
        app.engine.goto(args.scan)
        s = app.engine.scan
        if s:
            print(f"at scan {s['index']} (t = {s['stamp']:.3f}): {s['down_points']} down points, ikd-Tree {s['ikd_size']} points, {s['voxels']} voxels")
        elif app.engine.ended:
            print(f"the record ended before scan {args.scan}: {app.engine.ended}")
    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"viewer: {url}   (Ctrl-C stops it)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.engine.close()


if __name__ == "__main__":
    main()
