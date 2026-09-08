"""Packaging checks only. These tests never contact a robot or start ROS."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_python_syntax(self):
        for directory in ("host", "nav", "src", "scripts", "visualization", "docker"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_bytes(), filename=str(path))

    def test_shell_syntax(self):
        files = list((ROOT / "host/bin").iterdir())
        for directory in ("scripts", "nav", "visualization"):
            files.extend((ROOT / directory).rglob("*.sh"))
        for path in files:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_xml(self):
        ET.parse(ROOT / "config/cyclonedds.xml")
        ET.parse(ROOT / "nav/g1_nav2_bt.xml")
        for path in (ROOT / "src").rglob("package.xml"):
            ET.parse(path)

    def test_robot_algorithms_preserved(self):
        # The robot snapshot is authoritative. Portability changes are limited
        # to launchers / filesystem configuration, not estimator algorithms.
        for line in (ROOT / "manifests/robot_snapshot.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            protected = relative.startswith(("src/", "config/", "nav/"))
            if not protected or relative == "nav/g1_nav_planning.launch.py":
                continue
            path = ROOT / relative
            if path.suffix == ".pcd":  # upstream example data is not published
                continue
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_gateway_limits_and_nonfinite_input(self):
        spec = importlib.util.spec_from_file_location("gateway", ROOT / "host/g1_cmd_vel_bridge.py")
        gateway = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gateway)  # main() is NOT called
        self.assertEqual(gateway.MAX_VX, 0.20)
        self.assertEqual(gateway.MAX_WZ, 0.20)
        self.assertEqual(gateway.CMD_TIMEOUT_S, 0.30)
        for value in (float("nan"), float("inf"), -float("inf"), -1.0):
            self.assertEqual(gateway._finite_clamp(value, 0.0, 0.2), 0.0)
        self.assertEqual(gateway._finite_clamp(1.0, 0.0, 0.2), 0.2)

    def test_arm_requires_operator(self):
        environment = os.environ.copy()
        environment.pop("I_AM_SUPERVISING", None)
        result = subprocess.run(["bash", str(ROOT / "host/bin/g1-motion-arm")],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing to arm", result.stderr)

    def test_install_idempotent_and_preserves_existing(self):
        with tempfile.TemporaryDirectory(prefix="g1-package-test-") as temporary:
            destination = Path(temporary) / "bin"
            runtime = Path(temporary) / "runtime"
            environment = dict(os.environ, G1_BIN_DIR=str(destination), G1_SLAM_RUNTIME_DIR=str(runtime))
            command = ["bash", str(ROOT / "scripts/install_shortcuts.sh")]
            for _ in range(2):
                subprocess.run(command, env=environment, check=True, capture_output=True)
            self.assertTrue((destination / "slam-start").is_symlink())
            self.assertFalse((runtime / "nav/g1_motion_armed").exists())
            # A separate existing command must never be replaced.
            other = Path(temporary) / "existing"
            other.mkdir()
            preserved = other / "slam-start"
            preserved.write_text("existing-project-command\n")
            environment["G1_BIN_DIR"] = str(other)
            result = subprocess.run(command, env=environment, capture_output=True)
            self.assertEqual(result.returncode, 3)
            self.assertEqual(preserved.read_text(), "existing-project-command\n")
            self.assertEqual(list(other.iterdir()), [preserved])


if __name__ == "__main__":
    unittest.main()
