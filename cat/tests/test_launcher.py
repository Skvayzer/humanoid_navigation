"""Exercise the actual launcher with fake Docker only; never contact a daemon."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_start_allows_armed_navigation_without_mounting_control_state(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'container.sh'
        with tempfile.TemporaryDirectory(prefix='cat-launcher-test-') as temporary:
            root = Path(temporary)
            binary = root / 'bin'
            binary.mkdir()
            # inspect: no existing container; run: print argv, no Docker call.
            sudo = binary / 'sudo'
            sudo.write_text('#!/bin/bash\n'
                            '[[ "$1" == docker ]] || exit 90\n'
                            'case "$2" in\n'
                            'inspect) exit 1 ;;\n'
                            'run) printf "%s\\n" "$@" ;;\n'
                            '*) exit 91 ;;\n'
                            'esac\n')
            sudo.chmod(0o755)
            navigation = root / 'nav'
            navigation.mkdir()
            token = navigation / 'g1_motion_armed'
            token.touch()
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'],
                       G1_CAT_RUNTIME_DIR=str(root / 'cat_runtime'),
                       G1_NAV_RUNTIME_DIR=str(navigation))
            result = subprocess.run(['bash', str(script), 'start'], env=env,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Starting PERCEPTION ONLY', result.stdout)
            self.assertIn('--read-only', result.stdout)
            self.assertIn('--cap-drop\nALL', result.stdout)
            self.assertIn('--cpus\n2', result.stdout)
            self.assertIn('--memory\n1536m', result.stdout)
            self.assertNotIn(str(navigation), result.stdout)
            self.assertNotIn('/run/g1_nav', result.stdout)
            self.assertTrue(token.exists())


if __name__ == '__main__':
    unittest.main()
