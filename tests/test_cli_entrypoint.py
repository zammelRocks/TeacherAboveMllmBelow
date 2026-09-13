"""Regression test: `python -m kinematics_grading ...` failed with
"No module named kinematics_grading.__main__" until __main__.py was added --
pinning it down so packaging changes can't silently drop it again.
"""

import subprocess
import sys


def test_module_invocation_does_not_error_on_missing_main():
    result = subprocess.run(
        [sys.executable, "-m", "kinematics_grading", "generate", "--help"],
        capture_output=True,
        text=True,
    )
    assert "No module named" not in result.stderr
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
