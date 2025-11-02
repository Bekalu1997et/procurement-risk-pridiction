"""Run all system tests and generate a report."""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_tests():
    """Execute pytest and display results."""
    print("=" * 60)
    print("RUNNING SYSTEM TESTS")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
