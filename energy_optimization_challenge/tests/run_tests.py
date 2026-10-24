"""
Test runner for the energy optimization challenge.

This script discovers and runs all tests in the package.
"""

import unittest
import sys
import os


def run_tests():
    """Discover and run all tests in the package."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover tests in the test directory
    test_suite = unittest.defaultTestLoader.discover(test_dir)
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests()) 