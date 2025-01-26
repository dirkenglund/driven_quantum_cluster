#!/usr/bin/env python3

import unittest
import sys
import os

def run_all_tests():
    """Run all test suites with detailed output"""
    # Get the directory containing this script
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover and load all tests from the tests directory
    suite = loader.discover(tests_dir, pattern='test_*.py')
    
    # Create test runner with verbosity=2 for detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests and get result
    result = runner.run(suite)
    
    # Return 0 if tests passed, 1 if any failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests()) 