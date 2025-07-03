#!/usr/bin/env python3
"""
Test runner for Data Analytics Agent Swarm
Runs all unit tests with coverage reporting
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def discover_and_run_tests():
    """Discover and run all tests"""
    # Set up test discovery
    test_dir = Path(__file__).parent / "tests"
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests
    print("🔍 Discovering tests...")
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"📊 Found {test_count} test cases")
    
    # Run tests
    print("🧪 Running tests...\n")
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print results summary
    print(f"\n{'='*60}")
    print("📈 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split(chr(10))[0]}")
    
    if result.errors:
        print(f"\n⚠️  ERRORS:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2] if '\n' in traceback else traceback
            print(f"  - {test}: {error_msg}")
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n💥 SOME TESTS FAILED")
    
    return success

def run_specific_test(test_module):
    """Run a specific test module"""
    print(f"🧪 Running tests for {test_module}...")
    
    # Import and run specific test
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f'tests.{test_module}')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    except Exception as e:
        print(f"❌ Error running {test_module}: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Data Analytics Agent Swarm - Test Suite")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = discover_and_run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
