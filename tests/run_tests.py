#!/usr/bin/env python3
"""
Test runner script for the MolecuGen testing suite.

This script provides convenient ways to run different categories of tests
with appropriate configurations and reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def run_command(cmd, description=""):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run MolecuGen test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run only unit tests
  python run_tests.py --integration             # Run only integration tests
  python run_tests.py --performance             # Run only performance tests
  python run_tests.py --fast                    # Run fast tests only
  python run_tests.py --coverage                # Run with coverage report
  python run_tests.py --verbose                 # Run with verbose output
  python run_tests.py --file test_smiles_processor.py  # Run specific file
        """
    )
    
    # Test selection options
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests only')
    parser.add_argument('--fast', action='store_true',
                       help='Run fast tests only (exclude slow tests)')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU tests only')
    parser.add_argument('--rdkit', action='store_true',
                       help='Run RDKit-dependent tests only')
    
    # Test execution options
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    parser.add_argument('--parallel', '-n', type=int, metavar='N',
                       help='Run tests in parallel with N workers')
    parser.add_argument('--file', type=str, metavar='FILE',
                       help='Run specific test file')
    parser.add_argument('--test', type=str, metavar='TEST',
                       help='Run specific test function')
    
    # Output options
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML test report')
    parser.add_argument('--junit-xml', type=str, metavar='FILE',
                       help='Generate JUnit XML report')
    
    # Other options
    parser.add_argument('--install-deps', action='store_true',
                       help='Install test dependencies before running')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Change to tests directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Install dependencies if requested
    if args.install_deps:
        deps_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'pytest', 'pytest-cov', 'pytest-html', 'pytest-xdist',
            'psutil', 'numpy', 'torch', 'torch-geometric'
        ]
        if not run_command(deps_cmd, "Installing test dependencies"):
            return 1
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add test selection markers
    markers = []
    if args.unit:
        markers.append('unit')
    if args.integration:
        markers.append('integration')
    if args.performance:
        markers.append('performance')
    if args.fast:
        markers.append('not slow')
    if args.gpu:
        markers.append('gpu')
    if args.rdkit:
        markers.append('rdkit')
    
    if markers:
        cmd.extend(['-m', ' or '.join(markers)])
    
    # Add verbosity options
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')
    else:
        cmd.append('-v')  # Default to verbose
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            '--cov=src',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov'
        ])
    
    # Add HTML report
    if args.html_report:
        cmd.extend(['--html=report.html', '--self-contained-html'])
    
    # Add JUnit XML
    if args.junit_xml:
        cmd.extend(['--junit-xml', args.junit_xml])
    
    # Add specific file or test
    if args.file:
        cmd.append(args.file)
    elif args.test:
        cmd.extend(['-k', args.test])
    else:
        cmd.append('.')  # Run all tests in current directory
    
    # Add common pytest options
    cmd.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Strict marker checking
        '--disable-warnings',  # Disable warnings for cleaner output
    ])
    
    # Show command if dry run
    if args.dry_run:
        print("Would run command:")
        print(' '.join(cmd))
        return 0
    
    # Run the tests
    success = run_command(cmd, "Running tests")
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
        if args.html_report:
            print("📋 Test report generated in report.html")
    else:
        print("💥 Some tests failed!")
        print("Check the output above for details.")
    print('='*60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())