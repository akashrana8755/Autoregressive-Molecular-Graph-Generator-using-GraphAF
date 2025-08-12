#!/usr/bin/env python3
"""Validation script to check project setup."""

import os
import sys
from pathlib import Path

def check_directory_structure():
    """Check if all required directories exist."""
    required_dirs = [
        "src",
        "src/data",
        "src/models", 
        "src/training",
        "src/generate",
        "src/evaluate",
        "config",
        "scripts",
        "tests"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def check_required_files():
    """Check if all required files exist."""
    required_files = [
        "requirements.txt",
        "setup.py",
        "README.md",
        ".gitignore",
        "src/__init__.py",
        "src/config.py",
        "config/graphdiff_default.yaml",
        "config/graphaf_default.yaml",
        "config/test_config.yaml",
        "tests/test_config.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def check_config_files():
    """Check if configuration files are valid YAML."""
    config_files = [
        "config/graphdiff_default.yaml",
        "config/graphaf_default.yaml", 
        "config/test_config.yaml"
    ]
    
    try:
        import yaml
        yaml_available = True
    except ImportError:
        print("⚠️  PyYAML not installed, skipping YAML validation")
        yaml_available = False
    
    if not yaml_available:
        return True
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print(f"✅ {config_file} is valid YAML")
        except Exception as e:
            print(f"❌ {config_file} has YAML errors: {e}")
            return False
    
    return True

def main():
    """Run all validation checks."""
    print("🔍 Validating MolecuGen project setup...")
    print()
    
    checks = [
        ("Directory structure", check_directory_structure),
        ("Required files", check_required_files),
        ("Configuration files", check_config_files)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"Checking {check_name}...")
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All validation checks passed!")
        print("📦 Project structure is set up correctly")
        print("🚀 Ready to install dependencies with: pip install -r requirements.txt")
    else:
        print("❌ Some validation checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()