#!/usr/bin/env python3
"""
Setup Verification Script
Checks that all components are properly installed and configured.
"""

import sys
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required packages are available"""
    
    required_packages = [
        "ray", "torch", "gymnasium", "pettingzoo", 
        "pygame", "numpy", "yaml", "matplotlib"
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - NOT FOUND")
    
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages found!")
    return True

def check_directory_structure():
    """Verify directory structure is complete"""
    
    required_dirs = [
        "src/environments", "src/agents", "src/training",
        "checkpoints/high_level", "checkpoints/low_level",
        "configs", "logs"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
            print(f"❌ {dir_path}")
        else:
            print(f"✅ {dir_path}")
    
    if missing:
        print(f"\n⚠️  Missing directories: {missing}")
        return False
        
    print("\n✅ Directory structure complete!")
    return True

def check_config_files():
    """Verify configuration files exist"""
    
    config_files = [
        "configs/training_config.yaml",
        "configs/environment_config.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file}")
            return False
    
    print("\n✅ Configuration files complete!")
    return True

def main():
    """Run complete verification"""
    print("🔍 Verifying Hierarchical Air Hockey Setup")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure), 
        ("Configuration Files", check_config_files)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 SETUP VERIFICATION PASSED!")
        print("Ready to begin implementation.")
    else:
        print("❌ SETUP VERIFICATION FAILED!")
        print("Please fix the issues above before continuing.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
