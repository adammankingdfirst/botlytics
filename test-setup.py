#!/usr/bin/env python3
"""
Test script to validate Botlytics setup
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"✅ {package_name} - Installed")
        return True
    else:
        print(f"❌ {package_name} - Not installed")
        return False

def check_gcp_auth():
    """Check GCP authentication"""
    try:
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'ACTIVE' in result.stdout:
            print("✅ GCP Authentication - OK")
            return True
        else:
            print("❌ GCP Authentication - Not configured")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ GCP CLI - Not installed or not in PATH")
        return False

def main():
    print("🔍 Botlytics Setup Validation\n")
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    
    # Required packages
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'google.cloud.storage',
        'google.cloud.aiplatform', 'matplotlib', 'streamlit'
    ]
    
    for package in required_packages:
        checks.append(check_package(package))
    
    # GCP setup
    checks.append(check_gcp_auth())
    
    # Summary
    print(f"\n📊 Results: {sum(checks)}/{len(checks)} checks passed")
    
    if all(checks):
        print("🎉 All checks passed! Ready to run Botlytics.")
        print("\nNext steps:")
        print("1. Set up your .env file with GCP credentials")
        print("2. Run: ./run-local.sh")
        print("3. Open http://localhost:8080 for API")
        print("4. Run Streamlit frontend separately")
    else:
        print("⚠️  Some checks failed. Please install missing dependencies.")
        print("\nFor backend: pip install -r backend/requirements.txt")
        print("For frontend: pip install -r frontend/requirements.txt")
        print("For GCP: Install Google Cloud SDK")

if __name__ == "__main__":
    main()