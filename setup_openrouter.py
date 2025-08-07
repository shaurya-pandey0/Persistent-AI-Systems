#!/usr/bin/env python3
"""
OpenRouter Setup and Test Script
===============================

This script helps you set up and test OpenRouter API authentication
to fix the 401 "No auth credentials found" error.

Usage:
    python setup_openrouter.py           # Interactive setup
    python setup_openrouter.py test      # Test existing setup
    python setup_openrouter.py fix       # Apply all fixes automatically
"""

import os
import sys
from pathlib import Path
import subprocess

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🔧 {title}")
    print("="*60)

def print_step(step: str, description: str):
    """Print a formatted step"""
    print(f"\n{step} {description}")
    print("-" * 40)

def check_environment():
    """Check if OpenRouter environment is properly configured"""
    print_header("Environment Check")
    
    issues = []
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        issues.append("missing_api_key")
    elif not api_key.startswith("sk-or-v1-"):
        print(f"⚠️  API key format may be incorrect: '{api_key[:15]}...'")
        print("   OpenRouter keys should start with 'sk-or-v1-'")
        issues.append("invalid_api_key_format")
    else:
        print(f"✅ OPENROUTER_API_KEY found: '{api_key[:15]}...'")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if "OPENROUTER_API_KEY" in content:
                    print("✅ .env file contains OPENROUTER_API_KEY")
                else:
                    print("⚠️  .env file exists but doesn't contain OPENROUTER_API_KEY")
        except Exception as e:
            print(f"⚠️  Could not read .env file: {e}")
    else:
        print("ℹ️  No .env file found (optional)")
    
    # Check required files
    api_client_path = Path("core/api_client.py")
    if api_client_path.exists():
        print("✅ core/api_client.py exists")
        
        # Check if it's the fixed version
        try:
            with open(api_client_path, 'r') as f:
                content = f.read()
                if "OpenRouterClient" in content and "Bearer" in content:
                    print("✅ API client appears to be fixed version")
                else:
                    print("❌ API client needs to be updated with fixed version")
                    issues.append("outdated_api_client")
        except Exception as e:
            print(f"⚠️  Could not read api_client.py: {e}")
    else:
        print("❌ core/api_client.py not found")
        issues.append("missing_api_client")
    
    # Check dependencies
    try:
        import requests
        print("✅ requests library available")
    except ImportError:
        print("❌ requests library not installed")
        issues.append("missing_requests")
    
    return issues

def setup_environment():
    """Interactive environment setup"""
    print_header("OpenRouter Environment Setup")
    
    print("This will help you configure OpenRouter API authentication.")
    print()
    
    # Step 1: Get API key
    print_step("Step 1:", "OpenRouter API Key")
    api_key = input("Enter your OpenRouter API key (starts with 'sk-or-v1-'): ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    if not api_key.startswith("sk-or-v1-"):
        confirm = input("⚠️  API key doesn't start with 'sk-or-v1-'. Continue anyway? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Step 2: Create .env file
    print_step("Step 2:", "Creating .env file")
    env_content = f"OPENROUTER_API_KEY={api_key}\n"
    
    env_file = Path(".env")
    if env_file.exists():
        backup = input("📁 .env file exists. Create backup? (Y/n): ")
        if backup.lower() != 'n':
            backup_file = Path(".env.backup")
            env_file.rename(backup_file)
            print(f"✅ Backup created: {backup_file}")
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created .env file with API key")
        
        # Set environment variable for current session
        os.environ["OPENROUTER_API_KEY"] = api_key
        print("✅ Environment variable set for current session")
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False
    
    # Step 3: Install dependencies
    print_step("Step 3:", "Installing dependencies")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "python-dotenv"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to install dependencies: {e}")
        print("   Please run manually: pip install requests python-dotenv")
    
    return True

def test_connection():
    """Test OpenRouter API connection"""
    print_header("Testing OpenRouter Connection")
    
    try:
        # Import and test
        sys.path.insert(0, '.')
        from core.api_client import test_openrouter_connection
        
        print("🧪 Running connection test...")
        success = test_openrouter_connection()
        
        if success:
            print("\n🎉 SUCCESS: OpenRouter connection working!")
            print("✅ Your authentication is properly configured")
            print("✅ You can now run your Streamlit app: streamlit run app.py")
            return True
        else:
            print("\n❌ FAILED: OpenRouter connection failed")
            print("💡 Check the error messages above for troubleshooting steps")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure core/api_client.py exists and is the fixed version")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def apply_fixes():
    """Apply all fixes automatically"""
    print_header("Applying OpenRouter Fixes")
    
    issues = check_environment()
    
    if not issues:
        print("✅ No issues found! Running connection test...")
        return test_connection()
    
    print(f"Found {len(issues)} issues to fix:")
    for issue in issues:
        print(f"  • {issue}")
    
    print()
    
    # Fix missing dependencies
    if "missing_requests" in issues:
        print_step("Fix 1:", "Installing missing dependencies")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "requests", "python-dotenv"], 
                          check=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
    
    # Fix API key issues
    if "missing_api_key" in issues or "invalid_api_key_format" in issues:
        print_step("Fix 2:", "API Key Configuration")
        print("❌ API key issue detected")
        print("💡 Please run: python setup_openrouter.py")
        print("   This will guide you through API key setup")
        return False
    
    # Fix API client
    if "outdated_api_client" in issues or "missing_api_client" in issues:
        print_step("Fix 3:", "API Client Update Required")
        print("❌ API client needs to be updated")
        print("💡 Please replace core/api_client.py with the fixed version")
        print("   cp api_client_fixed.py core/api_client.py")
        return False
    
    # Test after fixes
    print_step("Final Step:", "Testing Connection")
    return test_connection()

def show_help():
    """Show usage help"""
    print("🔧 OpenRouter Setup Script")
    print("=" * 30)
    print()
    print("Commands:")
    print("  python setup_openrouter.py        - Interactive setup")
    print("  python setup_openrouter.py test   - Test existing setup")
    print("  python setup_openrouter.py fix    - Apply fixes automatically")
    print("  python setup_openrouter.py check  - Check environment only")
    print("  python setup_openrouter.py help   - Show this help")
    print()
    print("Quick Start:")
    print("1. Get your API key: https://openrouter.ai/keys")
    print("2. Add credits: https://openrouter.ai/account")
    print("3. Run: python setup_openrouter.py")
    print("4. Test: python setup_openrouter.py test")
    print("5. Start app: streamlit run app.py")

def main():
    """Main function"""
    if len(sys.argv) == 1:
        # Interactive setup
        if setup_environment():
            test_connection()
    
    elif len(sys.argv) == 2:
        command = sys.argv[1].lower()
        
        if command == "test":
            test_connection()
        elif command == "fix":
            apply_fixes()
        elif command == "check":
            issues = check_environment()
            if not issues:
                print("\n✅ All checks passed!")
            else:
                print(f"\n❌ Found {len(issues)} issues")
        elif command == "help":
            show_help()
        else:
            print(f"Unknown command: {command}")
            show_help()
    else:
        show_help()

if __name__ == "__main__":
    main()