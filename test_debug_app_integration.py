
# debug_app_integration.py
# Quick diagnosis script for app.py hormone integration issues

import os
import sys
import traceback
from pathlib import Path

print("🔍 APP.PY INTEGRATION DEBUGGING")
print("=" * 50)

# 1. Check working directory
print(f"1. Working Directory: {os.getcwd()}")

# 2. Check if persona directory exists
persona_dir = Path("persona")
print(f"2. Persona directory exists: {persona_dir.exists()}")
if persona_dir.exists():
    print(f"   Contents: {list(persona_dir.glob('*.py'))}")

# 3. Test individual imports
print("\n3. Testing imports:")
imports_to_test = [
    ("persona.mood_tracker", "apply_sentiment_to_mood"),
    ("persona.hormone_api", "load_hormone_levels"),
    ("persona.hormone_adjuster", "apply_contextual_hormone_adjustments"),
]

for module_name, function_name in imports_to_test:
    try:
        module = __import__(module_name, fromlist=[function_name])
        func = getattr(module, function_name)
        print(f"   ✅ {module_name}.{function_name}")
    except ImportError as e:
        print(f"   ❌ {module_name}.{function_name} - Import Error: {e}")
    except AttributeError as e:
        print(f"   ❌ {module_name}.{function_name} - Attribute Error: {e}")
    except Exception as e:
        print(f"   ❌ {module_name}.{function_name} - Other Error: {e}")

# 4. Check critical files
print("\n4. Checking critical files:")
critical_files = [
    "persona/mood_adjustments.json",
    "persona/hormone_levels.json",
    "persona/personality.json",
    "config/constants.py"
]

for file_path in critical_files:
    exists = Path(file_path).exists()
    print(f"   {'✅' if exists else '❌'} {file_path}")

# 5. Test hormone system functionality
print("\n5. Testing hormone system:")
try:
    from persona.hormone_api import load_hormone_levels, save_hormone_levels
    hormones = load_hormone_levels()
    print(f"   ✅ Hormone loading works: {hormones}")

    # Test mood processing
    from persona.mood_tracker import apply_sentiment_to_mood
    print("   ✅ Mood tracker import works")

    # Test a simple sentiment
    print("   🧪 Testing sentiment processing...")
    apply_sentiment_to_mood("I feel great today!")
    print("   ✅ Sentiment processing works!")

except Exception as e:
    print(f"   ❌ Hormone system test failed: {e}")
    print(f"   Full traceback: {traceback.format_exc()}")

print("\n" + "=" * 50)
print("🎯 DIAGNOSIS COMPLETE")
print("If all tests pass, the issue is Streamlit-specific.")
print("If tests fail, fix the failing components first.")
