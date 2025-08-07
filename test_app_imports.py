#!/usr/bin/env python3
"""
COMPREHENSIVE IMPORT TEST FOR APP.PY - FINAL FIXED VERSION
===========================================================
This test file checks all imports, functions, and files referenced in app.py
Tests are ordered starting from app.py as pivot point
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

# Test the main app import first
try:
    from app import *  # triggers every import transitively
    print("✅ app.py imported successfully")
except Exception as e:
    print(f"❌ Failed to import app.py: {e}")
    sys.exit(1)

class ImportTester:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        self.test_count = 0

    def test_simple_import(self, module_name, test_description):
        """Test a simple module import"""
        self.test_count += 1
        print(f"\n{self.test_count:2d}. Testing {test_description}")
        print(f"    Module: {module_name}")

        try:
            if ' as ' in module_name:
                # Handle import X as Y
                real_name, alias = module_name.split(' as ')
                exec(f"import {real_name.strip()} as {alias.strip()}")
            else:
                # Simple import
                __import__(module_name)

            print(f"    ✅ SUCCESS: {module_name}")
            self.results['passed'].append(f"{test_description}: {module_name}")
            return True

        except ImportError as e:
            print(f"    ❌ IMPORT ERROR: {e}")
            self.results['failed'].append(f"{test_description}: {module_name} - {e}")
            return False
        except Exception as e:
            print(f"    ⚠️  OTHER ERROR: {e}")
            self.results['failed'].append(f"{test_description}: {module_name} - {e}")
            return False

    def test_function_import(self, module_name, function_name, test_description):
        """Test importing a specific function from a module"""
        self.test_count += 1
        print(f"\n{self.test_count:2d}. Testing {test_description}")
        print(f"    Function: {function_name} from {module_name}")

        try:
            # Create a safe namespace for testing
            test_namespace = {}
            exec(f"from {module_name} import {function_name}", test_namespace)
            
            # Check if function is callable
            if callable(test_namespace[function_name]):
                print(f"    ✅ SUCCESS: {function_name} from {module_name}")
                self.results['passed'].append(f"{test_description}: {function_name}")
                return True
            else:
                print(f"    ⚠️  WARNING: {function_name} is not callable")
                self.results['warnings'].append(f"{test_description}: {function_name} - not callable")
                return False

        except ImportError as e:
            print(f"    ❌ IMPORT ERROR: {e}")
            self.results['failed'].append(f"{test_description}: {function_name} - {e}")
            return False
        except Exception as e:
            print(f"    ⚠️  OTHER ERROR: {e}")
            self.results['failed'].append(f"{test_description}: {function_name} - {e}")
            return False

    def print_summary(self):
        """Print test summary"""
        total = len(self.results['passed']) + len(self.results['failed'])
        passed = len(self.results['passed'])
        failed = len(self.results['failed'])

        print(f"\n" + "="*60)
        print(f"IMPORT TEST SUMMARY")
        print(f"="*60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🎯 Success Rate: {(passed/total*100):.1f}%" if total > 0 else "🎯 Success Rate: 0%")

        if self.results['failed']:
            print(f"\n❌ FAILED IMPORTS:")
            for i, failure in enumerate(self.results['failed'], 1):
                print(f"  {i}. {failure}")

        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.results['warnings'], 1):
                print(f"  {i}. {warning}")

        # Provide actionable suggestions
        if failed > 0:
            print(f"\n💡 SUGGESTIONS TO FIX FAILURES:")
            unique_modules = set()
            for failure in self.results['failed']:
                if 'from' in failure:
                    # Extract module name from "function from module" pattern
                    try:
                        module = failure.split(' from ')[1].split(' -')[0]
                        unique_modules.add(module)
                    except:
                        pass
            
            for module in unique_modules:
                print(f"  📝 Add to {module}.py: __all__ = ['function1', 'function2', ...]")

def main():
    """Main test execution"""
    print("🚀 Starting Comprehensive Import Test for APP.PY - FIXED VERSION")
    print("="*60)
    print(f"Test started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {Path.cwd()}")

    tester = ImportTester()

    # ========================================
    # PHASE 1: STANDARD LIBRARY IMPORTS
    # ========================================
    print("\n" + "="*60)
    print("PHASE 1: STANDARD LIBRARY IMPORTS")
    print("="*60)

    standard_libs = [
        'json',
        'os', 
        'pathlib',
        'sys',
        'traceback'
    ]

    for lib in standard_libs:
        tester.test_simple_import(lib, f"Standard Library: {lib}")

    # ========================================
    # PHASE 2: EXTERNAL LIBRARY IMPORTS
    # ========================================
    print("\n" + "="*60)
    print("PHASE 2: EXTERNAL LIBRARY IMPORTS")
    print("="*60)

    # Test streamlit
    tester.test_simple_import('streamlit', 'External Library: streamlit')
    tester.test_simple_import('streamlit as st', 'External Library: streamlit as st')

    # ========================================
    # PHASE 3: CUSTOM MODULE IMPORTS
    # ========================================
    print("\n" + "="*60)
    print("PHASE 3: CUSTOM MODULE IMPORTS (FROM APP.PY)")
    print("="*60)

    custom_modules = [
        'utils.session_id',
        'utils.ui_helpers',
        'core.context_block_builder',
        'core.fact_extractor',
        'memory.turn_memory',
        'memory.session_summarizer',
        'persona.mood_tracker',
        'vectorstore'
    ]

    for module in custom_modules:
        tester.test_simple_import(module, f"Custom Module: {module}")

    # ========================================
    # PHASE 4: SPECIFIC FUNCTION IMPORTS
    # ========================================
    print("\n" + "="*60)
    print("PHASE 4: SPECIFIC FUNCTION IMPORTS")
    print("="*60)

    function_imports = [
        ('utils.session_id', 'get_or_create_session_file'),
        ('utils.session_id', 'save_turn_to_session'),
        ('utils.ui_helpers', 'render_message'),
        ('core.context_block_builder', 'build_session_init_prompt'),
        ('core.context_block_builder', 'build_turn_prompt'), 
        ('core.fact_extractor', 'store_fact'),
        ('core.fact_extractor', 'load_facts'),
        ('memory.turn_memory', 'dump_turn'),
        ('memory.turn_memory', 'load_memory'),
        ('memory.session_summarizer', 'summarize_session'),
        ('persona.mood_tracker', 'apply_sentiment_to_mood'),
        ('persona.mood_tracker', 'get_current_mood'),
        ('persona.mood_tracker', 'update_mood'),
        ('vectorstore', 'get_store'),
    ]

    for module, function in function_imports:
        tester.test_function_import(module, function, f"Function Import: {function} from {module}")

    # ========================================
    # PHASE 5: CONDITIONAL IMPORTS (FROM TRY BLOCKS)
    # ========================================
    print("\n" + "="*60)
    print("PHASE 5: CONDITIONAL IMPORTS (FROM TRY BLOCKS IN APP.PY)")
    print("="*60)

    conditional_imports = [
        ('persona.hormone_api', 'load_hormone_levels'),
        ('persona.hormone_api', 'save_hormone_levels'),
        ('persona.faiss_memory_writer', 'update_faiss_memory_state_from_session'),
        ('persona.tiny_model_writer', 'update_tiny_model_state_from_session'),
    ]

    for module, function in conditional_imports:
        tester.test_function_import(module, function, f"Conditional Import: {function} from {module}")

    # ========================================
    # PHASE 6: FILE STRUCTURE VERIFICATION
    # ========================================
    print("\n" + "="*60)
    print("PHASE 6: FILE STRUCTURE VERIFICATION")
    print("="*60)

    expected_files = [
        'app.py',
        'utils/__init__.py',
        'utils/session_id.py',
        'utils/ui_helpers.py',
        'core/__init__.py', 
        'core/context_block_builder.py',
        'core/fact_extractor.py',
        'memory/__init__.py',
        'memory/turn_memory.py',
        'memory/session_summarizer.py',
        'persona/__init__.py',
        'persona/mood_tracker.py',
        'persona/hormone_api.py',
        'persona/faiss_memory_writer.py',
        'persona/tiny_model_writer.py',
        'vectorstore.py',
    ]

    missing_files = []
    existing_files = []

    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            print(f"    ✅ Found: {file_path}")
            existing_files.append(file_path)
        else:
            print(f"    ❌ Missing: {file_path}")
            missing_files.append(file_path)
            tester.results['failed'].append(f"Missing file: {file_path}")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    tester.print_summary()

    if len(tester.results['failed']) == 0:
        print("\n🎉 ALL IMPORTS SUCCESSFUL! APP.PY SHOULD WORK PERFECTLY!")
        return True
    else:
        print("\n⚠️  SOME IMPORTS FAILED - CHECK THE ERRORS ABOVE")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\nExiting with code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN TEST SCRIPT: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)