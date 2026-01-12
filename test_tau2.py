"""
Comprehensive test script for Tau2 integration with HRL framework.

Run this script to verify all components are working:
    python test_tau2_integration.py

This will test:
1. Imports and module loading
2. Tau2 tool registry initialization
3. Tau2 dataset loading
4. Environment initialization
5. Single episode execution
"""

import os
import sys
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test 1: Check all imports work"""
    print("\n" + "="*70)
    print("TEST 1: Checking Imports")
    print("="*70)
    
    try:
        print("  ✓ Importing tau2...")
        import tau2
        print(f"  ✓ tau2 imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import tau2: {e}")
        print("  → Install tau2: pip install tau2-bench")
        return False
    
    try:
        print("  ✓ Importing Tau2ToolRegistry...")
        from tools.tau2_tool_registry import Tau2ToolRegistry
        print("  ✓ Tau2ToolRegistry imported")
    except ImportError as e:
        print(f"  ✗ Failed to import Tau2ToolRegistry: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing Tau2ExecutionWrapper...")
        # Try different import paths
        try:
            from tau2_executions_wrapper import Tau2ExecutionWrapper
        except ImportError:
            # Try relative import
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from tau2_executions_wrapper import Tau2ExecutionWrapper
        print("  ✓ Tau2ExecutionWrapper imported")
    except ImportError as e:
        print(f"  ✗ Failed to import Tau2ExecutionWrapper: {e}")
        print("  → Make sure tau2_executions_wrapper.py is in the root directory")
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing Tau2Dataset...")
        from utils.data_loader.tau2_loader import Tau2Dataset
        print("  ✓ Tau2Dataset imported")
    except ImportError as e:
        print(f"  ✗ Failed to import Tau2Dataset: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing environments...")
        from env.structure_env import StructureEnv
        from env.prompt_env import PromptEnv
        print("  ✓ Environments imported")
    except ImportError as e:
        print(f"  ✗ Failed to import environments: {e}")
        traceback.print_exc()
        return False
    
    print("  ✅ All imports successful!")
    return True


def test_tau2_data_dir():
    """Test 2: Check TAU2_DATA_DIR is set"""
    print("\n" + "="*70)
    print("TEST 2: Checking TAU2_DATA_DIR")
    print("="*70)
    
    tau2_data_dir = os.environ.get('TAU2_DATA_DIR')
    if tau2_data_dir:
        print(f"  ✓ TAU2_DATA_DIR is set: {tau2_data_dir}")
        if os.path.exists(tau2_data_dir):
            print(f"  ✓ Directory exists")
            return True
        else:
            print(f"  ✗ Directory does not exist: {tau2_data_dir}")
            return False
    else:
        print("  ⚠ TAU2_DATA_DIR not set, will try to auto-detect...")
        # Check common locations
        possible_paths = [
            './data_cache/tau2_data_root',
            '../data_cache/tau2_data_root',
        ]
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                print(f"  ✓ Found data directory: {abs_path}")
                os.environ['TAU2_DATA_DIR'] = abs_path
                return True
        print("  ✗ Could not find tau2 data directory")
        print("  → Set TAU2_DATA_DIR environment variable or place data in ./tau2_data_root")
        return False


def test_tau2_tool_registry():
    """Test 3: Initialize Tau2ToolRegistry"""
    print("\n" + "="*70)
    print("TEST 3: Testing Tau2ToolRegistry")
    print("="*70)
    
    try:
        from tools.tau2_tool_registry import Tau2ToolRegistry
        
        print("  ✓ Testing airline domain...")
        airline_tools = Tau2ToolRegistry("airline")
        tool_count = airline_tools.get_tool_count()
        tool_names = airline_tools.get_tool_names()
        print(f"  ✓ Airline domain: {tool_count} tools")
        if tool_names:
            print(f"    Tools: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}")
        else:
            print("    ⚠ No tools found (may need to configure manually)")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize Tau2ToolRegistry: {e}")
        traceback.print_exc()
        return False


def test_tau2_dataset():
    """Test 4: Load Tau2Dataset"""
    print("\n" + "="*70)
    print("TEST 4: Testing Tau2Dataset")
    print("="*70)
    
    try:
        from utils.data_loader.tau2_loader import Tau2Dataset
        
        print("  ✓ Loading airline dataset...")
        dataset = Tau2Dataset(domain="airline", split="test")
        print(f"  ✓ Loaded {len(dataset.tasks)} tasks")
        
        print("  ✓ Getting sample...")
        question, task = dataset.get_sample()
        print(f"  ✓ Got sample: task_id={task.get('task_id', 'Unknown')}")
        print(f"    Question length: {len(question)} chars")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to load Tau2Dataset: {e}")
        traceback.print_exc()
        return False


def test_get_dataset_loader():
    """Test 5: Test get_dataset_loader with tau2"""
    print("\n" + "="*70)
    print("TEST 5: Testing get_dataset_loader")
    print("="*70)
    
    try:
        from utils import get_dataset_loader
        
        print("  ✓ Testing tau2_airline...")
        dataset = get_dataset_loader("tau2_airline", is_eval=False)
        print(f"  ✓ Loaded dataset: {dataset.name}")
        print(f"    Domain: {dataset.domain}")
        print(f"    Tasks: {len(dataset.tasks)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to load dataset via get_dataset_loader: {e}")
        traceback.print_exc()
        return False


def test_structure_env():
    """Test 6: Initialize StructureEnv with tau2"""
    print("\n" + "="*70)
    print("TEST 6: Testing StructureEnv with tau2")
    print("="*70)
    
    try:
        from configs import load_config
        from env.structure_env import StructureEnv
        
        print("  ✓ Loading config...")
        cfg = load_config("hierarchical")
        cfg.DATASET_NAME = "tau2_airline"
        
        print("  ✓ Initializing StructureEnv...")
        env = StructureEnv(cfg, is_eval=True)
        print(f"  ✓ StructureEnv initialized")
        print(f"    Is tau2: {env.is_tau2}")
        print(f"    Action space: {env.action_space}")
        
        print("  ✓ Testing reset...")
        obs, info = env.reset()
        print(f"  ✓ Reset successful")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Question: {info.get('question', '')[:100]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize StructureEnv: {e}")
        traceback.print_exc()
        return False


def test_prompt_env():
    """Test 7: Initialize PromptEnv with tau2"""
    print("\n" + "="*70)
    print("TEST 7: Testing PromptEnv with tau2")
    print("="*70)
    
    try:
        from configs import load_config
        from env.prompt_env import PromptEnv
        
        print("  ✓ Loading config...")
        cfg = load_config("hierarchical")
        cfg.DATASET_NAME = "tau2_airline"
        
        print("  ✓ Initializing PromptEnv...")
        env = PromptEnv(cfg, is_eval=True)
        print(f"  ✓ PromptEnv initialized")
        print(f"    Is tau2: {env.is_tau2}")
        print(f"    Has tau2_executor: {env.tau2_executor is not None}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize PromptEnv: {e}")
        traceback.print_exc()
        return False


def test_single_episode():
    """Test 8: Run a single episode (without actual LLM calls)"""
    print("\n" + "="*70)
    print("TEST 8: Testing Single Episode (Structure + Prompt)")
    print("="*70)
    
    try:
        from configs import load_config
        from env.structure_env import StructureEnv
        from env.prompt_env import PromptEnv
        
        print("  ✓ Loading config...")
        cfg = load_config("hierarchical")
        cfg.DATASET_NAME = "tau2_airline"
        
        print("  ✓ Initializing environments...")
        struct_env = StructureEnv(cfg, is_eval=True)
        prompt_env = PromptEnv(cfg, is_eval=True)
        
        print("  ✓ Resetting structure env...")
        struct_obs, struct_info = struct_env.reset()
        print(f"    Got question and task")
        
        print("  ✓ Setting up prompt env with structure...")
        # Set structure manually for testing
        prompt_env.set_structure(
            question=struct_info["question"],
            answer=struct_info["answer"],
            embedding=struct_env.question_embedding,
            structure={
                "workflow_depth": 0,  # Direct workflow
                "agent1_tools_idx": 1,
                "agent1_budget_idx": 1,
                "agent2_tools_idx": 0,
                "agent2_budget_idx": 0,
                "answerer_budget_idx": 1,
            }
        )
        
        print("  ✓ Resetting prompt env...")
        prompt_obs, _ = prompt_env.reset()
        print(f"    Observation shape: {prompt_obs.shape}")
        
        print("  ✓ Episode setup complete!")
        print("  ⚠ Note: Actual execution requires LLM calls and will be slow")
        print("  → To test full execution, run: python train.py --dataset tau2_airline --episodes 1")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to run episode: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TAU2 INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("TAU2_DATA_DIR", test_tau2_data_dir),
        ("Tau2ToolRegistry", test_tau2_tool_registry),
        ("Tau2Dataset", test_tau2_dataset),
        ("get_dataset_loader", test_get_dataset_loader),
        ("StructureEnv", test_structure_env),
        ("PromptEnv", test_prompt_env),
        ("Single Episode", test_single_episode),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Integration is working correctly.")
        print("\n  Next steps:")
        print("    1. Run training: python train.py --dataset tau2_airline --episodes 10")
        print("    2. Or test with a different domain: python train.py --dataset tau2_retail --episodes 10")
    else:
        print("\n  ⚠ Some tests failed. Please fix the issues above before running training.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)