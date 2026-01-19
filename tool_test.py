"""
Test script to verify Tau2 tool execution is working correctly.
Tests tool registry initialization, environment connection, and actual tool execution.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Set TAU2_DATA_DIR
if 'TAU2_DATA_DIR' not in os.environ:
    possible_paths = [
        './data_cache/tau2_data_root',
        '../data_cache/tau2_data_root',
    ]
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            os.environ['TAU2_DATA_DIR'] = abs_path
            break

# Suppress verbose logging
os.environ.setdefault("LOGURU_LEVEL", "WARNING")

print("="*70)
print("TESTING TAU2 TOOL EXECUTION")
print("="*70)

try:
    import gymnasium as gym
    from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
    from tools.tau2_tool_registry import Tau2ToolRegistry
    from utils.data_loader.tau2_loader import Tau2Dataset
    
    # Test 1: Initialize tool registry
    print("\n1. Initializing Tau2ToolRegistry...")
    domain = "airline"
    tool_registry = Tau2ToolRegistry(domain)
    tool_count = tool_registry.get_tool_count()
    tool_names = tool_registry.get_tool_names()
    print(f"   ✓ Loaded {tool_count} tools: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}")
    
    # Test 2: Register gym and create environment
    print("\n2. Creating Tau2 gym environment...")
    register_gym_agent()
    task_id = "36"  # Use task 36 from your log (has known parameters)
    env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task_id)
    obs, info = env.reset()
    print(f"   ✓ Environment created and reset for task {task_id}")
    
    # Test 3: Connect environment to tool registry
    print("\n3. Connecting environment to tool registry...")
    tool_registry.set_env(env)
    print(f"   ✓ Environment connected")
    print(f"   Tool kit available: {tool_registry._tool_kit is not None}")
    print(f"   Environment available: {tool_registry.env is not None}")
    
    # Test 4: Load task data to get correct parameters
    print("\n4. Loading task data for correct parameters...")
    dataset = Tau2Dataset(domain=domain, split="train")
    task_data = None
    for task in dataset.tasks:
        if task.get('task_id') == task_id:
            task_data = task
            break
    
    if task_data:
        print(f"   ✓ Found task {task_id}")
        print(f"   User goal: {task_data.get('user_goal', 'N/A')[:80]}...")
        if 'actions' in task_data and task_data['actions']:
            expected_action = task_data['actions'][0]
            print(f"   Expected action: {expected_action.get('name', 'N/A')}")
            print(f"   Expected args: {expected_action.get('arguments', {})}")
    else:
        print(f"   ⚠ Task {task_id} not found in dataset")
    
    # Test 5: Test tool execution with known good parameters
    print("\n5. Testing tool execution...")
    
    # Test get_reservation_details with correct reservation_id from task 36
    test_tools = [
        ("get_reservation_details", '{"reservation_id": "EUJUY6"}'),
        ("get_user_details", '{"user_id": "lucas_brown_4047"}'),
    ]
    
    for tool_name, query in test_tools:
        if tool_name not in tool_names:
            print(f"   ⚠ Tool {tool_name} not in registry, skipping")
            continue
            
        print(f"\n   Testing {tool_name}...")
        print(f"   Query: {query}")
        try:
            result = tool_registry.execute(tool_name, query)
            print(f"   Result type: {type(result)}")
            print(f"   Result length: {len(str(result))} chars")
            print(f"   Result preview: {str(result)[:200]}...")
            
            # Check if result looks like an error
            if "not available" in str(result).lower() or "error" in str(result).lower():
                print(f"   ⚠ WARNING: Result looks like an error message!")
            else:
                print(f"   ✓ Tool executed successfully")
                
        except Exception as e:
            print(f"   ✗ ERROR executing tool: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 6: Check environment's tool access directly
    print("\n6. Testing direct environment tool access...")
    try:
        if hasattr(env, 'tools'):
            print(f"   ✓ Environment has 'tools' attribute")
            if hasattr(env.tools, 'tools'):
                env_tools = env.tools.tools
                print(f"   ✓ Environment tools dict has {len(env_tools)} tools")
                if 'get_reservation_details' in env_tools:
                    print(f"   ✓ 'get_reservation_details' found in environment")
                    # Try direct call
                    try:
                        direct_result = env_tools['get_reservation_details']('{"reservation_id": "EUJUY6"}')
                        print(f"   ✓ Direct tool call succeeded")
                        print(f"   Direct result type: {type(direct_result)}")
                        print(f"   Direct result preview: {str(direct_result)[:200]}...")
                    except Exception as e:
                        print(f"   ✗ Direct tool call failed: {e}")
        else:
            print(f"   ⚠ Environment does not have 'tools' attribute")
    except Exception as e:
        print(f"   ⚠ Could not check environment tools: {e}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  - Tool registry initialized: ✓")
    print(f"  - Environment created: ✓")
    print(f"  - Tools connected: {'✓' if tool_registry._tool_kit or tool_registry.env else '✗'}")
    print(f"  - Tool execution: Check results above")
    
except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)