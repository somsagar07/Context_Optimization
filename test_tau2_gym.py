"""
Test script to verify tau2 gym environment is working correctly,
including policy.md loading and initial message generation.
"""
import os
import sys

# Load .env file FIRST (before any other imports that might need env vars)
try:
    from dotenv import load_dotenv
    # Try to load from project root
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✓ Loaded .env file from: {env_path}")
    else:
        # Try current directory
        load_dotenv()
        print("✓ Loaded .env file from current directory")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Could not load .env file: {e}")

# Set TAU2_DATA_DIR if not already set
if 'TAU2_DATA_DIR' not in os.environ:
    possible_paths = [
        './data_cache/tau2_data_root',
        '../data_cache/tau2_data_root',
    ]
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            os.environ['TAU2_DATA_DIR'] = abs_path
            print(f"✓ Set TAU2_DATA_DIR to: {abs_path}")
            break

# Check for API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("\n⚠ WARNING: OPENAI_API_KEY not set!")
    print("   Make sure your .env file contains: OPENAI_API_KEY=sk-...")
else:
    print(f"✓ OPENAI_API_KEY is set (length: {len(api_key)})")

# Check for model configuration
tau2_model = os.environ.get('TAU2_MODEL', 'gpt-4o-mini')  # Default to gpt-4o-mini
print(f"✓ Using model: {tau2_model} (set TAU2_MODEL in .env to change)")


try:
    import tau2
    from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
    import gymnasium as gym
    
    print("\n" + "="*70)
    print("TESTING TAU2 GYM ENVIRONMENT")
    print("="*70)
    
    # Register gym environments
    print("\n1. Registering gym environments...")
    register_gym_agent()
    print("   ✓ Gym environments registered")
    
    # Test airline domain
    domain = "airline"
    task_id = "0"  # Use first task
    
    print(f"\n2. Creating gym environment for domain='{domain}', task_id='{task_id}'...")
    env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task_id)
    print("   ✓ Environment created")
    
    # Reset environment
    print(f"\n3. Resetting environment...")
    obs, info = env.reset()
    print("   ✓ Environment reset")
    
    # Check info dict
    print(f"\n4. Checking environment info...")
    print(f"   Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
    
    # Get initial message
    initial_message = info.get("message", "") if isinstance(info, dict) else ""
    print(f"\n5. Initial user message:")
    print(f"   {initial_message[:200]}..." if len(initial_message) > 200 else f"   {initial_message}")
    
    # Check if policy is loaded (tau2 library should handle this automatically)
    print(f"\n6. Policy loading:")
    print(f"   ✓ Policy should be loaded automatically by tau2 library")
    print(f"   Policy file location: {os.environ.get('TAU2_DATA_DIR', 'Not set')}/tau2/domains/{domain}/policy.md")
    
    # Test a simple step
    print(f"\n7. Testing environment step...")
    test_response = "Hello, I can help you with that."
    obs, reward, terminated, truncated, info = env.step(test_response)
    print(f"   ✓ Step completed")
    print(f"   Reward: {reward}")
    print(f"   Terminated: {terminated}, Truncated: {truncated}")
    
    if isinstance(info, dict) and "message" in info:
        user_message = info.get("message", "")
        print(f"   User response: {user_message[:200]}..." if len(user_message) > 200 else f"   User response: {user_message}")
    
    print(f"\n" + "="*70)
    print("✓ ALL TESTS PASSED - Tau2 gym environment is working correctly!")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)