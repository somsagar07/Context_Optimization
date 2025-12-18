"""Test script for all tools."""

from .registry import ToolRegistry

if __name__ == "__main__":
    print("=" * 50)
    print("TOOL REGISTRY TESTS")
    print("=" * 50)
    
    registry = ToolRegistry()
    
    # Test 1: Calculator
    print("\n[TEST 1] Calculator")
    print("-" * 30)
    tests_calc = ["2 + 2", "100 * 50", "(10 + 5) * 3", "2^10"]
    for expr in tests_calc:
        result = registry.execute("calculator", expr)
        print(f"  {expr} = {result}")
    
    # Test 2: Python Executor - Basic
    print("\n[TEST 2] Python Executor - Basic")
    print("-" * 30)
    code1 = "print(2 + 2)"
    print(f"  Code: {code1}")
    print(f"  Output: {registry.execute('python', code1)}")
    
    # Test 3: Python Executor - Loop
    print("\n[TEST 3] Python Executor - Loop")
    print("-" * 30)
    code2 = "for i in range(5): print(i)"
    print(f"  Code: {code2}")
    print(f"  Output: {registry.execute('python', code2)}")
    
    # Test 4: Python Executor - Math
    print("\n[TEST 4] Python Executor - Math Functions")
    print("-" * 30)
    code3 = "print(f'sqrt(16) = {math.sqrt(16)}')"
    print(f"  Code: math.sqrt(16)")
    print(f"  Output: {registry.execute('python', code3)}")
    
    # Test 5: Python Executor - Security
    print("\n[TEST 5] Python Executor - Security (blocked import)")
    print("-" * 30)
    code4 = "import os; print(os.getcwd())"
    print(f"  Code: import os")
    print(f"  Output: {registry.execute('python', code4)}")
    
    # Test 6: Web Search
    print("\n[TEST 6] Web Search")
    print("-" * 30)
    try:
        result = registry.execute("web_search", "Python programming")
        print(f"  Query: 'Python programming'")
        print(f"  Output: {result[:100]}..." if len(result) > 100 else f"  Output: {result}")
    except Exception as e:
        print(f"  Skipped: {e}")
    
    # Test 7: Tool listing
    print("\n[TEST 7] Available Tools")
    print("-" * 30)
    print(f"  Tools: {registry.list_tools()}")
    


