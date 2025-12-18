import io
import signal
import math
from contextlib import redirect_stdout, redirect_stderr

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

# Safe builtins for Python executor
_SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin,
    'bool': bool, 'chr': chr, 'dict': dict, 'divmod': divmod,
    'enumerate': enumerate, 'filter': filter, 'float': float,
    'format': format, 'frozenset': frozenset, 'hex': hex,
    'int': int, 'isinstance': isinstance, 'len': len,
    'list': list, 'map': map, 'max': max, 'min': min,
    'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
    'range': range, 'repr': repr, 'reversed': reversed,
    'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
    'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
    'zip': zip, 'True': True, 'False': False, 'None': None,
    'math': math,
}

def python_executor(code: str, timeout: int = 5) -> str:
    """
    Safely execute Python code with restrictions.
    - No file I/O, no imports, no system access
    - 5 second timeout
    - Captures stdout as result
    """
    # Clean up code - remove markdown code blocks if present
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    
    if not code:
        return "Error: No code provided"
    
    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Restricted globals
    restricted_globals = {
        '__builtins__': _SAFE_BUILTINS,
        '__name__': '__main__',
    }
    local_vars = {}
    
    try:
        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals, local_vars)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        if not output and 'result' in local_vars:
            output = str(local_vars['result'])
        
        if errors:
            return f"Output: {output}\nErrors: {errors}" if output else f"Errors: {errors}"
        
        return output.strip() if output else "Code executed successfully (no output)"
        
    except TimeoutError:
        return "Error: Code execution timed out (5s limit)"
    except SyntaxError as e:
        return f"Syntax Error: {e}"
    except NameError as e:
        return f"Name Error: {e} (Note: imports are disabled for safety)"
    except Exception as e:
        return f"Execution Error: {type(e).__name__}: {e}"

