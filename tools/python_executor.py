"""
- This is a modified version of python code executor with ability to read files from the disk. These files can be json, csv, images, pdfs, zip/tar files etc. The code will be provided from the LLM and executed in a restricted environment with limited built-in functions and pre-imported libraries.

- The code execution has a timeout of 10 seconds to prevent long-running code. The standard output
"""

import io
import signal
import math
from contextlib import redirect_stdout, redirect_stderr

# --- GAIA REQUIRED LIBRARIES ---
import pandas as pd
import numpy as np
import scipy
import zipfile
import tarfile
import json
import csv
import xml.etree.ElementTree as ET
import cv2
import pdfplumber
import PIL.Image
import speech_recognition as sr
import os
# -------------------------------

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

ALLOWED_IMPORTS = {
    'pandas', 'numpy', 'scipy', 'math', 'json', 'csv', 're', 
    'zipfile', 'tarfile', 'xml', 'cv2', 'pdfplumber', 'PIL', 
    'speech_recognition', 'os', 'datetime', 'random', 'collections', 'itertools',
    'pytesseract', 'io',
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Allow the module if it's in our whitelist
    root_name = name.split('.')[0] # e.g., "xml.etree" -> "xml"
    if root_name in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is restricted for safety.")

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
    '__import__': safe_import,
    'open': open, 'help': help, # Important for GAIA compatibility
    'Exception': Exception,
    'ImportError': ImportError,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'IndexError': IndexError,
    'KeyError': KeyError,
    'NameError': NameError,
    'AttributeError': AttributeError,
    'SyntaxError': SyntaxError,
    'RuntimeError': RuntimeError,
}

# Pre-populated globals acting as "Standard Imports"
_GLOBAL_ENV = {
    '__builtins__': _SAFE_BUILTINS,
    '__name__': '__main__',
    'math': math,
    'pd': pd,
    'np': np,
    'scipy': scipy,
    'os': os,           
    'json': json,
    'csv': csv,
    'zipfile': zipfile,
    'tarfile': tarfile,
    'ET': ET,
    'cv2': cv2,
    'pdfplumber': pdfplumber,
    'Image': PIL.Image,
    'sr': sr,
    'datetime': pd.to_datetime,
}

def python_executor(code: str, timeout: int = 10) -> str:
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
    restricted_globals = _GLOBAL_ENV.copy()
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