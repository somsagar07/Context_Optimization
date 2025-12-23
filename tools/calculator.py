import re
from sympy import sympify, N

def calculator(query: str) -> str:
    """Safe math evaluation using SymPy."""
    try:
        # Clean non-math chars
        clean_expr = re.sub(r'[^\d+\-*/().^]', '', query)
        if not clean_expr: 
            return "Error: Invalid Math"
        val = N(sympify(clean_expr))
        
        # Formatting
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return str(round(float(val), 4))
    except:
        return "Error: Calculation Failed"

