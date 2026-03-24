"""
Quick test for MCP tool functions — no server, no agent needed.
Just imports and calls the functions directly.
"""

from mcp_server import read_code, detect_syntax_errors, extract_code_structure

# --- Test read_code with a raw string ---
print("=== read_code (raw string) ===")
print(read_code("print('hello')"))


# --- Test read_code with a file ---
print("\n=== read_code (file) ===")
print(read_code("mcp_server.py")[:300] + "...")                # reads itself as test


# --- Test detect_syntax_errors with clean code ---
print("\n=== detect_syntax_errors (clean) ===")
print(detect_syntax_errors("def add(a, b):\n    return a + b"))

# --- Test detect_syntax_errors with bad code ---
print("\n=== detect_syntax_errors (issues) ===")
bad_code = """
import os
password = "admin123"
eval(input("Enter command: "))
"""
print(detect_syntax_errors(bad_code))             # should trigger both ruff and bandit

# --- Test extract_code_structure ---
print("\n=== extract_code_structure ===")
sample = """
import os
from pathlib import Path

class FileProcessor:
    def __init__(self, path):
        self.path = path

    def process(self):
        pass

def helper(x, y):
    \"\"\"A helper function.\"\"\"
    return x + y
"""
print(extract_code_structure(sample))