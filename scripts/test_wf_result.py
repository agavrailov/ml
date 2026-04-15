"""Manual debugging script (not an automated test).

This file starts with "test_" so pytest will try to collect it. Skip collection
to keep the automated suite stable.
"""

import pytest

pytest.skip("manual script - not a pytest test", allow_module_level=True)

from src.core.contracts import WalkForwardResult
import json

job_id = "a09f6d204be44589a357b698821cb1e3"
result_path = f"runs/{job_id}/result.json"

with open(result_path) as f:
    res_obj = json.load(f)

print("Result JSON keys:", list(res_obj.keys()))
print("\nResult JSON content:")
print(json.dumps(res_obj, indent=2))

print("\n" + "="*50)
print("Attempting to create WalkForwardResult...")
print("="*50)

try:
    res = WalkForwardResult(**res_obj)
    print(f"✓ Success: {res}")
except TypeError as e:
    print(f"✗ Failed with TypeError: {e}")
    print("\nThe issue: result.json contains extra 'summary' key not in WalkForwardResult dataclass")
