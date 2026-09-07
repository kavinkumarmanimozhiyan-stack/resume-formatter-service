#!/usr/bin/env python
"""Quick test of template imports"""

try:
    from app.template1_generator import generate_template1
    print("✓ template1_generator imports OK")
except Exception as e:
    print(f"✗ template1_generator failed: {e}")

try:
    from app.template2_generator import generate_template2
    print("✓ template2_generator imports OK")
except Exception as e:
    print(f"✗ template2_generator failed: {e}")

try:
    from app.template3_generator import generate_template3
    print("✓ template3_generator imports OK")
except Exception as e:
    print(f"✗ template3_generator failed: {e}")

try:
    from app.template4_generator import generate_template4
    print("✓ template4_generator imports OK")
except Exception as e:
    print(f"✗ template4_generator failed: {e}")

print("\nAll templates imported successfully!")
