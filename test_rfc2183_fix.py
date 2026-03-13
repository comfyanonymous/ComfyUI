#!/usr/bin/env python3
"""
Test for Content-Disposition header fix
Verifies that the header matches RFC 2183 format
"""

import re

def test_content_disposition_header():
    """Test that Content-Disposition header is RFC 2183 compliant"""
    
    # Read the server.py file
    with open('server.py', 'r') as f:
        content = f.read()
    
    # Find all Content-Disposition headers
    pattern = r'"Content-Disposition":\s*f"([^"]+)"'
    matches = re.findall(pattern, content)
    
    print(f"Found {len(matches)} Content-Disposition headers")
    
    all_pass = True
    for i, match in enumerate(matches, 1):
        print(f"\nHeader {i}: {match}")
        
        # Check RFC 2183 compliance
        # Should start with "attachment;" or "inline;" followed by filename
        if match.startswith("attachment;") and "filename=" in match:
            print(f"  ✅ RFC 2183 compliant (attachment disposition)")
        elif match.startswith("inline;") and "filename=" in match:
            print(f"  ✅ RFC 2183 compliant (inline disposition)")
        elif match.startswith("filename="):
            print(f"  ❌ NOT RFC 2183 compliant (missing disposition type)")
            all_pass = False
        else:
            print(f"  ⚠️  Unknown format")
    
    print("\n" + "="*50)
    if all_pass:
        print("✅ All headers are RFC 2183 compliant!")
    else:
        print("❌ Some headers are NOT RFC 2183 compliant!")
    
    return all_pass

if __name__ == "__main__":
    import sys
    success = test_content_disposition_header()
    sys.exit(0 if success else 1)
