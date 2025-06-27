#!/usr/bin/env python3
"""
Debug MCP Return Format

Check what the MCP read_resource handler should return.
"""

import sys
sys.path.insert(0, "src")

# Check MCP types
from mcp.types import ReadResourceResult, TextResourceContents, BlobResourceContents

print("=== DEBUG MCP RETURN FORMAT ===")

print("ReadResourceResult type:")
print(f"  {ReadResourceResult}")
print(f"  Fields: {ReadResourceResult.model_fields if hasattr(ReadResourceResult, 'model_fields') else 'No model_fields'}")

print("\nTextResourceContents type:")
print(f"  {TextResourceContents}")
print(f"  Fields: {TextResourceContents.model_fields if hasattr(TextResourceContents, 'model_fields') else 'No model_fields'}")

print("\nBlobResourceContents type:")
print(f"  {BlobResourceContents}")
print(f"  Fields: {BlobResourceContents.model_fields if hasattr(BlobResourceContents, 'model_fields') else 'No model_fields'}")

# Test creating a proper return format
print("\nTesting return format creation:")

try:
    # Create text content
    text_content = TextResourceContents(
        uri="test://debug",
        mimeType="text/plain",
        text="Hello World"
    )
    print(f"  TextResourceContents created: {text_content}")
    print(f"  Type: {type(text_content)}")
    print(f"  Dict format: {text_content.model_dump()}")
    
    # Create result
    result = ReadResourceResult(contents=[text_content])
    print(f"  ReadResourceResult created: {result}")
    print(f"  Type: {type(result)}")
    print(f"  Dict format: {result.model_dump()}")
    
except Exception as e:
    print(f"  ❌ Failed to create return format: {e}")

# Check what our current implementation returns
print("\nChecking our current implementation:")
from lightmcp import LightMCP

app = LightMCP(name="Debug Server", version="1.0.0")

@app.resource(uri="test://format", description="Test format")
async def test_format_resource():
    return {"test": "data"}

try:
    # Call _create_mcp_resource_content directly
    result = app._create_mcp_resource_content("test://format", {"test": "data"}, "application/json")
    print(f"  Our method returns: {type(result)}")
    print(f"  Content: {result}")
    
    # Check if it's the right format
    if "contents" in result:
        contents = result["contents"]
        print(f"  Contents length: {len(contents)}")
        if contents:
            first_content = contents[0]
            print(f"  First content type: {type(first_content)}")
            print(f"  First content: {first_content}")
            
            # Try to create MCP objects from this
            if isinstance(first_content, dict):
                if "text" in first_content:
                    mcp_content = TextResourceContents(**first_content)
                    print(f"  ✅ Successfully created TextResourceContents: {mcp_content}")
                elif "blob" in first_content:
                    mcp_content = BlobResourceContents(**first_content)
                    print(f"  ✅ Successfully created BlobResourceContents: {mcp_content}")
                
                # Try to create result
                mcp_result = ReadResourceResult(contents=[mcp_content])
                print(f"  ✅ Successfully created ReadResourceResult: {mcp_result}")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()