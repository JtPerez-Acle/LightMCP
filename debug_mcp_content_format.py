#!/usr/bin/env python3
"""
Debug MCP Content Format

Try different content formats to find what works.
"""

import asyncio
import sys
sys.path.insert(0, "src")

from mcp.server import Server
from mcp.types import ReadResourceResult, TextResourceContents, Resource
from mcp import Resource as ResourceType

print("=== DEBUG MCP CONTENT FORMAT ===")

# Create server
server = Server("test-server")

@server.list_resources()
async def list_resources():
    return [ResourceType(uri="test://format", name="format", description="Format test")]

@server.read_resource()
async def read_resource(uri):
    print(f"Testing different content formats for: {uri}")
    
    # Try different formats
    formats_to_test = [
        "direct_string",
        "text_resource_contents",
        "read_resource_result",
        "dict_format"
    ]
    
    test_format = "iterable_contents"  # Change this to test different formats
    
    if test_format == "direct_string":
        print("  Trying direct string...")
        return "Hello World"
        
    elif test_format == "text_resource_contents":
        print("  Trying TextResourceContents...")
        content = TextResourceContents(
            uri=uri,
            mimeType="text/plain", 
            text="Hello from TextResourceContents"
        )
        print(f"  Created content: {content}")
        print(f"  Content attributes: {dir(content)}")
        return content
        
    elif test_format == "read_resource_result":
        print("  Trying ReadResourceResult...")
        content = TextResourceContents(
            uri=uri,
            mimeType="text/plain",
            text="Hello from ReadResourceResult"
        )
        result = ReadResourceResult(contents=[content])
        return result
        
    elif test_format == "dict_format":
        print("  Trying dict format...")
        return {
            "contents": [{
                "uri": str(uri),
                "mimeType": "text/plain",
                "text": "Hello from dict"
            }]
        }
        
    elif test_format == "iterable_contents":
        print("  Trying iterable of ReadResourceContents...")
        content = TextResourceContents(
            uri=uri,
            mimeType="text/plain",
            text="Hello from iterable"
        )
        return [content]  # Return list of contents

async def test_content_format():
    """Test different content formats."""
    from mcp.types import ReadResourceRequest, ReadResourceRequestParams
    
    try:
        handler = server.request_handlers[ReadResourceRequest]
        
        params = ReadResourceRequestParams(uri="test://format")
        request = ReadResourceRequest(method='resources/read', params=params)
        
        result = await handler(request)
        
        print(f"✅ Content format test success!")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Content format test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_content_format())