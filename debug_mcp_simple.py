#!/usr/bin/env python3
"""
Simple MCP Handler Test

Test the simplest possible MCP resource handler.
"""

import asyncio
import sys
sys.path.insert(0, "src")

from mcp.server import Server
from mcp.types import ReadResourceResult, TextResourceContents, Resource
from mcp import Resource as ResourceType

print("=== SIMPLE MCP HANDLER TEST ===")

# Create a simple MCP server without LightMCP wrapper
server = Server("test-server")

# Register a simple resource
@server.list_resources()
async def list_resources():
    return [
        ResourceType(
            uri="test://simple",
            name="simple",
            description="Simple test resource",
            mimeType="text/plain"
        )
    ]

@server.read_resource()
async def read_resource(uri):
    print(f"Simple handler called with: {uri} (type: {type(uri)})")
    
    # Create the simplest possible response
    content = TextResourceContents(
        uri=uri,
        mimeType="text/plain",
        text="Hello from simple handler"
    )
    
    result = ReadResourceResult(contents=[content])
    print(f"Simple handler returning: {type(result)}")
    print(f"Result: {result}")
    
    return result

async def test_simple_handler():
    """Test the simple handler directly."""
    from mcp.types import ReadResourceRequest, ReadResourceRequestParams
    
    # Test handler directly
    try:
        handler = server.request_handlers[ReadResourceRequest]
        print(f"Handler found: {handler}")
        
        # Create request
        params = ReadResourceRequestParams(uri="test://simple")
        request = ReadResourceRequest(method='resources/read', params=params)
        
        print(f"Calling handler with request...")
        result = await handler(request)
        
        print(f"✅ Simple handler success!")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Simple handler failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_handler())