#!/usr/bin/env python3
"""
Debug MCP Handler Signature

Check the actual signature expected by MCP handlers.
"""

import asyncio
import sys
sys.path.insert(0, "src")

from lightmcp import LightMCP
from mcp.types import ReadResourceRequest, ReadResourceRequestParams

app = LightMCP(name="Debug Server", version="1.0.0")

@app.resource(uri="stream://debug/test", description="Debug resource")
async def debug_resource():
    return {"debug": True}

async def test_handler_signature():
    print("=== DEBUG MCP HANDLER SIGNATURE ===")
    
    # Get the actual MCP handler
    read_handler = app._mcp_server.request_handlers[ReadResourceRequest]
    
    # Check the handler signature
    import inspect
    sig = inspect.signature(read_handler)
    print(f"Handler signature: {sig}")
    print(f"Parameters: {list(sig.parameters.keys())}")
    
    # Create a test request
    params = ReadResourceRequestParams(uri="stream://debug/test")
    request = ReadResourceRequest(method='resources/read', params=params)
    
    print(f"\nRequest object:")
    print(f"  Type: {type(request)}")
    print(f"  Params: {request.params}")
    print(f"  Params URI: {request.params.uri}")
    print(f"  Params URI type: {type(request.params.uri)}")
    
    # Try to call the handler step by step
    print(f"\nTesting handler call...")
    
    try:
        # The MCP server might expect just the request object, not uri as string
        result = await read_handler(request)
        print(f"✅ Handler call successful")
        print(f"Result type: {type(result)}")
    except Exception as e:
        print(f"❌ Handler call failed: {e}")
        print(f"Error type: {type(e)}")
        
        # Let's check if our read_resource implementation expects the wrong signature
        print(f"\nChecking our implementation...")
        
        # Our implementation expects uri: str, but MCP might be passing the full request
        # Let's see what our actual registered handler looks like
        print(f"Our handler function: {read_handler}")

if __name__ == "__main__":
    asyncio.run(test_handler_signature())