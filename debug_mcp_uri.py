#!/usr/bin/env python3
"""
Debug MCP URI Handling

Debug the exact URI handling in MCP protocol.
"""

import asyncio
import sys
sys.path.insert(0, "src")

from lightmcp import LightMCP

app = LightMCP(name="Debug Server", version="1.0.0")

@app.resource(
    uri="stream://debug/test",
    description="Debug resource",
    streaming=True
)
async def debug_resource():
    return {"debug": True}

async def debug_uri_handling():
    print("=== DEBUG MCP URI HANDLING ===")
    
    # Check what's in the registry
    print("Resources in registry:")
    for i, (uri, reg) in enumerate(app._resources.items()):
        print(f"  {i}: '{uri}' (type: {type(uri)}, len: {len(uri)})")
        print(f"      repr: {repr(uri)}")
        print(f"      bytes: {uri.encode('utf-8') if isinstance(uri, str) else 'not string'}")
    
    print("\nTesting MCP resource reading:")
    
    # Test direct access to read_resource handler
    test_uri = "stream://debug/test"
    print(f"Test URI: '{test_uri}' (type: {type(test_uri)}, len: {len(test_uri)})")
    print(f"Test URI repr: {repr(test_uri)}")
    print(f"Test URI bytes: {test_uri.encode('utf-8')}")
    
    # Check if URI is in resources manually
    print(f"\nManual check:")
    print(f"  test_uri in app._resources: {test_uri in app._resources}")
    print(f"  test_uri == list(app._resources.keys())[0]: {test_uri == list(app._resources.keys())[0]}")
    
    # Check each character
    reg_uri = list(app._resources.keys())[0]
    print(f"\nCharacter comparison:")
    print(f"  Registry URI: '{reg_uri}'")
    print(f"  Test URI:     '{test_uri}'")
    print(f"  Same length: {len(reg_uri) == len(test_uri)}")
    
    if len(reg_uri) == len(test_uri):
        for i, (a, b) in enumerate(zip(reg_uri, test_uri)):
            if a != b:
                print(f"  Difference at position {i}: '{a}' vs '{b}' (ord: {ord(a)} vs {ord(b)})")
                break
        else:
            print("  All characters match!")
    
    # Test via MCP protocol
    try:
        from mcp.types import ReadResourceRequest, ReadResourceRequestParams
        
        print(f"\nTesting MCP protocol:")
        params = ReadResourceRequestParams(uri=test_uri)
        print(f"  Created params with URI: {repr(params.uri)}")
        print(f"  Params URI type: {type(params.uri)}")
        
        read_request = ReadResourceRequest(method='resources/read', params=params)
        print(f"  Created request")
        
        # Get the handler
        read_handler = app._mcp_server.request_handlers[ReadResourceRequest]
        print(f"  Got handler: {read_handler}")
        
        # This is where it should fail, let's see what URI actually gets passed
        print(f"  About to call handler...")
        
        # Let's monkey patch to see what URI gets passed
        original_read_resource = read_handler
        
        async def debug_read_resource(request):
            print(f"    Handler received request type: {type(request)}")
            print(f"    Request params: {request.params}")
            print(f"    Request params URI: {repr(request.params.uri)}")
            print(f"    Request params URI type: {type(request.params.uri)}")
            
            # Check if URI is in resources
            uri_from_request = str(request.params.uri)  # Convert to string
            print(f"    URI as string: '{uri_from_request}'")
            print(f"    URI in resources: {uri_from_request in app._resources}")
            
            return await original_read_resource(request)
        
        # Call with debug wrapper
        result = await debug_read_resource(read_request)
        print(f"  ✅ Success!")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_uri_handling())