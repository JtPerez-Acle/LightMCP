#!/usr/bin/env python3
"""
Debug Streaming Resource Registration

Quick debug script to understand the registration issue.
"""

import sys
sys.path.insert(0, "src")

from lightmcp import LightMCP

app = LightMCP(name="Debug Server", version="1.0.0")

@app.resource(
    uri="stream://test/debug",
    description="Debug streaming resource",
    streaming=True
)
async def debug_resource():
    return {"message": "debug"}

print("=== DEBUG STREAMING REGISTRATION ===")
print(f"Registered resources count: {len(app._resources)}")
print("Registered URIs:")
for uri, registration in app._resources.items():
    print(f"  '{uri}' -> streaming={registration.streaming}, func={registration.func.__name__}")

print("\nTesting direct access:")
test_uri = "stream://test/debug"
if test_uri in app._resources:
    print(f"✅ Found resource: {test_uri}")
    registration = app._resources[test_uri]
    print(f"  Function: {registration.func.__name__}")
    print(f"  Streaming: {registration.streaming}")
    print(f"  Stream type: {registration.stream_type}")
else:
    print(f"❌ Resource not found: {test_uri}")
    print(f"Available keys: {list(app._resources.keys())}")