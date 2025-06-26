"""Simple MCP functionality test."""

import asyncio
import sys
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, "src")

from lightmcp import LightMCP


async def test_mcp_tools():
    """Test MCP tool functionality directly."""
    
    # Create app
    app = LightMCP(name="Test MCP Server")
    
    class EchoInput(BaseModel):
        message: str
        count: int = 1
    
    @app.tool(description="Echo a message")
    async def echo(input: EchoInput):
        return {
            "echoed": input.message * input.count,
            "original": input.message,
            "count": input.count
        }
    
    @app.tool(description="Simple ping")
    async def ping():
        return {"status": "pong"}
    
    print("✓ MCP app created successfully")
    print(f"✓ Registered {len(app._tools)} tools: {list(app._tools.keys())}")
    
    # Test tool execution directly
    echo_registration = app._tools["echo"]
    echo_input = EchoInput(message="Hello", count=3)
    result = await echo_registration.func(echo_input)
    
    assert result["echoed"] == "HelloHelloHello"
    assert result["original"] == "Hello"
    assert result["count"] == 3
    print("✓ Echo tool works correctly")
    
    # Test ping tool
    ping_registration = app._tools["ping"]
    ping_result = await ping_registration.func()
    assert ping_result["status"] == "pong"
    print("✓ Ping tool works correctly")
    
    # Test input schema generation
    echo_tool = app._tools["echo"]
    assert echo_tool.input_model == EchoInput
    print("✓ Input model detected correctly")
    
    print("\nMCP functionality test PASSED!")


if __name__ == "__main__":
    asyncio.run(test_mcp_tools())