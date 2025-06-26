"""End-to-end tests for MCP protocol functionality."""

import asyncio
import json
import sys
from typing import Any, Dict

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, "../src")

from lightmcp import LightMCP


class TestMCPServer:
    """Test server for E2E MCP testing."""
    
    def __init__(self):
        self.app = LightMCP(
            name="E2E Test Server",
            version="1.0.0",
            description="Server for end-to-end MCP testing"
        )
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up test tools."""
        
        class EchoInput(BaseModel):
            message: str
            count: int = 1
        
        class MathInput(BaseModel):
            a: float
            b: float
            operation: str = "add"
        
        @self.app.tool(description="Echo a message with optional repetition")
        async def echo(input: EchoInput) -> Dict[str, Any]:
            """Echo tool that repeats a message."""
            return {
                "echoed": input.message * input.count,
                "original": input.message,
                "count": input.count
            }
        
        @self.app.tool(description="Perform basic math operations")
        async def calculate(input: MathInput) -> Dict[str, Any]:
            """Math tool that performs calculations."""
            operations = {
                "add": input.a + input.b,
                "subtract": input.a - input.b,
                "multiply": input.a * input.b,
                "divide": input.a / input.b if input.b != 0 else None
            }
            
            if input.operation not in operations:
                raise ValueError(f"Unknown operation: {input.operation}")
            
            result = operations[input.operation]
            if result is None:
                raise ValueError("Division by zero")
            
            return {
                "result": result,
                "operation": input.operation,
                "operands": [input.a, input.b]
            }
        
        @self.app.tool(description="Simple tool with no input validation")
        async def ping() -> Dict[str, str]:
            """Simple ping tool."""
            return {"status": "pong", "timestamp": "2024-01-01T00:00:00Z"}
    
    async def run_server(self):
        """Run the MCP server."""
        await self.app.run_mcp()


@pytest.fixture
async def mcp_server_process():
    """Start MCP server in a subprocess for testing."""
    import subprocess
    import tempfile
    import os
    
    # Create a temporary server script
    server_script = '''
import asyncio
import sys
sys.path.insert(0, "src")

from tests.test_e2e_mcp import TestMCPServer

async def main():
    server = TestMCPServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(server_script)
        script_path = f.name
    
    try:
        # Start server process
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        yield process
        
    finally:
        # Clean up
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.unlink(script_path)


@pytest.mark.asyncio
async def test_mcp_list_tools():
    """Test listing MCP tools."""
    server = TestMCPServer()
    
    # Get tools list directly from server
    tools = await server.app._mcp_server._request_handlers["tools/list"]()
    
    assert len(tools) == 3
    tool_names = [tool.name for tool in tools]
    assert "echo" in tool_names
    assert "calculate" in tool_names  
    assert "ping" in tool_names
    
    # Verify tool descriptions
    echo_tool = next(t for t in tools if t.name == "echo")
    assert "Echo a message" in echo_tool.description
    assert echo_tool.inputSchema is not None


@pytest.mark.asyncio
async def test_mcp_call_tool_with_validation():
    """Test calling MCP tools with Pydantic validation."""
    server = TestMCPServer()
    
    # Test echo tool with valid input
    call_handler = server.app._mcp_server._request_handlers["tools/call"]
    
    result = await call_handler(
        name="echo",
        arguments={"message": "Hello", "count": 3}
    )
    
    assert result["echoed"] == "HelloHelloHello"
    assert result["original"] == "Hello"
    assert result["count"] == 3


@pytest.mark.asyncio
async def test_mcp_call_tool_validation_error():
    """Test MCP tool calls with validation errors."""
    server = TestMCPServer()
    call_handler = server.app._mcp_server._request_handlers["tools/call"]
    
    # Test with missing required field
    with pytest.raises(Exception):  # Should raise validation error
        await call_handler(
            name="echo",
            arguments={"count": 3}  # Missing 'message'
        )


@pytest.mark.asyncio
async def test_mcp_math_tool():
    """Test math tool with different operations."""
    server = TestMCPServer()
    call_handler = server.app._mcp_server._request_handlers["tools/call"]
    
    # Test addition
    result = await call_handler(
        name="calculate",
        arguments={"a": 5.0, "b": 3.0, "operation": "add"}
    )
    assert result["result"] == 8.0
    assert result["operation"] == "add"
    
    # Test multiplication
    result = await call_handler(
        name="calculate", 
        arguments={"a": 4.0, "b": 2.5, "operation": "multiply"}
    )
    assert result["result"] == 10.0
    
    # Test division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        await call_handler(
            name="calculate",
            arguments={"a": 5.0, "b": 0.0, "operation": "divide"}
        )


@pytest.mark.asyncio
async def test_mcp_tool_no_input():
    """Test MCP tool that requires no input."""
    server = TestMCPServer()
    call_handler = server.app._mcp_server._request_handlers["tools/call"]
    
    result = await call_handler(name="ping", arguments={})
    
    assert result["status"] == "pong"
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_mcp_unknown_tool():
    """Test calling unknown MCP tool."""
    server = TestMCPServer()
    call_handler = server.app._mcp_server._request_handlers["tools/call"]
    
    with pytest.raises(ValueError, match="Unknown tool"):
        await call_handler(name="nonexistent", arguments={})


@pytest.mark.asyncio
async def test_mcp_tool_input_schema_generation():
    """Test that input schemas are properly generated from Pydantic models."""
    server = TestMCPServer()
    
    tools = await server.app._mcp_server._request_handlers["tools/list"]()
    
    echo_tool = next(t for t in tools if t.name == "echo")
    schema = echo_tool.inputSchema
    
    # Verify schema structure
    assert "properties" in schema
    assert "message" in schema["properties"]
    assert "count" in schema["properties"]
    
    # Verify required fields
    assert "required" in schema
    assert "message" in schema["required"]
    assert "count" not in schema.get("required", [])  # Has default value
    
    # Verify types
    assert schema["properties"]["message"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"


if __name__ == "__main__":
    # Run a quick smoke test
    async def smoke_test():
        print("Running MCP E2E smoke test...")
        server = TestMCPServer()
        
        # Test tool listing
        tools = await server.app._mcp_server._request_handlers["tools/list"]()
        print(f"✓ Found {len(tools)} tools: {[t.name for t in tools]}")
        
        # Test tool calling
        call_handler = server.app._mcp_server._request_handlers["tools/call"]
        result = await call_handler(
            name="echo",
            arguments={"message": "Test", "count": 2}
        )
        print(f"✓ Echo tool result: {result}")
        
        print("MCP E2E smoke test passed!")
    
    asyncio.run(smoke_test())