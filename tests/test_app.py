"""Tests for the core LightMCP application."""

import pytest
from pydantic import BaseModel

from lightmcp import LightMCP


def test_app_initialization():
    """Test basic app initialization."""
    app = LightMCP(name="Test Server", version="1.0.0")
    assert app.name == "Test Server"
    assert app.version == "1.0.0"
    assert app.fastapi_app is not None
    assert app.mcp_server is not None


def test_tool_registration():
    """Test tool registration."""
    app = LightMCP()
    
    @app.tool(description="A test tool that returns success")
    async def test_tool():
        return {"result": "success"}
    
    assert "test_tool" in app._tools
    assert app._tools["test_tool"].name == "test_tool"


def test_tool_with_pydantic_model():
    """Test tool registration with Pydantic model."""
    app = LightMCP()
    
    class TestInput(BaseModel):
        value: str
        count: int = 1
    
    @app.tool(description="Process input with a Pydantic model")
    async def process_input(input: TestInput):
        return {"processed": input.value * input.count}
    
    registration = app._tools["process_input"]
    assert registration.input_model == TestInput


def test_http_endpoint_registration():
    """Test HTTP endpoint registration."""
    app = LightMCP()
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    # Verify endpoint is registered in FastAPI app
    routes = [route.path for route in app.fastapi_app.routes]
    assert "/test" in routes


def test_dual_protocol_registration():
    """Test registering a function for both protocols."""
    app = LightMCP()
    
    class InputModel(BaseModel):
        text: str
    
    @app.tool(description="Process text input for both MCP and HTTP")
    @app.post("/process")
    async def process(input: InputModel):
        return {"result": input.text.upper()}
    
    # Check MCP registration
    assert "process" in app._tools
    
    # Check HTTP registration
    routes = [route.path for route in app.fastapi_app.routes]
    assert "/process" in routes


@pytest.mark.asyncio
async def test_mcp_list_tools():
    """Test MCP list_tools handler."""
    app = LightMCP()
    
    @app.tool(description="A test tool")
    async def test_tool():
        return "result"
    
    # Test that tools are registered correctly
    assert "test_tool" in app._tools
    registration = app._tools["test_tool"]
    assert registration.name == "test_tool"
    assert registration.description == "A test tool"