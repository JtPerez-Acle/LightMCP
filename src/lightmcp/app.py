"""Core LightMCP application class."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, Union

from fastapi import FastAPI
from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel

from lightmcp.exceptions import ToolRegistrationError


class ToolRegistration:
    """Container for tool registration details."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.input_model = input_model


class LightMCP:
    """
    FastAPI-inspired framework for building MCP servers.
    
    Enables dual-protocol support: both HTTP REST endpoints and MCP tools
    from the same codebase.
    """

    def __init__(
        self,
        name: str = "LightMCP Server",
        version: str = "0.1.0",
        description: str = "",
    ):
        self.name = name
        self.version = version
        self.description = description or f"{name} - Built with LightMCP"
        
        # FastAPI app for HTTP endpoints
        self._fastapi_app = FastAPI(
            title=name,
            version=version,
            description=self.description,
        )
        
        # MCP server
        self._mcp_server = Server(name)
        
        # Registry for tools
        self._tools: Dict[str, ToolRegistration] = {}
        
        # Setup MCP handlers
        self._setup_mcp_handlers()

    def _setup_mcp_handlers(self):
        """Set up MCP server handlers."""
        
        @self._mcp_server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available MCP tools."""
            tools = []
            for tool_name, registration in self._tools.items():
                # Build input schema from Pydantic model
                input_schema = {}
                if registration.input_model:
                    input_schema = registration.input_model.model_json_schema()
                
                tools.append(
                    Tool(
                        name=tool_name,
                        description=registration.description,
                        inputSchema=input_schema,
                    )
                )
            return tools

        @self._mcp_server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
            """Execute a tool with the given arguments."""
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")
            
            registration = self._tools[name]
            
            # Validate input if model is provided
            if registration.input_model:
                validated_input = registration.input_model(**arguments)
                result = await registration.func(validated_input)
            else:
                result = await registration.func(**arguments)
            
            return result

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a function as an MCP tool.
        
        Args:
            name: Optional custom name for the tool
            description: Optional description of what the tool does
        """
        def decorator(func: Callable) -> Callable:
            # Extract input model from function annotations
            input_model = None
            if hasattr(func, "__annotations__"):
                # Look for Pydantic model in first parameter
                params = list(func.__annotations__.items())
                if params and len(params) > 0:
                    param_name, param_type = params[0]
                    if param_name != "return" and hasattr(param_type, "model_fields"):
                        input_model = param_type
            
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or ""
            
            # Register the tool
            self._tools[tool_name] = ToolRegistration(
                func=func,
                name=tool_name,
                description=tool_description,
                input_model=input_model,
            )
            
            return func
        
        return decorator

    def get(self, path: str, **kwargs) -> Callable:
        """Register a GET endpoint (delegates to FastAPI)."""
        return self._fastapi_app.get(path, **kwargs)

    def post(self, path: str, **kwargs) -> Callable:
        """Register a POST endpoint (delegates to FastAPI)."""
        return self._fastapi_app.post(path, **kwargs)

    def put(self, path: str, **kwargs) -> Callable:
        """Register a PUT endpoint (delegates to FastAPI)."""
        return self._fastapi_app.put(path, **kwargs)

    def delete(self, path: str, **kwargs) -> Callable:
        """Register a DELETE endpoint (delegates to FastAPI)."""
        return self._fastapi_app.delete(path, **kwargs)

    def patch(self, path: str, **kwargs) -> Callable:
        """Register a PATCH endpoint (delegates to FastAPI)."""
        return self._fastapi_app.patch(path, **kwargs)

    async def run_mcp(self):
        """Run the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._mcp_server.run(
                read_stream,
                write_stream,
                self._mcp_server.create_initialization_options(),
            )

    def run_http(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the HTTP server using uvicorn."""
        import uvicorn
        uvicorn.run(self._fastapi_app, host=host, port=port, **kwargs)

    @property
    def fastapi_app(self) -> FastAPI:
        """Access the underlying FastAPI application."""
        return self._fastapi_app

    @property
    def mcp_server(self) -> Server:
        """Access the underlying MCP server."""
        return self._mcp_server