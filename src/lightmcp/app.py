"""Core LightMCP application class."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from fastapi import FastAPI
from mcp import Tool, Resource
from mcp.types import Prompt
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel

from lightmcp.exceptions import ToolRegistrationError


@dataclass
class ToolRegistration:
    """Container for tool registration details."""
    func: Callable
    name: str
    description: str
    input_model: Optional[Type[BaseModel]] = None


@dataclass
class ResourceRegistration:
    """Container for resource registration details."""
    func: Callable
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None


@dataclass 
class PromptRegistration:
    """Container for prompt registration details."""
    func: Callable
    name: str
    description: str
    arguments: List[dict]


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
        
        # Registry for tools, resources, and prompts
        self._tools: Dict[str, ToolRegistration] = {}
        self._resources: Dict[str, ResourceRegistration] = {}
        self._prompts: Dict[str, PromptRegistration] = {}
        
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

        @self._mcp_server.list_resources()
        async def list_resources() -> List[Resource]:
            """List all available MCP resources."""
            resources = []
            for resource_uri, registration in self._resources.items():
                resources.append(
                    Resource(
                        uri=resource_uri,
                        name=registration.name,
                        description=registration.description,
                        mimeType=registration.mime_type,
                    )
                )
            return resources

        @self._mcp_server.read_resource()
        async def read_resource(uri: str) -> Dict[str, Any]:
            """Read content of a specific resource."""
            if uri not in self._resources:
                raise ValueError(f"Unknown resource: {uri}")
            
            registration = self._resources[uri]
            content = await registration.func()
            
            # Ensure content is properly formatted
            if isinstance(content, dict):
                import json
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": registration.mime_type or "application/json",
                            "text": json.dumps(content, indent=2)
                        }
                    ]
                }
            elif isinstance(content, str):
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": registration.mime_type or "text/plain",
                            "text": content
                        }
                    ]
                }
            else:
                import json
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": registration.mime_type or "application/json",
                            "text": json.dumps(content, indent=2)
                        }
                    ]
                }

        @self._mcp_server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """List all available MCP prompts."""
            prompts = []
            for prompt_name, registration in self._prompts.items():
                prompts.append(
                    Prompt(
                        name=prompt_name,
                        description=registration.description,
                        arguments=registration.arguments,
                    )
                )
            return prompts

        @self._mcp_server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Get prompt with optional arguments."""
            if name not in self._prompts:
                raise ValueError(f"Unknown prompt: {name}")
            
            registration = self._prompts[name]
            
            # Call the prompt function with arguments
            if arguments:
                result = await registration.func(**arguments)
            else:
                result = await registration.func()
            
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

    def resource(
        self,
        uri: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a function as an MCP resource.
        
        Args:
            uri: Resource URI (defaults to function name with resource:// scheme)
            name: Optional display name for the resource
            description: Optional description of the resource
            mime_type: MIME type of the resource content
        """
        def decorator(func: Callable) -> Callable:
            resource_uri = uri or f"resource://{func.__name__}"
            resource_name = name or func.__name__
            resource_description = description or func.__doc__ or ""
            
            # Register the resource
            self._resources[resource_uri] = ResourceRegistration(
                func=func,
                uri=resource_uri,
                name=resource_name,
                description=resource_description,
                mime_type=mime_type,
            )
            
            return func
        
        return decorator

    def prompt(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        arguments: Optional[List[dict]] = None,
    ) -> Callable:
        """
        Decorator to register a function as an MCP prompt.
        
        Args:
            name: Prompt name (defaults to function name)
            description: Optional description of the prompt
            arguments: List of argument definitions for the prompt
        """
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            prompt_description = description or func.__doc__ or ""
            
            # Extract arguments from function signature if not provided
            prompt_arguments = arguments or []
            if not prompt_arguments and hasattr(func, "__annotations__"):
                import inspect
                sig = inspect.signature(func)
                for param_name, param in sig.parameters.items():
                    if param_name != "return":
                        arg_def = {
                            "name": param_name,
                            "description": f"Parameter {param_name}",
                            "required": param.default == inspect.Parameter.empty
                        }
                        prompt_arguments.append(arg_def)
            
            # Register the prompt
            self._prompts[prompt_name] = PromptRegistration(
                func=func,
                name=prompt_name,
                description=prompt_description,
                arguments=prompt_arguments,
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