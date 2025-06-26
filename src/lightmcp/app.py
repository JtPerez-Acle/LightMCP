"""Core LightMCP application class."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, Union
import warnings
from dataclasses import dataclass

from fastapi import FastAPI
from mcp import Tool, Resource
from mcp.types import Prompt
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel

from lightmcp.exceptions import (
    ToolRegistrationError,
    ResourceRegistrationError,
    PromptRegistrationError,
    ToolExecutionError,
    ResourceAccessError,
    PromptExecutionError,
    ValidationError,
    TypeValidationError,
    ProtocolError,
    create_validation_error_from_pydantic,
    wrap_async_execution_error
)


@dataclass
class ToolRegistration:
    """Container for tool registration details."""
    func: Callable
    name: str
    description: str
    input_model: Optional[Type[BaseModel]] = None
    input_type_info: Optional[Dict[str, Any]] = None  # Store type analysis info


@dataclass
class ResourceRegistration:
    """Container for resource registration details with binary and streaming support."""
    func: Callable
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None
    content_type: str = "auto"  # "text", "binary", "auto"
    streaming: bool = False
    cache_ttl: Optional[int] = None  # Cache TTL in seconds
    serializer: Optional[str] = None  # "json", "xml", "yaml", "csv", etc.
    max_size: Optional[int] = None  # Max content size in bytes
    compression: Optional[str] = None  # "gzip", "brotli", None
    chunk_size: int = 8192  # Chunk size for streaming in bytes
    stream_type: str = "data"  # "file", "data", "progress", "realtime"
    max_chunks: Optional[int] = None  # Maximum number of chunks for safety


@dataclass 
class PromptRegistration:
    """Container for prompt registration details."""
    func: Callable
    name: str
    description: str
    arguments: List[dict]


@dataclass
class StreamManifest:
    """Container for stream manifest information."""
    stream_id: str
    stream_type: str
    content_type: str
    total_size: Optional[int] = None
    chunk_count: Optional[int] = None
    created_at: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StreamChunk:
    """Container for individual stream chunk."""
    stream_id: str
    chunk_index: int
    data: Any
    is_final: bool = False
    timestamp: float = 0.0
    size: Optional[int] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()
        if self.size is None and hasattr(self.data, '__len__'):
            self.size = len(str(self.data))


class StreamingResourceHandler:
    """Enhanced resource handler with streaming capabilities."""
    
    def __init__(self, chunk_size: int = 8192, max_chunks: int = 1000):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.active_streams: Dict[str, StreamManifest] = {}
    
    async def create_file_stream(self, file_path: str, chunk_size: Optional[int] = None) -> List[StreamChunk]:
        """Create chunks from file for streaming."""
        import os
        import time
        from typing import AsyncGenerator
        
        chunk_size = chunk_size or self.chunk_size
        stream_id = f"file_stream_{hash(file_path)}_{int(time.time())}"
        
        if not os.path.exists(file_path):
            raise ResourceAccessError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        estimated_chunks = (file_size // chunk_size) + 1
        
        # Create manifest
        manifest = StreamManifest(
            stream_id=stream_id,
            stream_type="file",
            content_type="application/octet-stream",
            total_size=file_size,
            chunk_count=estimated_chunks,
            metadata={"file_path": file_path, "chunk_size": chunk_size}
        )
        self.active_streams[stream_id] = manifest
        
        chunks = []
        try:
            with open(file_path, 'rb') as f:
                chunk_index = 0
                while chunk_data := f.read(chunk_size):
                    is_final = (len(chunk_data) < chunk_size)
                    
                    chunk = StreamChunk(
                        stream_id=stream_id,
                        chunk_index=chunk_index,
                        data=chunk_data,
                        is_final=is_final,
                        size=len(chunk_data)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    if is_final:
                        break
                    
                    # Safety check
                    if chunk_index >= self.max_chunks:
                        raise ResourceAccessError(f"File too large: exceeds {self.max_chunks} chunks")
                        
        except Exception as e:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            raise ResourceAccessError(f"File streaming failed: {str(e)}", resource_uri=file_path)
        
        return chunks
    
    async def create_data_stream(self, data_generator, stream_type: str = "data") -> List[StreamChunk]:
        """Create chunks from async data generator."""
        import uuid
        
        stream_id = f"{stream_type}_stream_{uuid.uuid4().hex[:8]}"
        
        # Create manifest (size unknown for generators)
        manifest = StreamManifest(
            stream_id=stream_id,
            stream_type=stream_type,
            content_type="application/json",
            metadata={"generator_type": str(type(data_generator))}
        )
        self.active_streams[stream_id] = manifest
        
        chunks = []
        try:
            chunk_index = 0
            
            # Handle both async generators and regular iterables
            if hasattr(data_generator, '__aiter__'):
                # Async generator
                async for data_item in data_generator:
                    chunk = StreamChunk(
                        stream_id=stream_id,
                        chunk_index=chunk_index,
                        data=data_item,
                        is_final=False
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Safety check
                    if chunk_index >= self.max_chunks:
                        break
            else:
                # Regular iterable
                for data_item in data_generator:
                    chunk = StreamChunk(
                        stream_id=stream_id,
                        chunk_index=chunk_index,
                        data=data_item,
                        is_final=False
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Safety check
                    if chunk_index >= self.max_chunks:
                        break
            
            # Mark the last chunk as final
            if chunks:
                chunks[-1].is_final = True
                
        except Exception as e:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            raise ResourceAccessError(f"Data streaming failed: {str(e)}")
        
        return chunks
    
    def get_stream_manifest(self, stream_id: str) -> Optional[StreamManifest]:
        """Get stream manifest by ID."""
        return self.active_streams.get(stream_id)
    
    def cleanup_stream(self, stream_id: str) -> bool:
        """Clean up stream resources."""
        return self.active_streams.pop(stream_id, None) is not None


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
        
        # Streaming handler
        self._streaming_handler = StreamingResourceHandler()
        
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
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Any]:
            """Execute a tool with the given arguments with comprehensive error handling."""
            from mcp.types import TextContent
            import json
            
            if name not in self._tools:
                raise ToolExecutionError(
                    f"Tool '{name}' not found",
                    tool_name=name,
                    context={"available_tools": list(self._tools.keys())},
                    recovery_suggestions=[
                        f"Check available tools: {', '.join(self._tools.keys())}",
                        "Verify tool registration was successful",
                        "Check for typos in tool name"
                    ]
                )
            
            registration = self._tools[name]
            
            try:
                # Validate input using enhanced container-aware validation
                if registration.input_model:
                    try:
                        validated_input = self._validate_container_input(registration, arguments)
                    except Exception as e:
                        if isinstance(e, ValidationError):
                            raise e
                        else:
                            raise create_validation_error_from_pydantic(e, arguments)
                    
                    # Execute tool function with validated input
                    try:
                        # Handle different calling patterns based on type info
                        if registration.input_type_info:
                            type_info = registration.input_type_info
                            if type_info["type"] in ("list", "dict"):
                                # For container types, pass the validated data as named parameter
                                result = await registration.func(**{type_info["param_name"]: validated_input})
                            elif type_info["type"] == "union":
                                # For union types, pass as named parameter
                                result = await registration.func(**{type_info["param_name"]: validated_input})
                            else:
                                # Direct model - pass as single argument
                                result = await registration.func(validated_input)
                        else:
                            # Fallback to original behavior
                            result = await registration.func(validated_input)
                    except Exception as e:
                        raise wrap_async_execution_error(
                            func_name=name,
                            original_error=e,
                            input_data=arguments
                        )
                else:
                    # Execute tool function without input model
                    try:
                        result = await registration.func(**arguments)
                    except TypeError as e:
                        raise TypeValidationError(
                            f"Invalid arguments for tool '{name}': {str(e)}",
                            parameter_name="function_arguments",
                            context={"provided_arguments": list(arguments.keys())},
                            original_exception=e
                        )
                    except Exception as e:
                        raise wrap_async_execution_error(
                            func_name=name,
                            original_error=e,
                            input_data=arguments
                        )
                
                # Convert result to proper MCP format
                try:
                    # Serialize result to JSON string
                    if hasattr(result, 'model_dump'):  # Pydantic model
                        json_str = json.dumps(result.model_dump(), indent=2)
                    elif hasattr(result, 'dict'):  # Pydantic v1 model
                        json_str = json.dumps(result.dict(), indent=2)
                    else:
                        json_str = json.dumps(result, indent=2)
                    
                    # Wrap in TextContent
                    return [TextContent(type="text", text=json_str)]
                    
                except Exception as e:
                    raise ToolExecutionError(
                        f"Failed to serialize result for tool '{name}': {str(e)}",
                        tool_name=name,
                        context={"result_type": type(result).__name__},
                        original_exception=e
                    )
                
            except Exception as e:
                # Catch any remaining exceptions and wrap appropriately
                if isinstance(e, (ToolExecutionError, ValidationError, TypeValidationError)):
                    raise
                else:
                    raise ToolExecutionError(
                        f"Unexpected error executing tool '{name}': {str(e)}",
                        tool_name=name,
                        input_data=arguments,
                        original_exception=e
                    )

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
        async def read_resource(uri):
            """Read content of a specific resource with comprehensive error handling."""
            # Convert URI to string (handles both str and AnyUrl)
            uri = str(uri)
            
            if uri not in self._resources:
                raise ResourceAccessError(
                    f"Resource '{uri}' not found",
                    resource_uri=uri,
                    context={"available_resources": list(self._resources.keys())},
                    recovery_suggestions=[
                        f"Check available resources: {', '.join(self._resources.keys())}",
                        "Verify resource registration was successful",
                        "Check URI format and scheme"
                    ]
                )
            
            registration = self._resources[uri]
            
            try:
                content = await registration.func()
            except Exception as e:
                raise ResourceAccessError(
                    f"Failed to read resource '{uri}': {str(e)}",
                    resource_uri=uri,
                    original_exception=e,
                    recovery_suggestions=[
                        "Check resource function implementation",
                        "Verify resource data is available",
                        "Check permissions and access rights",
                        "Review error logs for details"
                    ]
                )
            
            # Handle streaming resources
            if registration.streaming:
                try:
                    # For now, return simplified streaming format as JSON string
                    # TODO: Implement proper streaming once MCP compatibility is fixed
                    streaming_result = await self._handle_streaming_resource(uri, content, registration)
                    import json
                    return json.dumps(streaming_result, indent=2, default=str)
                except Exception as e:
                    raise ResourceAccessError(
                        f"Failed to process streaming resource '{uri}': {str(e)}",
                        resource_uri=uri,
                        original_exception=e,
                        recovery_suggestions=[
                            "Check streaming configuration",
                            "Verify content is suitable for streaming",
                            "Check chunk_size and max_chunks limits",
                            "Review streaming function implementation"
                        ]
                    )
            
            # Format content using enhanced binary-aware system
            try:
                # For now, return simple string format that works with MCP library
                # TODO: Implement proper streaming and binary support once MCP compatibility is fixed
                
                if isinstance(content, str):
                    return content
                elif isinstance(content, bytes):
                    # For binary content, return as base64 string
                    import base64
                    return base64.b64encode(content).decode('utf-8')
                else:
                    # For other types, serialize to JSON
                    import json
                    return json.dumps(content, indent=2)
                
            except Exception as e:
                raise ResourceAccessError(
                    f"Failed to format resource content for '{uri}': {str(e)}",
                    resource_uri=uri,
                    context={
                        "content_type": type(content).__name__,
                        "declared_mime_type": registration.mime_type,
                        "registration_type": registration.content_type
                    },
                    original_exception=e,
                    recovery_suggestions=[
                        "Check content format and encoding",
                        "Verify MIME type declaration",
                        "Review resource function return type",
                        "Check for binary vs text content mismatch"
                    ]
                )

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
            """Get prompt with optional arguments and comprehensive error handling."""
            if name not in self._prompts:
                raise PromptExecutionError(
                    f"Prompt '{name}' not found",
                    prompt_name=name,
                    context={"available_prompts": list(self._prompts.keys())},
                    recovery_suggestions=[
                        f"Check available prompts: {', '.join(self._prompts.keys())}",
                        "Verify prompt registration was successful",
                        "Check for typos in prompt name"
                    ]
                )
            
            registration = self._prompts[name]
            
            try:
                # Call the prompt function with arguments
                if arguments:
                    try:
                        result = await registration.func(**arguments)
                    except TypeError as e:
                        raise TypeValidationError(
                            f"Invalid arguments for prompt '{name}': {str(e)}",
                            parameter_name="prompt_arguments",
                            context={
                                "provided_arguments": list(arguments.keys()) if arguments else [],
                                "expected_arguments": [arg["name"] for arg in registration.arguments]
                            },
                            original_exception=e
                        )
                    except Exception as e:
                        raise PromptExecutionError(
                            f"Error executing prompt '{name}': {str(e)}",
                            prompt_name=name,
                            arguments=arguments,
                            original_exception=e
                        )
                else:
                    try:
                        result = await registration.func()
                    except Exception as e:
                        raise PromptExecutionError(
                            f"Error executing prompt '{name}': {str(e)}",
                            prompt_name=name,
                            arguments=arguments,
                            original_exception=e
                        )
                
                # Validate result format
                if not isinstance(result, dict):
                    raise PromptExecutionError(
                        f"Prompt '{name}' returned invalid format. Expected dict, got {type(result).__name__}",
                        prompt_name=name,
                        arguments=arguments,
                        context={"result_type": type(result).__name__}
                    )
                
                return result
                
            except Exception as e:
                if isinstance(e, (PromptExecutionError, TypeValidationError)):
                    raise
                else:
                    raise PromptExecutionError(
                        f"Unexpected error with prompt '{name}': {str(e)}",
                        prompt_name=name,
                        arguments=arguments,
                        original_exception=e
                    )

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a function as an MCP tool with enhanced validation.
        
        Args:
            name: Optional custom name for the tool
            description: Optional description of what the tool does
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            
            # Validation: Check if tool name already exists
            if tool_name in self._tools:
                raise ToolRegistrationError(
                    f"Tool '{tool_name}' is already registered",
                    tool_name=tool_name,
                    context={"existing_tools": list(self._tools.keys())}
                )
            
            # Validation: Ensure function is callable
            if not callable(func):
                raise ToolRegistrationError(
                    f"Tool '{tool_name}' must be a callable function",
                    tool_name=tool_name,
                    context={"provided_type": type(func).__name__}
                )
            
            # Validation: Check if function is async (recommended for MCP)
            import inspect
            if not inspect.iscoroutinefunction(func):
                # Log warning but don't fail - sync functions are supported
                import warnings
                warnings.warn(
                    f"Tool '{tool_name}' is not async. Consider using 'async def' for better performance.",
                    UserWarning
                )
            
            # Extract input model from function annotations with enhanced detection
            input_model = None
            input_type_info = None
            try:
                if hasattr(func, "__annotations__"):
                    # Look for Pydantic model in first parameter
                    params = list(func.__annotations__.items())
                    if params and len(params) > 0:
                        param_name, param_type = params[0]
                        if param_name != "return":
                            # Check if it's a direct Pydantic model
                            if hasattr(param_type, "model_fields") or hasattr(param_type, "__fields__"):
                                input_model = param_type
                                input_type_info = {
                                    "type": "direct",
                                    "original_type": param_type,
                                    "param_name": param_name
                                }
                            # Check for complex types with __origin__ (Union, List, Dict, etc.)
                            elif hasattr(param_type, "__origin__"):
                                origin = param_type.__origin__
                                args = getattr(param_type, "__args__", ())
                                
                                # Handle Union types (including Optional)
                                if origin is Union:
                                    for arg_type in args:
                                        if hasattr(arg_type, "model_fields") or hasattr(arg_type, "__fields__"):
                                            input_model = arg_type
                                            input_type_info = {
                                                "type": "union",
                                                "original_type": param_type,
                                                "param_name": param_name,
                                                "model_type": arg_type,
                                                "union_args": args
                                            }
                                            break
                                
                                # Handle List[Model] and Tuple[Model]
                                elif origin in (list, tuple) and args:
                                    inner_type = args[0]
                                    if hasattr(inner_type, "model_fields") or hasattr(inner_type, "__fields__"):
                                        input_model = inner_type
                                        input_type_info = {
                                            "type": "list" if origin is list else "tuple",
                                            "original_type": param_type,
                                            "param_name": param_name,
                                            "item_type": inner_type
                                        }
                                
                                # Handle Dict[str, Model]
                                elif origin is dict and len(args) >= 2:
                                    value_type = args[1]
                                    if hasattr(value_type, "model_fields") or hasattr(value_type, "__fields__"):
                                        input_model = value_type
                                        input_type_info = {
                                            "type": "dict",
                                            "original_type": param_type,
                                            "param_name": param_name,
                                            "key_type": args[0],
                                            "value_type": value_type
                                        }
            except Exception as e:
                raise ToolRegistrationError(
                    f"Failed to extract input model for tool '{tool_name}': {str(e)}",
                    tool_name=tool_name,
                    original_exception=e
                )
            
            tool_description = description or func.__doc__ or ""
            
            # Validation: Ensure description is provided
            if not tool_description.strip():
                raise ToolRegistrationError(
                    f"Tool '{tool_name}' must have a description (provide via parameter or docstring)",
                    tool_name=tool_name
                )
            
            try:
                # Register the tool
                self._tools[tool_name] = ToolRegistration(
                    func=func,
                    name=tool_name,
                    description=tool_description,
                    input_model=input_model,
                    input_type_info=input_type_info,
                )
            except Exception as e:
                raise ToolRegistrationError(
                    f"Failed to register tool '{tool_name}': {str(e)}",
                    tool_name=tool_name,
                    original_exception=e
                )
            
            return func
        
        return decorator

    def resource(
        self,
        uri: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
        content_type: str = "auto",
        streaming: bool = False,
        cache_ttl: Optional[int] = None,
        serializer: Optional[str] = None,
        max_size: Optional[int] = None,
        compression: Optional[str] = None,
        chunk_size: int = 8192,
        stream_type: str = "data",
        max_chunks: Optional[int] = None,
    ) -> Callable:
        """
        Decorator to register a function as an MCP resource with enhanced binary and streaming support.
        
        Args:
            uri: Resource URI (defaults to function name with resource:// scheme)
            name: Optional display name for the resource
            description: Optional description of the resource
            mime_type: MIME type of the resource content (auto-detected if not provided)
            content_type: Content type handling - "text", "binary", or "auto" (default)
            streaming: Whether this resource supports streaming for large datasets
            cache_ttl: Cache time-to-live in seconds (None = no caching)
            serializer: Custom serializer to use ("json", "xml", "yaml", "csv", etc.)
            max_size: Maximum content size in bytes (None = no limit)
            compression: Content compression ("gzip", "brotli", None)
            chunk_size: Chunk size for streaming in bytes (default: 8192)
            stream_type: Type of stream - "file", "data", "progress", "realtime" (default: "data")
            max_chunks: Maximum number of chunks for safety (None = no limit)
        """
        def decorator(func: Callable) -> Callable:
            resource_uri = uri or f"resource://{func.__name__}"
            resource_name = name or func.__name__
            
            # Validation: Check if resource URI already exists
            if resource_uri in self._resources:
                raise ResourceRegistrationError(
                    f"Resource with URI '{resource_uri}' is already registered",
                    resource_uri=resource_uri,
                    context={"existing_resources": list(self._resources.keys())}
                )
            
            # Validation: Ensure function is callable
            if not callable(func):
                raise ResourceRegistrationError(
                    f"Resource '{resource_uri}' must be a callable function",
                    resource_uri=resource_uri,
                    context={"provided_type": type(func).__name__}
                )
            
            # Validation: Check URI format
            if not self._validate_resource_uri(resource_uri):
                raise ResourceRegistrationError(
                    f"Invalid resource URI format: '{resource_uri}'. Must follow scheme://identifier pattern",
                    resource_uri=resource_uri,
                    recovery_suggestions=[
                        "Use format like 'config://database' or 'data://metrics'",
                        "Ensure URI has a valid scheme and identifier",
                        "Avoid special characters except :// and standard URI characters"
                    ]
                )
            
            # Validation: Check if function is async (recommended for MCP)
            import inspect
            if not inspect.iscoroutinefunction(func):
                import warnings
                warnings.warn(
                    f"Resource '{resource_uri}' is not async. Consider using 'async def' for better performance.",
                    UserWarning
                )
            
            resource_description = description or func.__doc__ or ""
            
            # Validation: Ensure description is provided
            if not resource_description.strip():
                raise ResourceRegistrationError(
                    f"Resource '{resource_uri}' must have a description (provide via parameter or docstring)",
                    resource_uri=resource_uri
                )
            
            # Validation: Check MIME type if provided
            if mime_type and not self._validate_mime_type(mime_type):
                import warnings
                warnings.warn(
                    f"Resource '{resource_uri}' has potentially invalid MIME type: '{mime_type}'",
                    UserWarning
                )
            
            try:
                # Register the resource with enhanced capabilities
                self._resources[resource_uri] = ResourceRegistration(
                    func=func,
                    uri=resource_uri,
                    name=resource_name,
                    description=resource_description,
                    mime_type=mime_type,
                    content_type=content_type,
                    streaming=streaming,
                    cache_ttl=cache_ttl,
                    serializer=serializer,
                    max_size=max_size,
                    compression=compression,
                    chunk_size=chunk_size,
                    stream_type=stream_type,
                    max_chunks=max_chunks,
                )
            except Exception as e:
                raise ResourceRegistrationError(
                    f"Failed to register resource '{resource_uri}': {str(e)}",
                    resource_uri=resource_uri,
                    original_exception=e,
                    recovery_suggestions=[
                        "Check parameter types and values",
                        "Verify content_type is 'text', 'binary', or 'auto'",
                        "Ensure cache_ttl and max_size are positive integers",
                        "Validate compression type if specified"
                    ]
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
        Decorator to register a function as an MCP prompt with enhanced validation.
        
        Args:
            name: Prompt name (defaults to function name)
            description: Optional description of the prompt
            arguments: List of argument definitions for the prompt
        """
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            
            # Validation: Check if prompt name already exists
            if prompt_name in self._prompts:
                raise PromptRegistrationError(
                    f"Prompt '{prompt_name}' is already registered",
                    prompt_name=prompt_name,
                    context={"existing_prompts": list(self._prompts.keys())}
                )
            
            # Validation: Ensure function is callable
            if not callable(func):
                raise PromptRegistrationError(
                    f"Prompt '{prompt_name}' must be a callable function",
                    prompt_name=prompt_name,
                    context={"provided_type": type(func).__name__}
                )
            
            # Validation: Check if function is async (recommended for MCP)
            import inspect
            if not inspect.iscoroutinefunction(func):
                import warnings
                warnings.warn(
                    f"Prompt '{prompt_name}' is not async. Consider using 'async def' for better performance.",
                    UserWarning
                )
            
            prompt_description = description or func.__doc__ or ""
            
            # Validation: Ensure description is provided
            if not prompt_description.strip():
                raise PromptRegistrationError(
                    f"Prompt '{prompt_name}' must have a description (provide via parameter or docstring)",
                    prompt_name=prompt_name
                )
            
            # Extract arguments from function signature if not provided
            prompt_arguments = arguments or []
            if not prompt_arguments and hasattr(func, "__annotations__"):
                try:
                    sig = inspect.signature(func)
                    for param_name, param in sig.parameters.items():
                        if param_name != "return":
                            # Try to get type information
                            param_type = "string"  # default
                            if param.annotation != inspect.Parameter.empty:
                                param_type = getattr(param.annotation, '__name__', str(param.annotation))
                            
                            arg_def = {
                                "name": param_name,
                                "description": f"Parameter {param_name} of type {param_type}",
                                "required": param.default == inspect.Parameter.empty,
                                "type": param_type
                            }
                            prompt_arguments.append(arg_def)
                except Exception as e:
                    raise PromptRegistrationError(
                        f"Failed to extract arguments for prompt '{prompt_name}': {str(e)}",
                        prompt_name=prompt_name,
                        original_exception=e
                    )
            
            # Validation: Check argument definitions format
            if prompt_arguments:
                for i, arg in enumerate(prompt_arguments):
                    if not isinstance(arg, dict):
                        raise PromptRegistrationError(
                            f"Prompt '{prompt_name}' argument {i} must be a dictionary",
                            prompt_name=prompt_name,
                            context={"argument_index": i, "argument_type": type(arg).__name__}
                        )
                    if "name" not in arg:
                        raise PromptRegistrationError(
                            f"Prompt '{prompt_name}' argument {i} must have a 'name' field",
                            prompt_name=prompt_name,
                            context={"argument_index": i, "argument": arg}
                        )
            
            try:
                # Register the prompt
                self._prompts[prompt_name] = PromptRegistration(
                    func=func,
                    name=prompt_name,
                    description=prompt_description,
                    arguments=prompt_arguments,
                )
            except Exception as e:
                raise PromptRegistrationError(
                    f"Failed to register prompt '{prompt_name}': {str(e)}",
                    prompt_name=prompt_name,
                    original_exception=e
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
    
    # =============================================================================
    # PRIVATE VALIDATION METHODS
    # =============================================================================
    
    def _validate_resource_uri(self, uri: str) -> bool:
        """Validate resource URI format.
        
        Args:
            uri: Resource URI to validate
            
        Returns:
            True if URI is valid, False otherwise
        """
        if not uri or not isinstance(uri, str):
            return False
        
        # Basic URI format: scheme://identifier
        if "://" not in uri:
            return False
        
        try:
            scheme, identifier = uri.split("://", 1)
            
            # Validate scheme (letters, numbers, +, -, .)
            if not scheme or not all(c.isalnum() or c in "+-." for c in scheme):
                return False
            
            # Validate identifier (non-empty, basic safe URI characters)
            if not identifier or not all(c.isalnum() or c in "-._~/:@" for c in identifier):
                return False
            
            return True
        except ValueError:
            return False
    
    def _validate_container_input(self, registration: ToolRegistration, arguments: Dict[str, Any]) -> Any:
        """Validate input based on container type information."""
        if not registration.input_type_info:
            # Fallback to direct model validation
            if registration.input_model:
                return registration.input_model(**arguments)
            return arguments
        
        type_info = registration.input_type_info
        param_name = type_info["param_name"]
        
        # Get the input data for the parameter
        if param_name in arguments:
            input_data = arguments[param_name]
        else:
            # Handle case where arguments are passed directly
            if len(arguments) == 1 and list(arguments.keys())[0] != param_name:
                # Single argument with wrong name - use its value
                input_data = list(arguments.values())[0]
            else:
                raise ValidationError(
                    f"Missing required parameter: {param_name}",
                    context={"expected_param": param_name, "provided_params": list(arguments.keys())}
                )
        
        # Validate based on container type
        if type_info["type"] == "direct":
            if isinstance(input_data, dict):
                return registration.input_model(**input_data)
            else:
                return registration.input_model(input_data)
        
        elif type_info["type"] == "union":
            # For union types, try to validate with the detected model
            if input_data is None and type(None) in type_info["union_args"]:
                return None  # Optional case
            try:
                return registration.input_model(**input_data) if isinstance(input_data, dict) else input_data
            except Exception:
                # If validation fails, return as-is (other union type)
                return input_data
        
        elif type_info["type"] == "list":
            # Validate list of models
            if not isinstance(input_data, list):
                raise ValidationError(
                    f"Expected list for parameter {param_name}, got {type(input_data).__name__}",
                    context={"expected_type": "list", "actual_type": type(input_data).__name__}
                )
            
            validated_items = []
            item_type = type_info["item_type"]
            for i, item in enumerate(input_data):
                try:
                    if isinstance(item, dict):
                        validated_items.append(item_type(**item))
                    elif hasattr(item, 'model_fields') or hasattr(item, '__fields__'):
                        # Already a Pydantic model
                        validated_items.append(item)
                    else:
                        validated_items.append(item)
                except Exception as e:
                    raise ValidationError(
                        f"Validation failed for item {i} in {param_name}: {str(e)}",
                        context={"item_index": i, "item_data": item}
                    )
            return validated_items
        
        elif type_info["type"] == "dict":
            # Validate dict with model values
            if not isinstance(input_data, dict):
                raise ValidationError(
                    f"Expected dict for parameter {param_name}, got {type(input_data).__name__}",
                    context={"expected_type": "dict", "actual_type": type(input_data).__name__}
                )
            
            validated_dict = {}
            value_type = type_info["value_type"]
            for key, value in input_data.items():
                try:
                    if isinstance(value, dict):
                        validated_dict[key] = value_type(**value)
                    elif hasattr(value, 'model_fields') or hasattr(value, '__fields__'):
                        # Already a Pydantic model
                        validated_dict[key] = value
                    else:
                        validated_dict[key] = value
                except Exception as e:
                    raise ValidationError(
                        f"Validation failed for key '{key}' in {param_name}: {str(e)}",
                        context={"key": key, "value_data": value}
                    )
            return validated_dict
        
        # Fallback
        return input_data

    def _validate_mime_type(self, mime_type: str) -> bool:
        """Validate MIME type format.
        
        Args:
            mime_type: MIME type to validate
            
        Returns:
            True if MIME type is valid, False otherwise
        """
        if not mime_type or not isinstance(mime_type, str):
            return False
        
        # Basic MIME type format: type/subtype
        if "/" not in mime_type:
            return False
        
        try:
            type_part, subtype_part = mime_type.split("/", 1)
            
            # Validate type and subtype (letters, numbers, hyphens)
            if not type_part or not all(c.isalnum() or c == "-" for c in type_part):
                return False
            
            if not subtype_part or not all(c.isalnum() or c in "-+." for c in subtype_part.split(";")[0]):
                return False
            
            return True
        except ValueError:
            return False
    
    # =============================================================================
    # BINARY RESOURCE UTILITIES
    # =============================================================================
    
    def _detect_content_type(self, content: Any, declared_mime_type: Optional[str] = None) -> str:
        """Detect content type from content and declared MIME type.
        
        Args:
            content: The content to analyze
            declared_mime_type: Explicitly declared MIME type
            
        Returns:
            Detected or default MIME type
        """
        # If explicitly declared, use it (after validation)
        if declared_mime_type and self._validate_mime_type(declared_mime_type):
            return declared_mime_type
        
        # Auto-detect based on content type
        if isinstance(content, bytes):
            # Try to detect from byte signatures
            if content.startswith(b'\x89PNG'):
                return "image/png"
            elif content.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            elif content.startswith(b'%PDF'):
                return "application/pdf"
            elif content.startswith(b'PK'):
                return "application/zip"
            else:
                return "application/octet-stream"
        
        elif isinstance(content, str):
            # Try to parse as JSON
            try:
                import json
                json.loads(content)
                return "application/json"
            except:
                pass
            
            # Check for XML-like content
            if content.strip().startswith('<') and content.strip().endswith('>'):
                return "application/xml"
            
            # Default to plain text
            return "text/plain"
        
        elif isinstance(content, dict):
            return "application/json"
        
        else:
            return "application/json"  # Will be serialized to JSON
    
    def _is_binary_content(self, mime_type: str) -> bool:
        """Check if MIME type represents binary content.
        
        Args:
            mime_type: MIME type to check
            
        Returns:
            True if content is binary, False if text
        """
        text_types = [
            "text/", 
            "application/json", 
            "application/xml",
            "application/javascript", 
            "application/yaml",
            "application/x-yaml",
            "application/csv",
            "text/csv",
        ]
        return not any(mime_type.startswith(t) for t in text_types)
    
    def _encode_binary_content(self, content: bytes) -> str:
        """Encode binary content to base64 for MCP transport.
        
        Args:
            content: Binary content to encode
            
        Returns:
            Base64-encoded string
        """
        import base64
        return base64.b64encode(content).decode('utf-8')
    
    def _create_mcp_resource_content(
        self, 
        uri: str, 
        content: Any, 
        mime_type: str
    ) -> Dict[str, Any]:
        """Create MCP-compliant resource content.
        
        Args:
            uri: Resource URI
            content: Content to format
            mime_type: MIME type of content
            
        Returns:
            MCP resource content dictionary
        """
        from mcp.types import BlobResourceContents, TextResourceContents
        import json
        
        try:
            # Determine if content should be treated as binary
            is_binary = False
            
            if isinstance(content, bytes):
                is_binary = True
            elif self._is_binary_content(mime_type):
                is_binary = True
            
            if is_binary:
                # Handle binary content
                if isinstance(content, str):
                    content_bytes = content.encode('utf-8')
                elif isinstance(content, bytes):
                    content_bytes = content
                else:
                    # Serialize to JSON first, then encode
                    json_str = json.dumps(content, indent=2)
                    content_bytes = json_str.encode('utf-8')
                
                blob_data = self._encode_binary_content(content_bytes)
                blob_content = BlobResourceContents(
                    uri=uri,
                    mimeType=mime_type,
                    blob=blob_data
                )
                
                return {
                    "contents": [blob_content.model_dump()]
                }
            
            else:
                # Handle text content
                if isinstance(content, bytes):
                    text_content = content.decode('utf-8')
                elif isinstance(content, str):
                    text_content = content
                else:
                    # Serialize to JSON
                    text_content = json.dumps(content, indent=2)
                
                text_resource = TextResourceContents(
                    uri=uri,
                    mimeType=mime_type,
                    text=text_content
                )
                
                return {
                    "contents": [text_resource.model_dump()]
                }
                
        except Exception as e:
            # Fallback to text content with error info
            error_content = f"Error formatting resource content: {str(e)}"
            text_resource = TextResourceContents(
                uri=uri,
                mimeType="text/plain",
                text=error_content
            )
            
            return {
                "contents": [text_resource.model_dump()]
            }
    
    # =============================================================================
    # STREAMING RESOURCE UTILITIES
    # =============================================================================
    
    async def _handle_streaming_resource(
        self, 
        uri: str, 
        content: Any, 
        registration: ResourceRegistration
    ) -> Dict[str, Any]:
        """Handle streaming resource content processing."""
        import time
        import uuid
        
        # Configure streaming handler with registration settings
        self._streaming_handler.chunk_size = registration.chunk_size
        if registration.max_chunks:
            self._streaming_handler.max_chunks = registration.max_chunks
        
        try:
            # Determine streaming approach based on content type and stream_type
            if registration.stream_type == "file" and isinstance(content, str):
                # Handle file path streaming
                chunks = await self._streaming_handler.create_file_stream(
                    content, 
                    chunk_size=registration.chunk_size
                )
            elif hasattr(content, '__aiter__') or hasattr(content, '__iter__'):
                # Handle generator/iterable streaming
                chunks = await self._streaming_handler.create_data_stream(
                    content, 
                    stream_type=registration.stream_type
                )
            else:
                # Convert single content to single chunk
                stream_id = f"single_stream_{uuid.uuid4().hex[:8]}"
                chunk = StreamChunk(
                    stream_id=stream_id,
                    chunk_index=0,
                    data=content,
                    is_final=True
                )
                chunks = [chunk]
            
            # Create streaming response format
            if chunks:
                stream_id = chunks[0].stream_id
                
                # Get or create manifest
                manifest = self._streaming_handler.get_stream_manifest(stream_id)
                if not manifest:
                    # Create manifest for single content
                    manifest = StreamManifest(
                        stream_id=stream_id,
                        stream_type=registration.stream_type,
                        content_type=registration.mime_type or "application/json",
                        chunk_count=len(chunks),
                        metadata={
                            "uri": uri,
                            "chunk_size": registration.chunk_size,
                            "source_type": type(content).__name__
                        }
                    )
                
                # Create MCP-compatible streaming response
                return self._create_streaming_mcp_response(uri, chunks, manifest, registration)
            else:
                # Fallback to regular content handling
                detected_mime_type = self._detect_content_type(content, registration.mime_type)
                return self._create_mcp_resource_content(uri, content, detected_mime_type)
                
        except Exception as e:
            raise ResourceAccessError(
                f"Streaming processing failed for '{uri}': {str(e)}",
                resource_uri=uri,
                original_exception=e
            )
    
    def _create_streaming_mcp_response(
        self, 
        uri: str, 
        chunks: List[StreamChunk], 
        manifest: StreamManifest,
        registration: ResourceRegistration
    ) -> Dict[str, Any]:
        """Create MCP-compatible response for streaming content."""
        from mcp.types import TextResourceContents
        import json
        
        # For MCP compatibility, we'll return the stream as a structured text response
        # containing the manifest and chunk information
        
        stream_data = {
            "streaming": True,
            "manifest": {
                "stream_id": manifest.stream_id,
                "stream_type": manifest.stream_type,
                "content_type": manifest.content_type,
                "total_size": manifest.total_size,
                "chunk_count": len(chunks),
                "created_at": manifest.created_at,
                "metadata": manifest.metadata
            },
            "chunks": []
        }
        
        # Process chunks based on content type
        for chunk in chunks:
            chunk_data = {
                "chunk_index": chunk.chunk_index,
                "is_final": chunk.is_final,
                "timestamp": chunk.timestamp,
                "size": chunk.size
            }
            
            # Handle chunk data encoding
            if isinstance(chunk.data, bytes):
                if self._is_binary_content(registration.mime_type or "application/octet-stream"):
                    # Encode binary data as base64
                    chunk_data["data"] = self._encode_binary_content(chunk.data)
                    chunk_data["encoding"] = "base64"
                else:
                    # Decode bytes to text
                    try:
                        chunk_data["data"] = chunk.data.decode('utf-8')
                        chunk_data["encoding"] = "utf-8"
                    except UnicodeDecodeError:
                        chunk_data["data"] = self._encode_binary_content(chunk.data)
                        chunk_data["encoding"] = "base64"
            else:
                # Serialize non-bytes data to JSON
                chunk_data["data"] = chunk.data
                chunk_data["encoding"] = "json"
            
            stream_data["chunks"].append(chunk_data)
        
        # Create text resource with streaming data (handle AnyUrl serialization)
        streaming_json = json.dumps(stream_data, indent=2, default=str)
        text_resource = TextResourceContents(
            uri=uri,
            mimeType="application/json",
            text=streaming_json
        )
        
        return {
            "contents": [text_resource.model_dump()]
        }
    
    def _convert_to_mcp_result(self, resource_dict: Dict[str, Any]):
        """Convert resource dictionary to proper MCP ReadResourceResult."""
        from mcp.types import ReadResourceResult, TextResourceContents, BlobResourceContents
        
        if "contents" not in resource_dict:
            raise ResourceAccessError("Invalid resource format: missing 'contents' field")
        
        mcp_contents = []
        for content_item in resource_dict["contents"]:
            if isinstance(content_item, dict):
                if "text" in content_item:
                    mcp_content = TextResourceContents(**content_item)
                elif "blob" in content_item:
                    mcp_content = BlobResourceContents(**content_item)
                else:
                    raise ResourceAccessError(f"Invalid content format: missing 'text' or 'blob' field")
                mcp_contents.append(mcp_content)
            else:
                # Already an MCP content object
                mcp_contents.append(content_item)
        
        return ReadResourceResult(contents=mcp_contents)