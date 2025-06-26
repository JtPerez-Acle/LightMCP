# LightMCP Project Memory

## PROJECT STATUS: Phase 2.1 - Advanced Resource Capabilities
**Version**: 2.1.3  
**Last Updated**: 2025-06-26  

LightMCP is a FastAPI-inspired Python framework for building Model Context Protocol (MCP) servers. The framework supports the complete MCP specification (Tools, Resources, Prompts) with dual-protocol exposure through both HTTP REST and native MCP.

## Core Features
- **Complete MCP Protocol**: Tools, Resources, and Prompts support
- **Dual Protocol**: Single codebase exposes both HTTP REST and MCP
- **FastAPI-like API**: Familiar decorators (`@app.tool()`, `@app.resource()`, `@app.prompt()`)
- **Type Safety**: Pydantic validation, automatic schema generation
- **Production Features**: Exception handling, streaming resources, binary content support

## Current Implementation Status

### Completed Components
- **Core Framework** (`src/lightmcp/app.py`): LightMCP class with full MCP protocol support
- **Exception System** (`src/lightmcp/exceptions.py`): Comprehensive error handling with context and recovery suggestions
- **Type System**: Enhanced detection for container types (List[Model], Dict[str, Model], Union types)
- **Binary Resources**: Base64 encoding for binary content, automatic MIME type detection
- **Streaming Resources**: Chunked processing for large files and real-time data streams
- **MCP Compatibility**: Fixed handler signatures and return formats (MCP 1.9.4)

### Test Coverage
- All core tests passing (unit, E2E HTTP, E2E MCP, dual protocol, validation)
- Binary resource handling validated
- Streaming functionality tested with files, generators, and real-time data
- MCP protocol compliance verified

### Examples
- **Basic Server** (`examples/basic_server.py`): Simple prediction service
- **Developer Toolkit** (`examples/developer_toolkit.py`): Git integration, code analysis, test execution
- **Complete MCP Example** (`examples/complete_mcp_example.py`): Demonstrates all three MCP capabilities

## Architecture Overview

```python
from lightmcp import LightMCP
from pydantic import BaseModel

app = LightMCP()

class TaskInput(BaseModel):
    title: str
    priority: int = 1

# TOOLS - Function execution
@app.tool(description="Create a task")  # ← Exposes as MCP tool
@app.post("/tasks")                     # ← ALSO exposes as REST endpoint
async def create_task(task: TaskInput) -> dict:
    return {"id": 123, "title": task.title}

# RESOURCES - Structured data access (with streaming support)
@app.resource(
    uri="data://metrics", 
    description="System metrics",
    streaming=True,  # ← NEW: Enable streaming
    chunk_size=1024
)
async def get_metrics() -> list:
    return [{"timestamp": i, "cpu": i*10} for i in range(100)]

# PROMPTS - AI interaction templates
@app.prompt(description="Code review", arguments=[
    {"name": "file_path", "required": True}
])
async def review_prompt(file_path: str) -> dict:
    return {"messages": [{"role": "user", "content": f"Review {file_path}"}]}
```

## Technical Details

### Core Components
- **LightMCP Class** (`src/lightmcp/app.py`): Main application with registries for tools, resources, and prompts
- **StreamingResourceHandler**: Handles chunked file reading and data streaming
- **Exception Hierarchy**: Custom exceptions with context and recovery suggestions
- **Type Detection**: Enhanced support for container types and Pydantic models

### Key Features
- **Enhanced Resource Registration**: Support for binary content, streaming, custom serializers
- **MCP Handler Compatibility**: Fixed to work with MCP 1.9.4 (returns simple strings, not complex objects)
- **Streaming Support**: Chunk-based processing for large files with configurable limits
- **Binary Content**: Automatic base64 encoding for MCP transport

### Dependencies
- Python 3.11+ | FastAPI 0.115.13 | Pydantic 2.11.7 | MCP 1.9.4 | Uvicorn 0.34.3

## Known Issues and Limitations

### MCP Library Compatibility
- MCP 1.9.4 expects read_resource handlers to return simple strings, not complex objects
- Workaround implemented: All content is JSON-serialized to strings
- Future versions may support proper ReadResourceResult objects

### Streaming Limitations
- Async generators must be materialized to lists for MCP transport
- Large files are chunked but still loaded into memory during processing
- Real-time streaming requires polling rather than true push updates

### Type Detection
- Generic types beyond simple containers not fully supported
- Complex nested unions may not validate correctly
- Workaround: Use explicit Pydantic models for complex types

## Development Roadmap

### Phase 1 - Core Robustness ✅ COMPLETED
- [x] Comprehensive exception hierarchy with context and recovery suggestions
- [x] Enhanced type detection for container types (List[Model], Dict[str, Model])
- [x] URI validation and resource collision detection
- [x] Graceful async error handling

### Phase 2.1 - Advanced Resource Capabilities (CURRENT)
- [x] Binary content support with base64 encoding
- [x] Streaming resources for large files and data generators
- [x] MCP handler compatibility fixes
- [ ] Custom serializers (XML, YAML, CSV, Protobuf) - **NEXT**
- [ ] Resource versioning and caching strategies

### Phase 2.2 - Enhanced Content Handling
- [ ] MIME type negotiation
- [ ] Content compression (gzip, brotli)
- [ ] Multi-format responses

### Phase 2.3 - Middleware Architecture
- [ ] Request/response pipeline
- [ ] Custom validation middleware
- [ ] Plugin system

### Phase 3 - Production Features
- [ ] Structured logging and metrics
- [ ] Authentication and rate limiting
- [ ] Performance optimizations

### Phase 4 - Developer Experience
- [ ] CLI tools
- [ ] Hot reload
- [ ] IDE integration

## Project Structure
```
LightMCP/
├── src/lightmcp/
│   ├── app.py                    # Core LightMCP class with streaming support
│   └── exceptions.py             # Comprehensive exception hierarchy
├── examples/
│   ├── basic_server.py          # Simple prediction service
│   ├── developer_toolkit.py     # Git, code analysis, test execution
│   └── complete_mcp_example.py  # Full protocol demonstration
├── tests/                       # All core tests passing
├── pyproject.toml              # Dependencies and package config
└── CLAUDE.md                   # Project memory for AI assistants
```

## Key Implementation Notes

### MCP Handler Signatures
The MCP library (1.9.4) expects specific handler signatures:
- `read_resource(uri)` receives AnyUrl object, must return string
- Tool handlers receive raw arguments dict
- All content must be JSON-serializable

### Streaming Implementation
- Resources with `streaming=True` process data in chunks
- File paths trigger automatic file streaming
- Generators and iterables are materialized to lists
- Results are JSON-serialized with chunk metadata

### Type System Enhancement
- Container types detected via `typing.get_origin()` and `get_args()`
- Pydantic models in List[Model] and Dict[str, Model] are validated
- Union types check each variant for Pydantic models

### Testing Philosophy
- No mocking - all tests verify real implementation
- E2E tests run actual HTTP servers and MCP handlers
- Streaming tested with real files and data generators