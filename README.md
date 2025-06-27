# LightMCP

A FastAPI-inspired framework for building Model Context Protocol (MCP) servers with familiar patterns and dual-protocol support. LightMCP implements the complete MCP specification (Tools, Resources, Prompts) while maintaining compatibility with HTTP REST APIs.

## Features

- **Complete MCP Protocol**: Tools, Resources, and Prompts support
- **Dual Protocol**: Same codebase exposes both REST APIs and MCP capabilities
- **FastAPI-like API**: Familiar decorators (`@app.tool()`, `@app.resource()`, `@app.prompt()`)
- **Type Safety**: Pydantic validation and automatic schema generation
- **Streaming Support**: Efficient handling of large files and real-time data
- **Binary Content**: Automatic base64 encoding for images, PDFs, and other binary data
- **Production Ready**: Comprehensive error handling, validation, and testing

## Quick Start

### Complete MCP Server Example

```python
from lightmcp import LightMCP
from pydantic import BaseModel

app = LightMCP(name="AI Development Assistant", version="1.0.0")

class TaskInput(BaseModel):
    title: str
    priority: int = 1

# TOOLS - Interactive operations
@app.tool(description="Create a new task")
@app.post("/tasks")  # Also available via HTTP REST
async def create_task(task: TaskInput) -> dict:
    return {"id": 123, "title": task.title, "priority": task.priority}

# RESOURCES - Structured data access (with streaming support)
@app.resource(
    uri="data://logs", 
    description="System logs",
    streaming=True,  # Enable streaming for large datasets
    chunk_size=1024
)
async def get_logs() -> str:
    return "/var/log/app.log"  # File path triggers automatic streaming

# PROMPTS - AI workflow templates
@app.prompt(description="Code review checklist generator")
async def code_review_prompt(file_path: str, focus: str = "general") -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"Review {file_path} focusing on {focus}"}
        ]
    }

# Run as MCP server
if __name__ == "__main__":
    import asyncio
    asyncio.run(app.run_mcp())
```

### Key Benefits

- **Write Once, Deploy Everywhere**: Same function works as both HTTP endpoint and MCP capability
- **Streaming by Default**: Handle large files and real-time data efficiently
- **Type Safety**: Pydantic validation across both protocols
- **Zero Learning Curve**: Uses FastAPI patterns you already know

## Installation

```bash
pip install lightmcp
```

*Note: LightMCP is currently in development. Clone the repository and install in development mode:*

```bash
git clone https://github.com/your-org/lightmcp
cd lightmcp
pip install -e ".[dev]"
```

## Running Your Server

### As MCP Server (stdio transport for AI assistants):
```bash
python your_server.py --mode mcp
```

### As HTTP Server (REST API for web applications):
```bash
python your_server.py --mode http --port 8000
```

### Dual Mode Support
Your LightMCP server automatically supports both protocols. You can:
- Start it as an HTTP server for web dashboard integration
- Start it as an MCP server for AI assistant integration  
- Use the same codebase for both without modification

## Real-World Examples

### Examples

- **Basic Server** (`examples/basic_server.py`): Simple prediction service with validation
- **Developer Toolkit** (`examples/developer_toolkit.py`): Git integration, code analysis, test execution
- **Complete MCP Example** (`examples/complete_mcp_example.py`): All three MCP capabilities demonstrated

## API Reference

### Core Decorators

```python
# Tools - Interactive operations that AI assistants can execute
@app.tool(name="optional_name", description="Tool description")
async def my_tool(input: PydanticModel) -> OutputType:
    pass

# Resources - Structured data that AI assistants can read
@app.resource(
    uri="scheme://identifier", 
    description="Resource description",
    streaming=True,  # Enable for large files
    chunk_size=8192  # Bytes per chunk
)
async def my_resource() -> DataType:
    pass

# Prompts - Templates for AI interactions  
@app.prompt(name="prompt_name", description="Prompt description", arguments=[...])
async def my_prompt(param: str) -> dict:
    return {"messages": [...]}
```

### HTTP Endpoints
All standard FastAPI decorators are supported and can be combined with MCP decorators:

```python
@app.get("/path")
@app.post("/path") 
@app.put("/path")
@app.delete("/path")
@app.patch("/path")
```

### Server Modes

```python
# Run as MCP server (stdio transport)
await app.run_mcp()

# Run as HTTP server  
app.run_http(host="0.0.0.0", port=8000)

# Access underlying FastAPI app
fastapi_app = app.fastapi_app

# Access underlying MCP server
mcp_server = app.mcp_server
```

## Use Cases

### For AI Assistant Developers
- Build tools that AI assistants can execute directly
- Provide structured data access through resources
- Create workflow templates with prompts
- Maintain the same codebase for testing via HTTP APIs

### For Application Developers  
- Add AI assistant capabilities to existing FastAPI applications
- Expose internal tools and data to AI systems
- Create hybrid applications supporting both human and AI users
- Implement AI-powered features with familiar patterns

### For DevOps and Platform Teams
- Create development tooling accessible by both humans and AI
- Build monitoring and management interfaces with dual access
- Implement CI/CD integrations that work with AI assistants
- Provide infrastructure APIs with AI-native interfaces

## Advanced Features

### Streaming Resources
```python
@app.resource(uri="data://large-file", streaming=True, chunk_size=1024)
async def stream_large_file() -> str:
    return "/path/to/large/file.log"  # Automatically streamed in chunks
```

### Binary Content
```python
@app.resource(uri="image://logo", mime_type="image/png")
async def get_logo() -> bytes:
    return b'\x89PNG\r\n\x1a\n...'  # Automatically base64 encoded for MCP
```

### Container Type Support
```python
@app.tool()
async def process_items(items: List[ItemModel]) -> Dict[str, ResultModel]:
    # Full validation for List[Model] and Dict[str, Model]
    return {"processed": ResultModel(...)}
```

## Architecture

LightMCP uses a dual-protocol architecture:
- Single function definition for business logic
- Automatic protocol adaptation (HTTP REST ↔ MCP)
- Shared Pydantic validation
- Unified error handling with context and recovery suggestions
- Streaming support for large data
- Async-first design

## Development and Testing

```bash
# Install in development mode
pip install -e ".[dev]"

# Run comprehensive test suite
pytest tests/

# Run specific test categories
pytest tests/test_e2e_http.py          # HTTP protocol tests
pytest tests/test_e2e_mcp.py           # MCP protocol tests
pytest tests/test_e2e_dual_protocol.py # Dual protocol validation

# Run complete protocol validation
python test_complete_mcp_protocol.py

# Format code
black src tests examples
ruff check --fix src tests examples

# Type checking
mypy src
```

## Contributing

LightMCP is in active development. We welcome contributions, especially:

- Additional MCP protocol features
- Performance optimizations  
- Documentation improvements
- Real-world usage examples
- Integration with popular AI frameworks

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built on top of:
- [FastAPI](https://fastapi.tiangolo.com/) for HTTP server capabilities
- [Pydantic](https://pydantic.dev/) for data validation  
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for Model Context Protocol support

LightMCP bridges the gap between traditional web APIs and the emerging AI assistant ecosystem, providing developers with a unified framework for building the next generation of AI-integrated applications.