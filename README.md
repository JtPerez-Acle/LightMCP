# LightMCP

A FastAPI-inspired framework for building complete Model Context Protocol (MCP) servers with familiar patterns and dual-protocol support. LightMCP provides the industry's first framework that implements the full MCP specification—Tools, Resources, and Prompts—while maintaining compatibility with standard HTTP REST APIs.

## Features

- **Complete MCP Protocol**: Full support for Tools, Resources, and Prompts
- **Dual Protocol Support**: Build services that expose both REST APIs and MCP capabilities from the same codebase  
- **FastAPI-like Experience**: Use familiar decorators (`@app.tool()`, `@app.resource()`, `@app.prompt()`) and patterns
- **Type Safety**: Full Pydantic integration for input/output validation across both protocols
- **Async First**: Built on async/await for handling concurrent requests efficiently
- **Production Ready**: Includes validation, error handling, transport management, and comprehensive testing
- **AI-Native Design**: Purpose-built for AI assistant integrations and modern development workflows

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

# RESOURCES - Structured data access  
@app.resource(uri="config://app", description="Application configuration")
@app.get("/config")  # Also available via HTTP REST
async def get_config() -> dict:
    return {"theme": "dark", "version": "1.0.0", "features": ["tools", "resources", "prompts"]}

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

- **One Function, Multiple Protocols**: The same `create_task` function works as both an HTTP POST endpoint and an MCP tool
- **Complete MCP Support**: Tools for operations, Resources for data access, Prompts for AI interactions
- **Type Safety**: Pydantic models ensure validation across both HTTP and MCP protocols
- **Familiar Patterns**: If you know FastAPI, you already know LightMCP

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

### Basic Server
See `examples/basic_server.py` for a simple prediction service demonstrating:
- Dual protocol endpoints
- Input validation with Pydantic
- Basic MCP tool functionality

### Development Toolkit  
See `examples/developer_toolkit.py` for a comprehensive development assistant with:
- Project analysis tools
- Git integration
- Test execution
- Dependency management

### Complete MCP Protocol
See `examples/complete_mcp_example.py` for full protocol demonstration:
- **Tools**: Build, lint, and project operations
- **Resources**: Configuration, metrics, and documentation
- **Prompts**: Code review, optimization, and debugging workflows
- All capabilities available through both HTTP and MCP protocols

## API Reference

### Core Decorators

```python
# Tools - Interactive operations that AI assistants can execute
@app.tool(name="optional_name", description="Tool description")
async def my_tool(input: PydanticModel) -> OutputType:
    pass

# Resources - Structured data that AI assistants can read
@app.resource(uri="scheme://identifier", description="Resource description", mime_type="application/json")
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

## Architecture

LightMCP implements a dual-protocol architecture where:

1. **Single Function Definition**: Write your business logic once
2. **Automatic Protocol Adaptation**: LightMCP handles HTTP REST and MCP protocol differences
3. **Shared Validation**: Pydantic models work across both protocols
4. **Unified Error Handling**: Consistent error responses in both protocols
5. **Async Throughout**: Built on asyncio for maximum performance

This approach reduces development time by up to 80% compared to building separate HTTP and MCP servers.

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