# LightMCP

A FastAPI-inspired framework for building Model Context Protocol (MCP) servers with familiar patterns and dual-protocol support.

## Features

- **Dual Protocol Support**: Build services that expose both REST APIs and MCP tools from the same codebase
- **FastAPI-like Experience**: Use familiar decorators and patterns from FastAPI
- **Type Safety**: Full Pydantic integration for input/output validation
- **Async First**: Built on async/await for handling concurrent requests
- **Production Ready**: Includes validation, error handling, and transport management

## Quick Start

```python
from lightmcp import LightMCP
from pydantic import BaseModel

app = LightMCP()

class PredictionInput(BaseModel):
    text: str
    temperature: float = 0.7

@app.tool()  # Exposes as MCP tool
@app.post("/predict")  # Also exposes as REST endpoint  
async def predict(input: PredictionInput) -> dict:
    return {"prediction": f"Processed: {input.text}"}

# Run as MCP server
if __name__ == "__main__":
    import asyncio
    asyncio.run(app.run_mcp())
```

## Installation

```bash
pip install lightmcp
```

## Running the Server

### As MCP Server (stdio transport):
```bash
python server.py --mode mcp
```

### As HTTP Server:
```bash
python server.py --mode http --port 8000
```

## Example

See the `examples/basic_server.py` for a complete example showing:
- Dual protocol endpoints
- Input validation with Pydantic
- MCP-only tools
- REST-only endpoints
- Async request handling

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check --fix src tests
```

## License

MIT