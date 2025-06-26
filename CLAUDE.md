# LightMCP Project Memory

## Project Objective
LightMCP is a FastAPI-inspired Python framework that enables developers to build Model Context Protocol (MCP) servers using familiar FastAPI patterns. The framework allows developers to create services that expose both traditional REST APIs and MCP tools from the same codebase.

## Core Value Proposition
- **Dual Protocol Support**: Single codebase that exposes both HTTP REST endpoints and MCP tools
- **Familiar Developer Experience**: Uses FastAPI-like decorators and patterns
- **Production Ready**: Built-in validation, dependency injection, and error handling
- **MCP Native**: Handles JSON-RPC transport, session management, and MCP lifecycle automatically

## Key Design Principles
1. **Developer Ergonomics**: Should feel as natural as FastAPI for web developers
2. **Protocol Abstraction**: Hide MCP complexity while remaining configurable
3. **Type Safety**: Leverage Pydantic for validation across both protocols
4. **Performance**: Async-first design for handling concurrent AI agent requests
5. **Extensibility**: Plugin architecture for custom transports and middleware

## Target Architecture
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
    return {"prediction": "example"}
```

## Technical Stack
- Python 3.11+
- FastAPI patterns and dependency injection
- Pydantic for validation
- Official MCP Python SDK for protocol implementation
- Support for stdio and HTTP+SSE transports
- JSON-RPC 2.0 compliance

## Success Metrics
- Reduce MCP server development time by 70%
- Maintain FastAPI-level developer experience
- Support all standard MCP capabilities (tools, resources, prompts)
- Enable seamless migration from FastAPI projects