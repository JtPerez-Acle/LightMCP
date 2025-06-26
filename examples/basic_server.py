"""
Basic example of a LightMCP server with dual-protocol support.

This example shows how to create a simple prediction service that's
accessible both as an MCP tool and as a REST API endpoint.
"""

import asyncio
import sys
from typing import Dict, Any

from pydantic import BaseModel, Field

# Add parent directory to path for development
sys.path.insert(0, "../src")

from lightmcp import LightMCP

# Initialize the LightMCP application
app = LightMCP(
    name="Prediction Service",
    version="1.0.0",
    description="A simple ML prediction service with dual protocol support",
)


# Define input/output models using Pydantic
class PredictionInput(BaseModel):
    """Input model for predictions."""
    text: str = Field(..., description="The text to analyze")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for predictions"
    )
    max_tokens: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Maximum tokens to generate"
    )


class PredictionOutput(BaseModel):
    """Output model for predictions."""
    prediction: str
    confidence: float
    metadata: Dict[str, Any]


# Register a function as both MCP tool and REST endpoint
@app.tool(description="Generate a prediction based on input text")
@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput) -> PredictionOutput:
    """
    Generate a prediction for the given text input.
    
    This function is exposed as:
    - MCP tool: 'predict'
    - REST endpoint: POST /predict
    """
    # Simulate some async processing
    await asyncio.sleep(0.1)
    
    # Mock prediction logic
    prediction_text = f"Based on '{input.text[:50]}...', the prediction is..."
    confidence = 0.85 * (1 - input.temperature * 0.2)
    
    return PredictionOutput(
        prediction=prediction_text,
        confidence=confidence,
        metadata={
            "model": "mock-model-v1",
            "temperature": input.temperature,
            "max_tokens": input.max_tokens,
            "input_length": len(input.text),
        }
    )


# Additional REST-only endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint (REST only)."""
    return {
        "status": "healthy",
        "service": app.name,
        "version": app.version,
    }


# MCP-only tool (no REST endpoint)
@app.tool(description="Get information about available models")
async def list_models() -> Dict[str, Any]:
    """List available prediction models."""
    return {
        "models": [
            {
                "id": "mock-model-v1",
                "name": "Mock Model v1",
                "description": "A mock model for demonstration",
                "capabilities": ["text-generation", "sentiment-analysis"],
            },
            {
                "id": "mock-model-v2",
                "name": "Mock Model v2",
                "description": "An improved mock model",
                "capabilities": ["text-generation", "summarization", "translation"],
            }
        ],
        "default": "mock-model-v1",
    }


# Entry point for running the server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightMCP Example Server")
    parser.add_argument(
        "--mode",
        choices=["mcp", "http"],
        default="mcp",
        help="Run as MCP server (stdio) or HTTP server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP server (default: 8000)",
    )
    
    args = parser.parse_args()
    
    if args.mode == "mcp":
        print("Starting MCP server on stdio...", file=sys.stderr)
        asyncio.run(app.run_mcp())
    else:
        print(f"Starting HTTP server on {args.host}:{args.port}...", file=sys.stderr)
        app.run_http(host=args.host, port=args.port)