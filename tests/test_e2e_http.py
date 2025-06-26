"""End-to-end tests for HTTP protocol functionality."""

import asyncio
import json
import multiprocessing
import sys
import time
from typing import Any, Dict

import httpx
import pytest
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, "../src")

from lightmcp import LightMCP


class TestHTTPServer:
    """Test server for E2E HTTP testing."""
    
    def __init__(self):
        self.app = LightMCP(
            name="E2E HTTP Test Server",
            version="1.0.0",
            description="Server for end-to-end HTTP testing"
        )
        self._setup_endpoints()
    
    def _setup_endpoints(self):
        """Set up test endpoints."""
        
        class UserInput(BaseModel):
            name: str
            age: int
            email: str = None
        
        class UserResponse(BaseModel):
            id: int
            name: str
            age: int
            email: str = None
            status: str = "active"
        
        class CalculationInput(BaseModel):
            x: float
            y: float
            operation: str = "add"
        
        # REST-only endpoints
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": self.app.name,
                "version": self.app.version,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        
        @self.app.get("/users/{user_id}")
        async def get_user(user_id: int):
            """Get user by ID."""
            return {
                "id": user_id,
                "name": f"User {user_id}",
                "age": 25 + (user_id % 50),
                "email": f"user{user_id}@example.com"
            }
        
        @self.app.post("/users", response_model=UserResponse)
        async def create_user(user: UserInput) -> UserResponse:
            """Create a new user."""
            return UserResponse(
                id=12345,
                name=user.name,
                age=user.age,
                email=user.email or f"{user.name.lower()}@example.com"
            )
        
        # Dual-protocol endpoints (both HTTP and MCP)
        @self.app.tool(description="Perform mathematical calculations")
        @self.app.post("/calculate")
        async def calculate(input: CalculationInput) -> Dict[str, Any]:
            """Calculate mathematical operations."""
            operations = {
                "add": input.x + input.y,
                "subtract": input.x - input.y,
                "multiply": input.x * input.y,
                "divide": input.x / input.y if input.y != 0 else None
            }
            
            if input.operation not in operations:
                raise ValueError(f"Unsupported operation: {input.operation}")
            
            result = operations[input.operation]
            if result is None:
                raise ValueError("Division by zero")
            
            return {
                "result": result,
                "operation": input.operation,
                "operands": [input.x, input.y]
            }
        
        @self.app.tool(description="Process text data")
        @self.app.post("/process-text")
        async def process_text(text: str, uppercase: bool = False) -> Dict[str, Any]:
            """Process text with various transformations."""
            result = text
            if uppercase:
                result = result.upper()
            
            return {
                "original": text,
                "processed": result,
                "length": len(result),
                "word_count": len(result.split()),
                "uppercase": uppercase
            }


def run_http_server(port: int = 8001):
    """Run HTTP server in subprocess."""
    import uvicorn
    
    server = TestHTTPServer()
    uvicorn.run(server.app.fastapi_app, host="127.0.0.1", port=port, log_level="critical")


@pytest.fixture(scope="module")
def http_server():
    """Start HTTP server for testing."""
    port = 8001
    process = multiprocessing.Process(target=run_http_server, args=(port,))
    process.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Verify server is running
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(10):  # Try for 5 seconds
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                break
        except:
            time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("HTTP server failed to start")
    
    yield base_url
    
    # Cleanup
    process.terminate()
    process.join(timeout=5)


@pytest.mark.asyncio
async def test_http_health_check(http_server):
    """Test basic health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{http_server}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "E2E HTTP Test Server"
        assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_http_path_parameters(http_server):
    """Test endpoint with path parameters."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{http_server}/users/42")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 42
        assert data["name"] == "User 42"
        assert "user42@example.com" in data["email"]


@pytest.mark.asyncio
async def test_http_post_with_validation(http_server):
    """Test POST endpoint with Pydantic validation."""
    async with httpx.AsyncClient() as client:
        user_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        response = await client.post(f"{http_server}/users", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 12345
        assert data["name"] == "John Doe"
        assert data["age"] == 30
        assert data["email"] == "john@example.com"
        assert data["status"] == "active"


@pytest.mark.asyncio
async def test_http_validation_error(http_server):
    """Test HTTP validation errors."""
    async with httpx.AsyncClient() as client:
        # Missing required field
        invalid_data = {"name": "John"}  # Missing age
        
        response = await client.post(f"{http_server}/users", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data


@pytest.mark.asyncio
async def test_http_dual_protocol_endpoint(http_server):
    """Test endpoint that's available as both HTTP and MCP tool."""
    async with httpx.AsyncClient() as client:
        calc_data = {
            "x": 10.5,
            "y": 3.2,
            "operation": "multiply"
        }
        
        response = await client.post(f"{http_server}/calculate", json=calc_data)
        
        assert response.status_code == 200
        data = response.json()
        assert abs(data["result"] - 33.6) < 0.001  # Float comparison
        assert data["operation"] == "multiply"
        assert data["operands"] == [10.5, 3.2]


@pytest.mark.asyncio
async def test_http_business_logic_error(http_server):
    """Test business logic errors (e.g., division by zero)."""
    async with httpx.AsyncClient() as client:
        calc_data = {
            "x": 5.0,
            "y": 0.0,
            "operation": "divide"
        }
        
        response = await client.post(f"{http_server}/calculate", json=calc_data)
        
        assert response.status_code == 500  # Internal server error
        # The exact error format depends on FastAPI's error handling


@pytest.mark.asyncio 
async def test_http_query_parameters(http_server):
    """Test endpoint with query parameters."""
    async with httpx.AsyncClient() as client:
        # Test with query parameters that map to function parameters
        response = await client.post(
            f"{http_server}/process-text",
            json="hello world",
            params={"uppercase": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["original"] == "hello world"
        assert data["processed"] == "HELLO WORLD"
        assert data["uppercase"] is True
        assert data["word_count"] == 2


@pytest.mark.asyncio
async def test_http_openapi_docs(http_server):
    """Test that OpenAPI documentation is available."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{http_server}/docs")
        assert response.status_code == 200
        
        # Test OpenAPI JSON schema
        response = await client.get(f"{http_server}/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert openapi_spec["info"]["title"] == "E2E HTTP Test Server"
        assert openapi_spec["info"]["version"] == "1.0.0"
        
        # Verify our endpoints are documented
        paths = openapi_spec.get("paths", {})
        assert "/health" in paths
        assert "/users" in paths
        assert "/calculate" in paths


@pytest.mark.asyncio
async def test_http_concurrent_requests(http_server):
    """Test handling concurrent HTTP requests."""
    async with httpx.AsyncClient() as client:
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            calc_data = {
                "x": float(i),
                "y": 2.0,
                "operation": "add"
            }
            task = client.post(f"{http_server}/calculate", json=calc_data)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert data["result"] == float(i) + 2.0


if __name__ == "__main__":
    # Run a quick smoke test
    async def smoke_test():
        print("Running HTTP E2E smoke test...")
        
        # Start server
        port = 8002
        process = multiprocessing.Process(target=run_http_server, args=(port,))
        process.start()
        
        try:
            # Wait for server to start
            time.sleep(2)
            base_url = f"http://127.0.0.1:{port}"
            
            async with httpx.AsyncClient() as client:
                # Test health check
                response = await client.get(f"{base_url}/health")
                assert response.status_code == 200
                print("✓ Health check passed")
                
                # Test calculation endpoint
                calc_data = {"x": 5.0, "y": 3.0, "operation": "add"}
                response = await client.post(f"{base_url}/calculate", json=calc_data)
                assert response.status_code == 200
                data = response.json()
                assert data["result"] == 8.0
                print("✓ Calculate endpoint passed")
                
                print("HTTP E2E smoke test passed!")
        
        finally:
            process.terminate()
            process.join(timeout=5)
    
    asyncio.run(smoke_test())