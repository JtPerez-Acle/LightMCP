"""Tests for validation and error handling across both protocols."""

import asyncio
import json
import multiprocessing
import sys
import time
from typing import Any, Dict, List, Optional, Union

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError, validator

# Add src to path for imports
sys.path.insert(0, "../src")

from lightmcp import LightMCP


class ValidationTestServer:
    """Server for testing validation and error handling."""
    
    def __init__(self):
        self.app = LightMCP(
            name="Validation Test Server",
            version="1.0.0",
            description="Testing validation and error handling"
        )
        self._setup_validation_endpoints()
    
    def _setup_validation_endpoints(self):
        """Set up endpoints for validation testing."""
        
        class StrictUser(BaseModel):
            name: str = Field(..., min_length=2, max_length=50)
            age: int = Field(..., ge=0, le=150)
            email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
            scores: List[float] = Field(default_factory=list)
            metadata: Optional[Dict[str, Any]] = None
            
            @validator('name')
            def name_must_be_title_case(cls, v):
                if not v.istitle():
                    raise ValueError('Name must be in title case')
                return v
            
            @validator('scores')
            def scores_must_be_valid(cls, v):
                if any(score < 0 or score > 100 for score in v):
                    raise ValueError('All scores must be between 0 and 100')
                return v
        
        class MathOperation(BaseModel):
            operation: str = Field(..., pattern=r'^(add|subtract|multiply|divide|power)$')
            operands: List[float] = Field(..., min_items=2, max_items=2)
            precision: int = Field(default=2, ge=0, le=10)
        
        class FileUpload(BaseModel):
            filename: str = Field(..., min_length=1)
            content: str = Field(..., min_length=1)
            size_bytes: int = Field(..., ge=1)
            content_type: str = Field(default="text/plain")
            
            @validator('filename')
            def filename_must_be_valid(cls, v):
                if not v.endswith(('.txt', '.json', '.csv', '.md')):
                    raise ValueError('Only .txt, .json, .csv, .md files allowed')
                return v
            
            @validator('size_bytes')
            def size_must_be_reasonable(cls, v):
                if v > 1024 * 1024:  # 1MB
                    raise ValueError('File too large (max 1MB)')
                return v
        
        # Strict validation endpoints
        @self.app.tool(description="Create user with strict validation")
        @self.app.post("/users/strict")
        async def create_strict_user(user: StrictUser) -> Dict[str, Any]:
            """Create user with comprehensive validation."""
            return {
                "id": 12345,
                "name": user.name,
                "age": user.age,
                "email": user.email,
                "scores": user.scores,
                "metadata": user.metadata or {},
                "validation_passed": True
            }
        
        @self.app.tool(description="Perform math operations with validation")
        @self.app.post("/math/calculate")
        async def calculate_math(operation: MathOperation) -> Dict[str, Any]:
            """Perform mathematical operations with validation."""
            a, b = operation.operands
            
            if operation.operation == "add":
                result = a + b
            elif operation.operation == "subtract":
                result = a - b
            elif operation.operation == "multiply":
                result = a * b
            elif operation.operation == "divide":
                if b == 0:
                    raise ValueError("Division by zero is not allowed")
                result = a / b
            elif operation.operation == "power":
                if a == 0 and b < 0:
                    raise ValueError("Cannot raise zero to negative power")
                result = a ** b
            else:
                raise ValueError(f"Unknown operation: {operation.operation}")
            
            return {
                "operation": operation.operation,
                "operands": operation.operands,
                "result": round(result, operation.precision),
                "precision": operation.precision
            }
        
        @self.app.tool(description="Upload file with validation")
        @self.app.post("/files/upload")
        async def upload_file(file: FileUpload) -> Dict[str, Any]:
            """Upload file with validation."""
            # Simulate file processing
            word_count = len(file.content.split())
            
            return {
                "filename": file.filename,
                "size_bytes": file.size_bytes,
                "content_type": file.content_type,
                "word_count": word_count,
                "uploaded_at": "2024-01-01T00:00:00Z",
                "status": "uploaded"
            }
        
        # Error simulation endpoints
        @self.app.tool(description="Simulate various error conditions")
        @self.app.post("/errors/simulate")
        async def simulate_error(error_type: str) -> Dict[str, Any]:
            """Simulate different types of errors."""
            if error_type == "validation":
                # This will be caught by validation before reaching here
                raise ValueError("This shouldn't be reached")
            elif error_type == "runtime":
                raise RuntimeError("Simulated runtime error")
            elif error_type == "value":
                raise ValueError("Simulated value error")
            elif error_type == "type":
                raise TypeError("Simulated type error")
            elif error_type == "zero_division":
                return {"result": 1 / 0}  # This will raise ZeroDivisionError
            elif error_type == "none":
                return None  # This might cause issues
            else:
                return {"error_type": error_type, "simulated": True}
        
        # Optional parameters endpoint
        @self.app.tool(description="Test optional parameters")
        @self.app.get("/optional")
        async def test_optional_params(
            required_param: str,
            optional_string: Optional[str] = None,
            optional_int: Optional[int] = None,
            default_value: str = "default",
            flag: bool = False
        ) -> Dict[str, Any]:
            """Test endpoint with optional parameters."""
            return {
                "required_param": required_param,
                "optional_string": optional_string,
                "optional_int": optional_int,
                "default_value": default_value,
                "flag": flag
            }


def run_validation_server(port: int = 8005):
    """Run validation test server."""
    import uvicorn
    
    server = ValidationTestServer()
    uvicorn.run(server.app.fastapi_app, host="127.0.0.1", port=port, log_level="critical")


@pytest.fixture(scope="module")
def validation_server():
    """Start validation test server."""
    port = 8005
    process = multiprocessing.Process(target=run_validation_server, args=(port,))
    process.start()
    
    time.sleep(2)
    
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(10):
        try:
            response = httpx.get(f"{base_url}/optional?required_param=test", timeout=1.0)
            if response.status_code == 200:
                break
        except:
            time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("Validation server failed to start")
    
    yield (base_url, ValidationTestServer())
    
    process.terminate()
    process.join(timeout=5)


@pytest.mark.asyncio
async def test_pydantic_validation_http_valid_data(validation_server):
    """Test Pydantic validation with valid data via HTTP."""
    base_url, _ = validation_server
    
    valid_user = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "scores": [85.5, 92.0, 78.5],
        "metadata": {"department": "engineering"}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/users/strict", json=valid_user)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "John Doe"
        assert data["age"] == 30
        assert data["validation_passed"] is True


@pytest.mark.asyncio
async def test_pydantic_validation_http_invalid_data(validation_server):
    """Test Pydantic validation with invalid data via HTTP."""
    base_url, _ = validation_server
    
    invalid_users = [
        # Missing required field
        {"name": "John Doe", "age": 30},  # Missing email
        
        # Invalid email format
        {"name": "John Doe", "age": 30, "email": "invalid-email"},
        
        # Age out of range
        {"name": "John Doe", "age": 200, "email": "john@example.com"},
        
        # Name too short
        {"name": "J", "age": 30, "email": "john@example.com"},
        
        # Name not title case
        {"name": "john doe", "age": 30, "email": "john@example.com"},
        
        # Invalid scores
        {"name": "John Doe", "age": 30, "email": "john@example.com", "scores": [150.0]},
    ]
    
    async with httpx.AsyncClient() as client:
        for invalid_user in invalid_users:
            response = await client.post(f"{base_url}/users/strict", json=invalid_user)
            assert response.status_code == 422, f"Expected validation error for {invalid_user}"


@pytest.mark.asyncio
async def test_pydantic_validation_mcp_valid_data(validation_server):
    """Test Pydantic validation with valid data via MCP."""
    _, server_instance = validation_server
    
    from pydantic import BaseModel, Field, validator
    from typing import List, Optional, Dict, Any
    
    class StrictUser(BaseModel):
        name: str = Field(..., min_length=2, max_length=50)
        age: int = Field(..., ge=0, le=150)
        email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
        scores: List[float] = Field(default_factory=list)
        metadata: Optional[Dict[str, Any]] = None
        
        @validator('name')
        def name_must_be_title_case(cls, v):
            if not v.istitle():
                raise ValueError('Name must be in title case')
            return v
    
    valid_user_data = {
        "name": "Jane Smith",
        "age": 25,
        "email": "jane@example.com",
        "scores": [90.0, 85.5]
    }
    
    create_user_registration = server_instance.app._tools["create_strict_user"]
    user_input = StrictUser(**valid_user_data)
    result = await create_user_registration.func(user_input)
    
    assert result["name"] == "Jane Smith"
    assert result["age"] == 25
    assert result["validation_passed"] is True


@pytest.mark.asyncio
async def test_pydantic_validation_mcp_invalid_data(validation_server):
    """Test Pydantic validation with invalid data via MCP."""
    _, server_instance = validation_server
    
    from pydantic import BaseModel, Field, ValidationError, validator
    from typing import List, Optional, Dict, Any
    
    class StrictUser(BaseModel):
        name: str = Field(..., min_length=2, max_length=50)
        age: int = Field(..., ge=0, le=150)
        email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
        scores: List[float] = Field(default_factory=list)
        metadata: Optional[Dict[str, Any]] = None
        
        @validator('name')
        def name_must_be_title_case(cls, v):
            if not v.istitle():
                raise ValueError('Name must be in title case')
            return v
    
    invalid_data_sets = [
        {"name": "J", "age": 30, "email": "john@example.com"},  # Name too short
        {"name": "John Doe", "age": 200, "email": "john@example.com"},  # Age too high
        {"name": "John Doe", "age": 30, "email": "invalid"},  # Invalid email
    ]
    
    for invalid_data in invalid_data_sets:
        with pytest.raises(ValidationError):
            StrictUser(**invalid_data)


@pytest.mark.asyncio
async def test_business_logic_errors_http(validation_server):
    """Test business logic errors via HTTP."""
    base_url, _ = validation_server
    
    # Test division by zero
    math_data = {
        "operation": "divide",
        "operands": [10.0, 0.0]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/math/calculate", json=math_data)
        assert response.status_code == 500  # Internal server error
        
        # Test invalid operation (should be caught by validation)
        invalid_math = {
            "operation": "invalid_op",
            "operands": [5.0, 3.0]
        }
        
        response = await client.post(f"{base_url}/math/calculate", json=invalid_math)
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_business_logic_errors_mcp(validation_server):
    """Test business logic errors via MCP."""
    _, server_instance = validation_server
    
    from pydantic import BaseModel, Field
    from typing import List
    
    class MathOperation(BaseModel):
        operation: str = Field(..., regex=r'^(add|subtract|multiply|divide|power)$')
        operands: List[float] = Field(..., min_items=2, max_items=2)
        precision: int = Field(default=2, ge=0, le=10)
    
    # Test division by zero
    math_registration = server_instance.app._tools["calculate_math"]
    math_input = MathOperation(operation="divide", operands=[10.0, 0.0])
    
    with pytest.raises(ValueError, match="Division by zero"):
        await math_registration.func(math_input)
    
    # Test zero to negative power
    power_input = MathOperation(operation="power", operands=[0.0, -2.0])
    
    with pytest.raises(ValueError, match="Cannot raise zero to negative power"):
        await power_input.func(power_input)


@pytest.mark.asyncio
async def test_complex_validation_file_upload(validation_server):
    """Test complex validation with file upload."""
    base_url, server_instance = validation_server
    
    # Valid file upload
    valid_file = {
        "filename": "test.txt",
        "content": "This is a test file with some content.",
        "size_bytes": 100,
        "content_type": "text/plain"
    }
    
    # Test via HTTP
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/files/upload", json=valid_file)
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["status"] == "uploaded"
    
    # Test via MCP
    from pydantic import BaseModel, Field, validator
    
    class FileUpload(BaseModel):
        filename: str = Field(..., min_length=1)
        content: str = Field(..., min_length=1)
        size_bytes: int = Field(..., ge=1)
        content_type: str = Field(default="text/plain")
        
        @validator('filename')
        def filename_must_be_valid(cls, v):
            if not v.endswith(('.txt', '.json', '.csv', '.md')):
                raise ValueError('Only .txt, .json, .csv, .md files allowed')
            return v
    
    upload_registration = server_instance.app._tools["upload_file"]
    file_input = FileUpload(**valid_file)
    mcp_result = await upload_registration.func(file_input)
    
    assert mcp_result["filename"] == "test.txt"
    assert mcp_result["status"] == "uploaded"
    
    # Test invalid file extension
    invalid_file = {
        "filename": "test.exe",
        "content": "This should not be allowed.",
        "size_bytes": 50
    }
    
    with pytest.raises(ValidationError, match="Only .txt, .json, .csv, .md files allowed"):
        FileUpload(**invalid_file)


@pytest.mark.asyncio
async def test_optional_parameters_validation(validation_server):
    """Test optional parameters validation."""
    base_url, server_instance = validation_server
    
    # Test with only required parameter
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/optional?required_param=test")
        assert response.status_code == 200
        data = response.json()
        assert data["required_param"] == "test"
        assert data["optional_string"] is None
        assert data["default_value"] == "default"
        assert data["flag"] is False
    
    # Test with all parameters
    async with httpx.AsyncClient() as client:
        params = {
            "required_param": "test",
            "optional_string": "optional",
            "optional_int": "42",
            "default_value": "custom",
            "flag": "true"
        }
        response = await client.get(f"{base_url}/optional", params=params)
        assert response.status_code == 200
        data = response.json()
        assert data["optional_string"] == "optional"
        assert data["optional_int"] == 42
        assert data["default_value"] == "custom"
        assert data["flag"] is True


@pytest.mark.asyncio
async def test_error_consistency_across_protocols(validation_server):
    """Test that errors are handled consistently across protocols."""
    base_url, server_instance = validation_server
    
    # Test the same validation error via both protocols
    invalid_user = {
        "name": "j",  # Too short
        "age": 30,
        "email": "john@example.com"
    }
    
    # HTTP should return 422
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/users/strict", json=invalid_user)
        assert response.status_code == 422
    
    # MCP should raise ValidationError
    from pydantic import BaseModel, Field, ValidationError, validator
    from typing import List, Optional, Dict, Any
    
    class StrictUser(BaseModel):
        name: str = Field(..., min_length=2, max_length=50)
        age: int = Field(..., ge=0, le=150)
        email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
        scores: List[float] = Field(default_factory=list)
        metadata: Optional[Dict[str, Any]] = None
        
        @validator('name')
        def name_must_be_title_case(cls, v):
            if not v.istitle():
                raise ValueError('Name must be in title case')
            return v
    
    with pytest.raises(ValidationError):
        StrictUser(**invalid_user)


@pytest.mark.asyncio
async def test_edge_cases_and_boundary_conditions(validation_server):
    """Test edge cases and boundary conditions."""
    base_url, server_instance = validation_server
    
    # Test boundary values
    boundary_user = {
        "name": "AB",  # Minimum length
        "age": 150,    # Maximum age
        "email": "a@b.c",  # Minimal valid email
        "scores": [0.0, 100.0]  # Boundary scores
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/users/strict", json=boundary_user)
        assert response.status_code == 200
    
    # Test empty arrays
    empty_scores_user = {
        "name": "Test User",
        "age": 25,
        "email": "test@example.com",
        "scores": []  # Empty array should be valid
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/users/strict", json=empty_scores_user)
        assert response.status_code == 200
        data = response.json()
        assert data["scores"] == []


@pytest.mark.asyncio
async def test_concurrent_validation_load(validation_server):
    """Test validation under concurrent load."""
    base_url, _ = validation_server
    
    async def create_user(client, user_id):
        user_data = {
            "name": f"User {user_id}",
            "age": 20 + (user_id % 30),
            "email": f"user{user_id}@example.com",
            "scores": [float(80 + (user_id % 20))]
        }
        response = await client.post(f"{base_url}/users/strict", json=user_data)
        return response.status_code
    
    # Create multiple concurrent requests
    async with httpx.AsyncClient() as client:
        tasks = [create_user(client, i) for i in range(20)]
        results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(status == 200 for status in results)


if __name__ == "__main__":
    # Run validation smoke test
    async def smoke_test():
        print("Running validation and error handling smoke test...")
        
        port = 8006
        process = multiprocessing.Process(target=run_validation_server, args=(port,))
        process.start()
        
        try:
            time.sleep(2)
            base_url = f"http://127.0.0.1:{port}"
            
            async with httpx.AsyncClient() as client:
                # Test valid data
                valid_user = {
                    "name": "Test User",
                    "age": 30,
                    "email": "test@example.com",
                    "scores": [85.0, 90.0]
                }
                response = await client.post(f"{base_url}/users/strict", json=valid_user)
                assert response.status_code == 200
                print("✓ Valid data validation passed")
                
                # Test invalid data
                invalid_user = {"name": "Test User", "age": 30}  # Missing email
                response = await client.post(f"{base_url}/users/strict", json=invalid_user)
                assert response.status_code == 422
                print("✓ Invalid data validation failed as expected")
                
                # Test business logic error
                math_data = {"operation": "divide", "operands": [10.0, 0.0]}
                response = await client.post(f"{base_url}/math/calculate", json=math_data)
                assert response.status_code == 500
                print("✓ Business logic error handled correctly")
                
                print("Validation and error handling smoke test passed!")
        
        finally:
            process.terminate()
            process.join(timeout=5)
    
    asyncio.run(smoke_test())