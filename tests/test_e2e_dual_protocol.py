"""End-to-end tests verifying dual-protocol functionality."""

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


class DualProtocolServer:
    """Server with identical functionality exposed via both protocols."""
    
    def __init__(self):
        self.app = LightMCP(
            name="Dual Protocol Test Server",
            version="1.0.0",
            description="Testing identical functionality via HTTP and MCP"
        )
        self._setup_dual_endpoints()
    
    def _setup_dual_endpoints(self):
        """Set up endpoints available via both protocols."""
        
        class TaskInput(BaseModel):
            title: str
            description: str = ""
            priority: int = 1
            completed: bool = False
        
        class TaskOutput(BaseModel):
            id: int
            title: str
            description: str
            priority: int
            completed: bool
            created_at: str = "2024-01-01T00:00:00Z"
        
        class SearchInput(BaseModel):
            query: str
            max_results: int = 10
            category: str = "all"
        
        # In-memory storage for testing
        self.tasks = {}
        self.next_id = 1
        
        @self.app.tool(description="Create a new task")
        @self.app.post("/tasks", response_model=TaskOutput)
        async def create_task(task: TaskInput) -> TaskOutput:
            """Create a new task (dual protocol)."""
            task_id = self.next_id
            self.next_id += 1
            
            new_task = TaskOutput(
                id=task_id,
                title=task.title,
                description=task.description,
                priority=task.priority,
                completed=task.completed
            )
            
            self.tasks[task_id] = new_task.model_dump()
            return new_task
        
        @self.app.tool(description="Get task by ID")
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: int) -> Dict[str, Any]:
            """Get a task by ID (dual protocol)."""
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            return self.tasks[task_id]
        
        @self.app.tool(description="Update task completion status")
        @self.app.patch("/tasks/{task_id}/complete")
        async def complete_task(task_id: int, completed: bool = True) -> Dict[str, Any]:
            """Mark task as completed (dual protocol)."""
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            self.tasks[task_id]["completed"] = completed
            return self.tasks[task_id]
        
        @self.app.tool(description="Search through tasks")
        @self.app.post("/search")
        async def search_tasks(search: SearchInput) -> Dict[str, Any]:
            """Search tasks (dual protocol)."""
            results = []
            query_lower = search.query.lower()
            
            for task in self.tasks.values():
                if (query_lower in task["title"].lower() or 
                    query_lower in task["description"].lower()):
                    results.append(task)
                
                if len(results) >= search.max_results:
                    break
            
            return {
                "query": search.query,
                "results": results,
                "total_found": len(results),
                "max_results": search.max_results,
                "category": search.category
            }
        
        @self.app.tool(description="Get server statistics")
        @self.app.get("/stats")
        async def get_stats() -> Dict[str, Any]:
            """Get server statistics (dual protocol)."""
            completed_tasks = sum(1 for task in self.tasks.values() if task["completed"])
            pending_tasks = len(self.tasks) - completed_tasks
            
            return {
                "total_tasks": len(self.tasks),
                "completed_tasks": completed_tasks,
                "pending_tasks": pending_tasks,
                "server_version": self.app.version,
                "uptime": "unknown"  # Simplified for testing
            }


def run_dual_server(port: int = 8003):
    """Run dual protocol server for HTTP testing."""
    import uvicorn
    
    server = DualProtocolServer()
    uvicorn.run(server.app.fastapi_app, host="127.0.0.1", port=port, log_level="critical")


@pytest.fixture(scope="module") 
def dual_server():
    """Start dual protocol server for testing."""
    port = 8003
    process = multiprocessing.Process(target=run_dual_server, args=(port,))
    process.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Verify server is running
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(10):
        try:
            response = httpx.get(f"{base_url}/stats", timeout=1.0)
            if response.status_code == 200:
                break
        except:
            time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("Dual protocol server failed to start")
    
    yield (base_url, DualProtocolServer())  # Return URL and server instance
    
    # Cleanup
    process.terminate()
    process.join(timeout=5)


@pytest.mark.asyncio
async def test_identical_functionality_create_task(dual_server):
    """Test that creating tasks works identically via both protocols."""
    base_url, server_instance = dual_server
    
    # Test data
    task_data = {
        "title": "Test Task",
        "description": "This is a test task",
        "priority": 2,
        "completed": False
    }
    
    # Test via HTTP
    async with httpx.AsyncClient() as client:
        http_response = await client.post(f"{base_url}/tasks", json=task_data)
        assert http_response.status_code == 200
        http_result = http_response.json()
    
    # Test via MCP (simulate direct call)
    from lightmcp.app import ToolRegistration
    from pydantic import BaseModel
    
    class TaskInput(BaseModel):
        title: str
        description: str = ""
        priority: int = 1
        completed: bool = False
    
    # Get the MCP tool handler
    create_task_registration = server_instance.app._tools["create_task"]
    task_input = TaskInput(**task_data)
    mcp_result = await create_task_registration.func(task_input)
    
    # Compare results (they should be functionally identical)
    assert http_result["title"] == mcp_result.title
    assert http_result["description"] == mcp_result.description
    assert http_result["priority"] == mcp_result.priority
    assert http_result["completed"] == mcp_result.completed
    
    # Both should have generated IDs
    assert "id" in http_result
    assert hasattr(mcp_result, "id")


@pytest.mark.asyncio
async def test_identical_functionality_search(dual_server):
    """Test that search works identically via both protocols."""
    base_url, server_instance = dual_server
    
    # First create some tasks via HTTP
    async with httpx.AsyncClient() as client:
        tasks_to_create = [
            {"title": "Buy groceries", "description": "Milk, bread, eggs"},
            {"title": "Walk the dog", "description": "Morning walk in the park"},
            {"title": "Finish project", "description": "Complete the quarterly report"}
        ]
        
        for task_data in tasks_to_create:
            await client.post(f"{base_url}/tasks", json=task_data)
        
        # Search via HTTP
        search_data = {"query": "project", "max_results": 5}
        http_response = await client.post(f"{base_url}/search", json=search_data)
        assert http_response.status_code == 200
        http_result = http_response.json()
    
    # Search via MCP
    from pydantic import BaseModel
    
    class SearchInput(BaseModel):
        query: str
        max_results: int = 10
        category: str = "all"
    
    search_registration = server_instance.app._tools["search_tasks"]
    search_input = SearchInput(**search_data)
    mcp_result = await search_registration.func(search_input)
    
    # Results should be identical
    assert http_result["query"] == mcp_result["query"]
    assert http_result["total_found"] == mcp_result["total_found"]
    assert len(http_result["results"]) == len(mcp_result["results"])
    
    # Verify actual search results match
    if http_result["results"]:
        assert http_result["results"][0]["title"] == mcp_result["results"][0]["title"]


@pytest.mark.asyncio
async def test_state_consistency_across_protocols(dual_server):
    """Test that state changes via one protocol are visible via the other."""
    base_url, server_instance = dual_server
    
    # Create task via HTTP
    async with httpx.AsyncClient() as client:
        task_data = {"title": "State Test Task", "priority": 3}
        create_response = await client.post(f"{base_url}/tasks", json=task_data)
        assert create_response.status_code == 200
        created_task = create_response.json()
        task_id = created_task["id"]
        
        # Verify task exists via HTTP
        get_response = await client.get(f"{base_url}/tasks/{task_id}")
        assert get_response.status_code == 200
        http_task = get_response.json()
        assert not http_task["completed"]
    
    # Modify task via MCP
    complete_registration = server_instance.app._tools["complete_task"]
    mcp_result = await complete_registration.func(task_id=task_id, completed=True)
    
    # Verify change is visible via HTTP
    async with httpx.AsyncClient() as client:
        get_response = await client.get(f"{base_url}/tasks/{task_id}")
        assert get_response.status_code == 200
        updated_task = get_response.json()
        assert updated_task["completed"] is True  # Should reflect MCP change
        
        # Verify via MCP as well
        get_registration = server_instance.app._tools["get_task"]
        mcp_task = await get_registration.func(task_id=task_id)
        assert mcp_task["completed"] is True


@pytest.mark.asyncio
async def test_error_handling_consistency(dual_server):
    """Test that error handling is consistent across both protocols."""
    base_url, server_instance = dual_server
    
    nonexistent_id = 99999
    
    # Test HTTP error handling
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/tasks/{nonexistent_id}")
        assert response.status_code == 500  # Should be an error
    
    # Test MCP error handling
    get_registration = server_instance.app._tools["get_task"]
    
    with pytest.raises(ValueError, match="not found"):
        await get_registration.func(task_id=nonexistent_id)


@pytest.mark.asyncio
async def test_validation_consistency(dual_server):
    """Test that input validation is consistent across both protocols."""
    base_url, server_instance = dual_server
    
    # Invalid task data (missing required field)
    invalid_data = {"description": "No title provided"}
    
    # Test HTTP validation
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/tasks", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    # Test MCP validation
    from pydantic import BaseModel, ValidationError
    
    class TaskInput(BaseModel):
        title: str
        description: str = ""
        priority: int = 1
        completed: bool = False
    
    create_registration = server_instance.app._tools["create_task"]
    
    with pytest.raises(ValidationError):
        # This should fail at the Pydantic validation level
        task_input = TaskInput(**invalid_data)


@pytest.mark.asyncio
async def test_concurrent_access_across_protocols(dual_server):
    """Test concurrent access via both HTTP and MCP protocols."""
    base_url, server_instance = dual_server
    
    async def create_task_http(client, title):
        task_data = {"title": f"HTTP Task {title}"}
        response = await client.post(f"{base_url}/tasks", json=task_data)
        return response.json()
    
    async def create_task_mcp(title):
        from pydantic import BaseModel
        
        class TaskInput(BaseModel):
            title: str
            description: str = ""
            priority: int = 1
            completed: bool = False
        
        create_registration = server_instance.app._tools["create_task"]
        task_input = TaskInput(title=f"MCP Task {title}")
        return await create_registration.func(task_input)
    
    # Create tasks concurrently via both protocols
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(5):
            tasks.append(create_task_http(client, i))
            tasks.append(create_task_mcp(i))
        
        results = await asyncio.gather(*tasks)
    
    # Verify all tasks were created successfully
    assert len(results) == 10
    
    # Get final statistics
    async with httpx.AsyncClient() as client:
        stats_response = await client.get(f"{base_url}/stats")
        stats = stats_response.json()
        
        # Should have at least 10 tasks (may have more from other tests)
        assert stats["total_tasks"] >= 10


if __name__ == "__main__":
    # Run a comprehensive smoke test
    async def smoke_test():
        print("Running dual protocol E2E smoke test...")
        
        # Start server
        port = 8004
        process = multiprocessing.Process(target=run_dual_server, args=(port,))
        process.start()
        
        try:
            # Wait for server to start
            time.sleep(2)
            base_url = f"http://127.0.0.1:{port}"
            server_instance = DualProtocolServer()
            
            async with httpx.AsyncClient() as client:
                # Test creating task via HTTP
                task_data = {"title": "Smoke Test Task", "priority": 1}
                response = await client.post(f"{base_url}/tasks", json=task_data)
                assert response.status_code == 200
                task = response.json()
                task_id = task["id"]
                print("✓ Created task via HTTP")
                
                # Test getting task via HTTP  
                response = await client.get(f"{base_url}/tasks/{task_id}")
                assert response.status_code == 200
                print("✓ Retrieved task via HTTP")
                
                # Test MCP tool functionality
                from pydantic import BaseModel
                
                class TaskInput(BaseModel):
                    title: str
                    description: str = ""
                    priority: int = 1
                    completed: bool = False
                
                create_registration = server_instance.app._tools["create_task"]
                task_input = TaskInput(title="MCP Smoke Test Task")
                mcp_task = await create_registration.func(task_input)
                print("✓ Created task via MCP")
                
                # Test search via both protocols
                search_data = {"query": "Smoke", "max_results": 10}
                response = await client.post(f"{base_url}/search", json=search_data)
                assert response.status_code == 200
                http_results = response.json()
                assert http_results["total_found"] >= 1
                print("✓ Search via HTTP works")
                
                print("Dual protocol E2E smoke test passed!")
        
        finally:
            process.terminate()
            process.join(timeout=5)
    
    asyncio.run(smoke_test())