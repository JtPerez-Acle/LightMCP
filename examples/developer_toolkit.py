"""
Real-world example: Developer Toolkit Service

This demonstrates LightMCP's practical value by creating a service that helps developers
manage projects, analyze code, and perform common development tasks. The same functionality
is available both as a web API for dashboards/UIs and as MCP tools for AI coding assistants.

Use cases:
- Web developers can use the HTTP API in their project dashboards
- AI coding assistants can use MCP tools to help with project management
- DevOps tools can integrate via either protocol as needed
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

# Add parent directory to path for development
sys.path.insert(0, "../src")

from lightmcp import LightMCP

# Initialize the service
app = LightMCP(
    name="Developer Toolkit Service",
    version="1.0.0",
    description="Dual-protocol developer tools for project management and code analysis",
)


# ================================
# PYDANTIC MODELS FOR TYPE SAFETY
# ================================

class ProjectInfo(BaseModel):
    """Project information model."""
    name: str
    path: str
    language: str
    framework: Optional[str] = None
    last_modified: str
    size_mb: float
    file_count: int


class CodeAnalysisRequest(BaseModel):
    """Request for code analysis."""
    project_path: str = Field(..., description="Path to the project directory")
    file_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".java", ".go", ".rs"],
        description="File extensions to analyze"
    )
    include_metrics: bool = Field(default=True, description="Include complexity metrics")


class CodeAnalysisResult(BaseModel):
    """Result of code analysis."""
    project_path: str
    total_files: int
    total_lines: int
    language_breakdown: Dict[str, int]
    largest_files: List[Dict[str, Any]]
    complexity_metrics: Optional[Dict[str, Any]] = None
    analysis_date: str


class GitStatusRequest(BaseModel):
    """Request for Git status."""
    project_path: str = Field(..., description="Path to the Git repository")
    include_diff: bool = Field(default=False, description="Include diff information")


class TestRunRequest(BaseModel):
    """Request to run tests."""
    project_path: str = Field(..., description="Path to the project")
    test_command: str = Field(default="pytest", description="Test command to run")
    verbose: bool = Field(default=False, description="Verbose output")


class DependencyCheckRequest(BaseModel):
    """Request to check dependencies."""
    project_path: str = Field(..., description="Path to the project")
    package_file: str = Field(
        default="requirements.txt",
        description="Package file to check (requirements.txt, package.json, etc.)"
    )


# ================================
# DUAL-PROTOCOL ENDPOINTS
# ================================

@app.tool(description="Analyze project directory structure and metrics")
@app.post("/projects/analyze", response_model=CodeAnalysisResult)
async def analyze_project(request: CodeAnalysisRequest) -> CodeAnalysisResult:
    """
    Analyze a code project to provide insights on structure, complexity, and metrics.
    
    Available as:
    - HTTP API: POST /projects/analyze (for web dashboards)
    - MCP Tool: analyze_project (for AI coding assistants)
    """
    project_path = Path(request.project_path)
    
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {request.project_path}")
    
    # Analyze files
    language_breakdown = {}
    total_files = 0
    total_lines = 0
    largest_files = []
    
    for ext in request.file_extensions:
        files = list(project_path.rglob(f"*{ext}"))
        total_files += len(files)
        
        language_name = _get_language_name(ext)
        file_lines = 0
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    file_lines += lines
                    
                    # Track largest files
                    largest_files.append({
                        "file": str(file_path.relative_to(project_path)),
                        "lines": lines,
                        "language": language_name
                    })
            except Exception:
                continue
        
        if file_lines > 0:
            language_breakdown[language_name] = file_lines
            total_lines += file_lines
    
    # Sort largest files
    largest_files.sort(key=lambda x: x["lines"], reverse=True)
    largest_files = largest_files[:10]  # Top 10
    
    # Basic complexity metrics
    complexity_metrics = None
    if request.include_metrics:
        complexity_metrics = {
            "avg_lines_per_file": round(total_lines / max(total_files, 1), 2),
            "language_diversity": len(language_breakdown),
            "largest_file_lines": largest_files[0]["lines"] if largest_files else 0,
            "project_size_category": _categorize_project_size(total_lines)
        }
    
    return CodeAnalysisResult(
        project_path=request.project_path,
        total_files=total_files,
        total_lines=total_lines,
        language_breakdown=language_breakdown,
        largest_files=largest_files,
        complexity_metrics=complexity_metrics,
        analysis_date=datetime.now().isoformat()
    )


@app.tool(description="Get Git repository status and changes")
@app.post("/git/status")
async def get_git_status(request: GitStatusRequest) -> Dict[str, Any]:
    """
    Get Git repository status including branch, changes, and optionally diff.
    
    Available as:
    - HTTP API: POST /git/status (for web Git interfaces)  
    - MCP Tool: get_git_status (for AI assistants managing code)
    """
    project_path = Path(request.project_path)
    
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {request.project_path}")
    
    if not (project_path / ".git").exists():
        raise ValueError(f"Not a Git repository: {request.project_path}")
    
    try:
        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = branch_result.stdout.strip()
        
        # Get status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse status
        changes = []
        for line in status_result.stdout.strip().split('\n'):
            if line:
                status_code = line[:2]
                file_path = line[3:]
                changes.append({
                    "file": file_path,
                    "status": _parse_git_status(status_code)
                })
        
        result = {
            "project_path": request.project_path,
            "current_branch": current_branch,
            "has_changes": len(changes) > 0,
            "changes": changes,
            "change_summary": {
                "modified": len([c for c in changes if "modified" in c["status"]]),
                "added": len([c for c in changes if "added" in c["status"]]),
                "deleted": len([c for c in changes if "deleted" in c["status"]]),
                "untracked": len([c for c in changes if "untracked" in c["status"]])
            }
        }
        
        # Add diff if requested
        if request.include_diff and changes:
            diff_result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            result["diff_files"] = diff_result.stdout.strip().split('\n') if diff_result.stdout.strip() else []
        
        return result
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Git command failed: {e}")


@app.tool(description="Run project tests and return results")
@app.post("/tests/run")
async def run_tests(request: TestRunRequest) -> Dict[str, Any]:
    """
    Run project tests and return results summary.
    
    Available as:
    - HTTP API: POST /tests/run (for CI/CD dashboards)
    - MCP Tool: run_tests (for AI assistants checking code quality)
    """
    project_path = Path(request.project_path)
    
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {request.project_path}")
    
    # Determine test command based on project type
    test_cmd = request.test_command
    if test_cmd == "auto":
        if (project_path / "pytest.ini").exists() or (project_path / "pyproject.toml").exists():
            test_cmd = "pytest"
        elif (project_path / "package.json").exists():
            test_cmd = "npm test"
        elif (project_path / "Cargo.toml").exists():
            test_cmd = "cargo test"
        else:
            test_cmd = "pytest"  # Default
    
    try:
        # Run tests
        cmd_args = test_cmd.split()
        if request.verbose and "pytest" in test_cmd:
            cmd_args.append("-v")
        
        start_time = datetime.now()
        result = subprocess.run(
            cmd_args,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = datetime.now()
        
        # Parse output (simplified for demo)
        output_lines = result.stdout.split('\n')
        error_lines = result.stderr.split('\n')
        
        # Basic parsing for pytest output
        passed = failed = skipped = 0
        for line in output_lines:
            if "passed" in line and "failed" in line:
                # Try to extract numbers
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        try:
                            passed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == "failed" and i > 0:
                        try:
                            failed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == "skipped" and i > 0:
                        try:
                            skipped = int(parts[i-1])
                        except ValueError:
                            pass
        
        return {
            "project_path": request.project_path,
            "test_command": test_cmd,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "summary": {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total": passed + failed + skipped
            },
            "output": result.stdout[:2000],  # Truncate for demo
            "errors": result.stderr[:1000] if result.stderr else None,
            "timestamp": start_time.isoformat()
        }
        
    except subprocess.TimeoutExpired:
        raise ValueError("Test execution timed out after 5 minutes")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Test command failed: {e}")


@app.tool(description="Check project dependencies for updates and security")
@app.post("/dependencies/check")
async def check_dependencies(request: DependencyCheckRequest) -> Dict[str, Any]:
    """
    Check project dependencies for updates and potential security issues.
    
    Available as:
    - HTTP API: POST /dependencies/check (for security dashboards)
    - MCP Tool: check_dependencies (for AI assistants managing dependencies)
    """
    project_path = Path(request.project_path)
    package_file_path = project_path / request.package_file
    
    if not package_file_path.exists():
        raise ValueError(f"Package file does not exist: {package_file_path}")
    
    dependencies = []
    
    try:
        if request.package_file == "requirements.txt":
            # Parse Python requirements
            with open(package_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Simple parsing for demo
                        if '>=' in line:
                            name, version = line.split('>=')
                        elif '==' in line:
                            name, version = line.split('==')
                        elif '>' in line:
                            name, version = line.split('>')
                        else:
                            name, version = line, "unknown"
                        
                        dependencies.append({
                            "name": name.strip(),
                            "current_version": version.strip(),
                            "type": "python"
                        })
        
        elif request.package_file == "package.json":
            # Parse Node.js dependencies
            with open(package_file_path, 'r') as f:
                package_data = json.load(f)
                
                for dep_type in ["dependencies", "devDependencies"]:
                    if dep_type in package_data:
                        for name, version in package_data[dep_type].items():
                            dependencies.append({
                                "name": name,
                                "current_version": version.replace('^', '').replace('~', ''),
                                "type": "nodejs",
                                "dev_dependency": dep_type == "devDependencies"
                            })
        
        # Simulate security check (in real implementation, integrate with security APIs)
        security_alerts = []
        for dep in dependencies:
            # Mock security check based on common vulnerable packages
            if dep["name"].lower() in ["lodash", "express", "moment", "requests"]:
                security_alerts.append({
                    "package": dep["name"],
                    "severity": "medium",
                    "description": f"Known security issue in {dep['name']} version {dep['current_version']}"
                })
        
        return {
            "project_path": request.project_path,
            "package_file": request.package_file,
            "total_dependencies": len(dependencies),
            "dependencies": dependencies,
            "security_alerts": security_alerts,
            "recommendations": {
                "outdated_count": len([d for d in dependencies if "0.1" in d.get("current_version", "")]),
                "security_issues": len(security_alerts),
                "should_update": len(security_alerts) > 0
            },
            "scan_date": datetime.now().isoformat()
        }
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in package file")
    except Exception as e:
        raise ValueError(f"Error reading package file: {e}")


# ================================
# HTTP-ONLY ENDPOINTS (Web Interface)
# ================================

@app.get("/projects/discover")
async def discover_projects(root_path: str = ".") -> List[ProjectInfo]:
    """Discover projects in a directory tree (HTTP-only for web interfaces)."""
    projects = []
    root = Path(root_path)
    
    for path in root.rglob("*"):
        if path.is_dir() and _is_project_directory(path):
            try:
                project_info = _get_project_info(path)
                projects.append(project_info)
            except Exception:
                continue
    
    return projects[:20]  # Limit results


@app.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with system info (HTTP-only)."""
    return {
        "status": "healthy",
        "service": app.name,
        "version": app.version,
        "timestamp": datetime.now().isoformat(),
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd()
        },
        "features": {
            "dual_protocol": True,
            "async_support": True,
            "type_validation": True,
            "mcp_tools": len(app._tools)
        }
    }


# ================================
# MCP-ONLY TOOLS (AI Assistant Features)
# ================================

@app.tool(description="Get quick project summary for AI context")
async def get_project_context(project_path: str) -> Dict[str, Any]:
    """Get concise project context for AI assistants (MCP-only)."""
    path = Path(project_path)
    
    if not path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    
    # Quick analysis for AI context
    return {
        "path": project_path,
        "name": path.name,
        "type": _detect_project_type(path),
        "size": _get_project_size(path),
        "recent_activity": _get_recent_activity(path),
        "key_files": _get_key_files(path)
    }


@app.tool(description="Generate project documentation outline")
async def generate_docs_outline(project_path: str) -> Dict[str, Any]:
    """Generate documentation outline based on project structure (MCP-only for AI writing)."""
    path = Path(project_path)
    
    outline = {
        "project_name": path.name,
        "suggested_sections": [
            "Installation",
            "Quick Start",
            "API Reference",
            "Configuration",
            "Contributing",
            "License"
        ],
        "api_endpoints": [],
        "key_modules": []
    }
    
    # Scan for API endpoints and modules
    for py_file in path.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if "@app.route" in content or "@app.get" in content or "@app.post" in content:
                    outline["api_endpoints"].append(str(py_file.relative_to(path)))
                if "class " in content:
                    outline["key_modules"].append(str(py_file.relative_to(path)))
        except Exception:
            continue
    
    return outline


# ================================
# UTILITY FUNCTIONS
# ================================

def _get_language_name(extension: str) -> str:
    """Map file extension to language name."""
    mapping = {
        ".py": "Python",
        ".js": "JavaScript", 
        ".ts": "TypeScript",
        ".java": "Java",
        ".go": "Go",
        ".rs": "Rust",
        ".cpp": "C++",
        ".c": "C",
        ".rb": "Ruby",
        ".php": "PHP"
    }
    return mapping.get(extension, extension[1:].title())


def _categorize_project_size(total_lines: int) -> str:
    """Categorize project by size."""
    if total_lines < 1000:
        return "small"
    elif total_lines < 10000:
        return "medium"
    elif total_lines < 100000:
        return "large"
    else:
        return "enterprise"


def _parse_git_status(status_code: str) -> str:
    """Parse Git status code."""
    status_map = {
        "M ": "modified",
        " M": "modified",
        "A ": "added",
        " A": "added", 
        "D ": "deleted",
        " D": "deleted",
        "??": "untracked",
        "R ": "renamed",
        "C ": "copied"
    }
    return status_map.get(status_code, "unknown")


def _is_project_directory(path: Path) -> bool:
    """Check if directory looks like a project."""
    indicators = [
        "requirements.txt", "package.json", "Cargo.toml", 
        "pyproject.toml", "setup.py", ".git", "src", "lib"
    ]
    return any((path / indicator).exists() for indicator in indicators)


def _get_project_info(path: Path) -> ProjectInfo:
    """Get project information."""
    # Mock implementation for demo
    return ProjectInfo(
        name=path.name,
        path=str(path),
        language=_detect_project_type(path),
        framework=None,
        last_modified=datetime.now().isoformat(),
        size_mb=1.5,
        file_count=42
    )


def _detect_project_type(path: Path) -> str:
    """Detect project type/language."""
    if (path / "requirements.txt").exists() or (path / "pyproject.toml").exists():
        return "Python"
    elif (path / "package.json").exists():
        return "Node.js"
    elif (path / "Cargo.toml").exists():
        return "Rust"
    elif (path / "pom.xml").exists():
        return "Java"
    else:
        return "Unknown"


def _get_project_size(path: Path) -> str:
    """Get project size category."""
    try:
        file_count = len(list(path.rglob("*")))
        return "small" if file_count < 100 else "medium" if file_count < 500 else "large"
    except Exception:
        return "unknown"


def _get_recent_activity(path: Path) -> bool:
    """Check if project has recent activity."""
    return (path / ".git").exists()


def _get_key_files(path: Path) -> List[str]:
    """Get list of key files."""
    key_files = []
    candidates = ["README.md", "requirements.txt", "package.json", "main.py", "app.py", "index.js"]
    
    for candidate in candidates:
        if (path / candidate).exists():
            key_files.append(candidate)
    
    return key_files


# ================================
# SERVER ENTRY POINT
# ================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Developer Toolkit Service")
    parser.add_argument(
        "--mode",
        choices=["mcp", "http"],
        default="http",
        help="Run as MCP server (stdio) or HTTP server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8010, help="Port for HTTP server")
    
    args = parser.parse_args()
    
    if args.mode == "mcp":
        print("🔧 Starting Developer Toolkit as MCP server on stdio...", file=sys.stderr)
        print("Available MCP tools:", file=sys.stderr)
        for tool_name in app._tools.keys():
            print(f"  - {tool_name}", file=sys.stderr)
        asyncio.run(app.run_mcp())
    else:
        print(f"🌐 Starting Developer Toolkit HTTP server on {args.host}:{args.port}...", file=sys.stderr)
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs", file=sys.stderr)
        print("🔧 Available as both HTTP API and MCP tools!", file=sys.stderr)
        app.run_http(host=args.host, port=args.port)