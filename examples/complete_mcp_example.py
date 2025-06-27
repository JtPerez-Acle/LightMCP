"""
Complete MCP Protocol Example: Smart Development Assistant

This example demonstrates ALL THREE MCP protocol capabilities in a real-world scenario:
- Tools: Interactive development operations (build, test, analyze)
- Resources: Project data and configurations (docs, settings, metrics)
- Prompts: AI-assisted development workflows (code review, optimization)

Use cases:
- AI coding assistants can use tools to execute development tasks
- IDEs can access resources for project information and settings
- Code review systems can use prompts for structured analysis
- CI/CD pipelines can consume both tools and resources
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

# Initialize the smart development assistant
app = LightMCP(
    name="Smart Development Assistant",
    version="2.0.0",
    description="Complete MCP server with Tools, Resources, and Prompts for development workflows",
)

# ================================
# PYDANTIC MODELS
# ================================

class BuildRequest(BaseModel):
    """Request to build a project."""
    project_path: str = Field(..., description="Path to the project")
    build_type: str = Field(default="development", description="Build type: development, production, test")
    clean_first: bool = Field(default=False, description="Clean before building")

class BuildResult(BaseModel):
    """Result of a build operation."""
    success: bool
    build_type: str
    duration_seconds: float
    output: str
    errors: Optional[str] = None
    timestamp: str

class LintRequest(BaseModel):
    """Request to lint code."""
    project_path: str = Field(..., description="Path to the project")
    file_patterns: List[str] = Field(default=["*.py"], description="File patterns to lint")
    fix_issues: bool = Field(default=False, description="Automatically fix issues")

class LintResult(BaseModel):
    """Result of linting operation."""
    files_checked: int
    issues_found: int
    issues_fixed: int
    severity_breakdown: Dict[str, int]
    details: List[Dict[str, Any]]
    timestamp: str

# ================================
# MCP TOOLS - Interactive Operations
# ================================

@app.tool(description="Build project with specified configuration")
@app.post("/build", response_model=BuildResult)
async def build_project(request: BuildRequest) -> BuildResult:
    """
    Build a project - available as both MCP tool and HTTP endpoint.
    
    This demonstrates how the same build functionality can be used by:
    - AI assistants via MCP tools
    - CI/CD systems via HTTP API
    """
    project_path = Path(request.project_path)
    
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {request.project_path}")
    
    start_time = datetime.now()
    
    try:
        # Determine build command based on project type
        if (project_path / "package.json").exists():
            build_cmd = ["npm", "run", "build"]
            if request.clean_first:
                subprocess.run(["npm", "run", "clean"], cwd=project_path, check=True)
        elif (project_path / "pyproject.toml").exists():
            build_cmd = ["python", "-m", "build"]
            if request.clean_first:
                subprocess.run(["rm", "-rf", "build", "dist"], cwd=project_path)
        elif (project_path / "Cargo.toml").exists():
            build_cmd = ["cargo", "build"]
            if request.clean_first:
                subprocess.run(["cargo", "clean"], cwd=project_path, check=True)
        else:
            # Generic Python project
            build_cmd = ["python", "-m", "pip", "install", "-e", "."]
        
        # Add build type flags
        if request.build_type == "production" and "npm" in build_cmd[0]:
            build_cmd.extend(["--", "--mode=production"])
        elif request.build_type == "production" and "cargo" in build_cmd[0]:
            build_cmd.append("--release")
        
        # Execute build
        result = subprocess.run(
            build_cmd,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return BuildResult(
            success=result.returncode == 0,
            build_type=request.build_type,
            duration_seconds=duration,
            output=result.stdout,
            errors=result.stderr if result.stderr else None,
            timestamp=start_time.isoformat()
        )
        
    except subprocess.TimeoutExpired:
        raise ValueError("Build timed out after 5 minutes")
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return BuildResult(
            success=False,
            build_type=request.build_type,
            duration_seconds=duration,
            output="",
            errors=str(e),
            timestamp=start_time.isoformat()
        )

@app.tool(description="Lint and analyze code quality")
@app.post("/lint", response_model=LintResult)
async def lint_code(request: LintRequest) -> LintResult:
    """
    Lint code for quality issues - dual protocol support.
    
    AI assistants can use this to check code quality,
    IDEs can integrate it for real-time feedback.
    """
    project_path = Path(request.project_path)
    
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {request.project_path}")
    
    try:
        # Collect files to lint
        files_to_check = []
        for pattern in request.file_patterns:
            files_to_check.extend(project_path.rglob(pattern))
        
        # Run linting (using flake8 for Python as example)
        lint_cmd = ["flake8"] + [str(f) for f in files_to_check]
        if request.fix_issues:
            # Use autopep8 or black for fixing
            fix_cmd = ["autopep8", "--in-place"] + [str(f) for f in files_to_check]
            subprocess.run(fix_cmd, cwd=project_path, capture_output=True)
        
        result = subprocess.run(
            lint_cmd,
            cwd=project_path,
            capture_output=True,
            text=True
        )
        
        # Parse linting output (simplified)
        issues = []
        severity_counts = {"error": 0, "warning": 0, "info": 0}
        
        for line in result.stdout.split('\n'):
            if line.strip() and ':' in line:
                parts = line.split(':')
                if len(parts) >= 4:
                    file_path = parts[0]
                    line_num = parts[1]
                    severity = "warning"  # Default
                    message = ':'.join(parts[3:]).strip()
                    
                    issues.append({
                        "file": file_path,
                        "line": line_num,
                        "severity": severity,
                        "message": message
                    })
                    severity_counts[severity] += 1
        
        return LintResult(
            files_checked=len(files_to_check),
            issues_found=len(issues),
            issues_fixed=0 if not request.fix_issues else len(issues) // 2,  # Estimate
            severity_breakdown=severity_counts,
            details=issues[:20],  # Limit for demo
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise ValueError(f"Linting failed: {e}")

@app.tool(description="Get comprehensive project development status")
async def get_project_status(project_path: str) -> Dict[str, Any]:
    """
    Get complete project development status - MCP tool only.
    
    This provides rich context for AI assistants to understand
    the current state of a development project.
    """
    path = Path(project_path)
    
    if not path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    
    status = {
        "project_path": project_path,
        "project_name": path.name,
        "timestamp": datetime.now().isoformat(),
        "git_info": {},
        "dependencies": {},
        "test_coverage": {},
        "build_status": "unknown",
        "code_quality": {}
    }
    
    # Git information
    if (path / ".git").exists():
        try:
            # Current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=path, capture_output=True, text=True, check=True
            )
            status["git_info"]["current_branch"] = branch_result.stdout.strip()
            
            # Recent commits
            log_result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=path, capture_output=True, text=True, check=True
            )
            status["git_info"]["recent_commits"] = log_result.stdout.strip().split('\n')
            
            # Status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path, capture_output=True, text=True, check=True
            )
            status["git_info"]["has_changes"] = bool(status_result.stdout.strip())
            
        except subprocess.CalledProcessError:
            status["git_info"]["error"] = "Could not read git information"
    
    # Dependencies status
    if (path / "requirements.txt").exists():
        try:
            with open(path / "requirements.txt", 'r') as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                status["dependencies"]["python"] = len(deps)
        except Exception:
            pass
    
    if (path / "package.json").exists():
        try:
            with open(path / "package.json", 'r') as f:
                package_data = json.load(f)
                deps = len(package_data.get("dependencies", {}))
                dev_deps = len(package_data.get("devDependencies", {}))
                status["dependencies"]["nodejs"] = {"prod": deps, "dev": dev_deps}
        except Exception:
            pass
    
    return status

# ================================
# MCP RESOURCES - Project Data Access
# ================================

@app.resource(
    uri="config://project/settings",
    description="Project configuration and settings",
    mime_type="application/json"
)
@app.get("/config/project")
async def get_project_config() -> Dict[str, Any]:
    """
    Project configuration - available as both MCP resource and HTTP endpoint.
    
    AI assistants can read this to understand project setup,
    IDEs can use it for configuration management.
    """
    return {
        "project": {
            "name": "Smart Development Assistant",
            "version": "2.0.0",
            "language": "Python",
            "framework": "LightMCP"
        },
        "development": {
            "auto_lint": True,
            "auto_format": True,
            "run_tests_on_save": False,
            "build_on_commit": True
        },
        "quality": {
            "min_test_coverage": 80,
            "max_complexity": 10,
            "enforce_type_hints": True,
            "style_guide": "PEP8"
        },
        "integrations": {
            "ci_cd": "GitHub Actions",
            "code_review": "automated",
            "deployment": "containerized"
        }
    }

@app.resource(
    uri="data://project/metrics",
    description="Real-time project development metrics",
    mime_type="application/json"
)
async def get_project_metrics() -> Dict[str, Any]:
    """
    Live project metrics - MCP resource only.
    
    Provides current development metrics for monitoring and analysis.
    """
    # Simulate getting real metrics (would integrate with actual tools)
    return {
        "code_quality": {
            "total_lines": 2547,
            "test_coverage": 87.3,
            "cyclomatic_complexity": 6.2,
            "maintainability_index": 92.4,
            "tech_debt_hours": 2.1
        },
        "development_velocity": {
            "commits_this_week": 23,
            "pull_requests_merged": 4,
            "issues_closed": 7,
            "average_pr_size": 156,
            "code_review_time_hours": 3.2
        },
        "build_performance": {
            "average_build_time": 45.2,
            "success_rate": 94.1,
            "test_execution_time": 12.8,
            "deployment_frequency": "daily"
        },
        "last_updated": datetime.now().isoformat()
    }

@app.resource(
    uri="docs://development/guidelines",
    description="Development guidelines and best practices",
    mime_type="text/markdown"
)
async def get_development_guidelines() -> str:
    """
    Development guidelines - MCP resource only.
    
    Provides team guidelines that AI assistants can reference
    when helping with code reviews and development decisions.
    """
    return """# Development Guidelines

## Code Quality Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable and function names

### Testing Requirements
- Minimum 80% test coverage for new code
- Write unit tests for all public functions
- Integration tests for API endpoints
- Performance tests for critical paths

### Code Review Process
- All code must be reviewed before merging
- Focus on correctness, readability, and maintainability
- Check for security vulnerabilities
- Verify test coverage meets requirements

## Git Workflow

### Branch Naming
- `feature/TICKET-description` for new features
- `bugfix/TICKET-description` for bug fixes
- `hotfix/TICKET-description` for urgent fixes

### Commit Messages
- Use conventional commit format
- Include ticket number in commit message
- Write clear, descriptive commit messages

### Pull Request Guidelines
- Small, focused changes (< 400 lines)
- Clear description of changes
- Link to relevant tickets/issues
- Update documentation if needed

## Development Environment

### Local Setup
1. Use Python 3.11+ for all development
2. Install dependencies with `pip install -e .[dev]`
3. Run tests with `pytest` before committing
4. Use pre-commit hooks for code formatting

### CI/CD Pipeline
- Automated testing on all pull requests
- Code quality checks with SonarQube
- Security scanning with Snyk
- Automatic deployment to staging

## Best Practices

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Log errors with appropriate severity
- Handle edge cases gracefully

### Performance
- Profile critical code paths
- Use async/await for I/O operations
- Cache expensive computations
- Monitor memory usage

### Security
- Validate all inputs
- Use parameterized queries
- Keep dependencies updated
- Never commit secrets to git
"""

# ================================
# MCP PROMPTS - AI-Assisted Workflows
# ================================

@app.prompt(
    description="Generate comprehensive code review checklist",
    arguments=[
        {"name": "file_path", "description": "Path to the file being reviewed", "required": True},
        {"name": "change_type", "description": "Type of change: feature, bugfix, refactor", "required": True},
        {"name": "complexity", "description": "Complexity level: low, medium, high", "required": False}
    ]
)
async def code_review_checklist(file_path: str, change_type: str, complexity: str = "medium") -> Dict[str, Any]:
    """
    Generate context-aware code review checklist.
    
    AI assistants can use this to provide structured code review guidance
    based on the specific file and type of changes being made.
    """
    base_checklist = [
        "Code follows established style guidelines",
        "All functions have appropriate type hints",
        "Error handling is comprehensive and appropriate",
        "Code is well-documented with clear docstrings",
        "No obvious security vulnerabilities",
        "Performance implications have been considered"
    ]
    
    # Customize based on change type
    if change_type == "feature":
        base_checklist.extend([
            "New feature is properly tested",
            "API changes are backward compatible",
            "Documentation has been updated",
            "Feature flags are used if appropriate"
        ])
    elif change_type == "bugfix":
        base_checklist.extend([
            "Root cause of bug has been identified",
            "Fix addresses the underlying issue",
            "Regression test has been added",
            "Similar issues in codebase have been checked"
        ])
    elif change_type == "refactor":
        base_checklist.extend([
            "Functionality remains unchanged",
            "Performance impact has been measured",
            "All tests still pass",
            "Code complexity has been reduced"
        ])
    
    # Adjust for complexity
    if complexity == "high":
        base_checklist.extend([
            "Complex logic is well-commented",
            "Algorithm choice is justified",
            "Memory usage patterns are appropriate",
            "Concurrent access patterns are safe"
        ])
    
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are reviewing changes to {file_path}. This is a {change_type} with {complexity} complexity."
            },
            {
                "role": "user",
                "content": f"Please review the code changes using this checklist:\n\n" + 
                          "\n".join([f"- [ ] {item}" for item in base_checklist]) +
                          f"\n\nFocus particularly on aspects relevant to {change_type} changes."
            }
        ],
        "checklist": base_checklist,
        "metadata": {
            "file_path": file_path,
            "change_type": change_type,
            "complexity": complexity,
            "checklist_items": len(base_checklist)
        }
    }

@app.prompt(
    description="Generate performance optimization suggestions",
    arguments=[
        {"name": "function_name", "description": "Name of the function to optimize", "required": True},
        {"name": "performance_metrics", "description": "Current performance metrics", "required": False},
        {"name": "optimization_goals", "description": "Specific optimization goals", "required": False}
    ]
)
async def performance_optimization_prompt(
    function_name: str, 
    performance_metrics: str = "", 
    optimization_goals: str = "general performance"
) -> Dict[str, Any]:
    """
    Generate targeted performance optimization guidance.
    
    Helps AI assistants provide specific, actionable performance improvement
    suggestions based on the function and current metrics.
    """
    optimization_areas = [
        "Algorithm complexity analysis",
        "Memory usage optimization",
        "I/O operations efficiency",
        "Caching opportunities",
        "Parallel processing potential",
        "Database query optimization",
        "Network request batching",
        "Resource pooling benefits"
    ]
    
    metrics_context = f"Current metrics: {performance_metrics}" if performance_metrics else "No specific metrics provided"
    goals_context = f"Optimization goals: {optimization_goals}"
    
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are a performance optimization expert analyzing the {function_name} function."
            },
            {
                "role": "user",
                "content": f"""Analyze the performance of the {function_name} function and suggest optimizations.

{metrics_context}
{goals_context}

Please consider these optimization areas:
{chr(10).join([f"- {area}" for area in optimization_areas])}

Provide specific, actionable recommendations with:
1. The optimization technique
2. Expected performance impact
3. Implementation complexity
4. Any trade-offs or risks
5. Code examples where helpful"""
            }
        ],
        "optimization_areas": optimization_areas,
        "metadata": {
            "function_name": function_name,
            "has_metrics": bool(performance_metrics),
            "optimization_goals": optimization_goals
        }
    }

@app.prompt(
    description="Generate debugging strategy for complex issues",
    arguments=[
        {"name": "error_description", "description": "Description of the error or issue", "required": True},
        {"name": "error_frequency", "description": "How often the error occurs", "required": False},
        {"name": "environment", "description": "Environment where error occurs", "required": False}
    ]
)
async def debugging_strategy_prompt(
    error_description: str,
    error_frequency: str = "occasional",
    environment: str = "production"
) -> Dict[str, Any]:
    """
    Generate systematic debugging approach for complex issues.
    
    Provides structured debugging methodology that AI assistants
    can use to help developers troubleshoot difficult problems.
    """
    debugging_steps = [
        "Reproduce the issue consistently",
        "Collect comprehensive error logs",
        "Identify the smallest failing case",
        "Check recent changes and deployments",
        "Verify environment configuration",
        "Review related code paths",
        "Test with different input variations",
        "Monitor resource usage patterns"
    ]
    
    # Customize based on frequency and environment
    if error_frequency == "intermittent":
        debugging_steps.extend([
            "Add detailed logging around suspected areas",
            "Implement error tracking and monitoring",
            "Check for race conditions or timing issues"
        ])
    
    if environment == "production":
        debugging_steps.extend([
            "Use production-safe debugging techniques",
            "Consider impact of debugging on performance",
            "Plan rollback strategy if needed"
        ])
    
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are helping debug an issue that occurs {error_frequency} in {environment}."
            },
            {
                "role": "user",
                "content": f"""Help me debug this issue: {error_description}

This error occurs {error_frequency} in {environment} environment.

Please provide a systematic debugging approach following these steps:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(debugging_steps)])}

For each step, provide:
- Specific actions to take
- Tools or commands to use
- What to look for in the results
- How to interpret the findings
- Next steps based on different outcomes"""
            }
        ],
        "debugging_steps": debugging_steps,
        "metadata": {
            "error_description": error_description,
            "error_frequency": error_frequency,
            "environment": environment,
            "total_steps": len(debugging_steps)
        }
    }

# ================================
# HTTP-ONLY ENDPOINTS
# ================================

@app.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Comprehensive health check for monitoring systems."""
    return {
        "status": "healthy",
        "service": app.name,
        "version": app.version,
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "mcp_tools": len(app._tools),
            "mcp_resources": len(app._resources),
            "mcp_prompts": len(app._prompts),
            "dual_protocol": True
        },
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd()
        }
    }

# ================================
# SERVER ENTRY POINT
# ================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Development Assistant")
    parser.add_argument(
        "--mode",
        choices=["mcp", "http"],
        default="http",
        help="Run as MCP server (stdio) or HTTP server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8011, help="Port for HTTP server")
    
    args = parser.parse_args()
    
    if args.mode == "mcp":
        print("🤖 Starting Smart Development Assistant as MCP server...", file=sys.stderr)
        print(f"📋 Available: {len(app._tools)} tools, {len(app._resources)} resources, {len(app._prompts)} prompts", file=sys.stderr)
        asyncio.run(app.run_mcp())
    else:
        print(f"🌐 Starting Smart Development Assistant HTTP server on {args.host}:{args.port}...", file=sys.stderr)  
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs", file=sys.stderr)
        print(f"🤖 MCP Capabilities: {len(app._tools)} tools, {len(app._resources)} resources, {len(app._prompts)} prompts", file=sys.stderr)
        app.run_http(host=args.host, port=args.port)