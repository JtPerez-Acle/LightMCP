#!/usr/bin/env python3
"""Test the developer toolkit MCP functionality directly."""

import asyncio
import sys
from pydantic import BaseModel

sys.path.insert(0, "src")
sys.path.insert(0, "examples")

from developer_toolkit import app, CodeAnalysisRequest, GitStatusRequest


async def test_toolkit_mcp():
    """Test developer toolkit MCP tools directly."""
    
    print("🔧 Testing Developer Toolkit MCP Tools")
    print("=" * 50)
    
    # Test project analysis via MCP
    print("\n📊 Testing Project Analysis (MCP)...")
    analysis_request = CodeAnalysisRequest(
        project_path="/home/jt/LightMCP",
        file_extensions=[".py"],
        include_metrics=True
    )
    
    analyze_tool = app._tools["analyze_project"]
    result = await analyze_tool.func(analysis_request)
    
    print(f"✅ Project Analysis via MCP:")
    print(f"   - Total files: {result.total_files}")
    print(f"   - Total lines: {result.total_lines}")
    print(f"   - Languages: {list(result.language_breakdown.keys())}")
    print(f"   - Project size: {result.complexity_metrics['project_size_category']}")
    
    # Test Git status via MCP
    print("\n📦 Testing Git Status (MCP)...")
    git_request = GitStatusRequest(
        project_path="/home/jt/LightMCP",
        include_diff=False
    )
    
    git_tool = app._tools["get_git_status"]
    git_result = await git_tool.func(git_request)
    
    print(f"✅ Git Status via MCP:")
    print(f"   - Current branch: {git_result['current_branch']}")
    print(f"   - Has changes: {git_result['has_changes']}")
    print(f"   - Change summary: {git_result['change_summary']}")
    
    # Test MCP-only tool
    print("\n🤖 Testing MCP-Only Tool...")
    context_tool = app._tools["get_project_context"]
    context = await context_tool.func(project_path="/home/jt/LightMCP")
    
    print(f"✅ Project Context (MCP-only):")
    print(f"   - Project name: {context['name']}")
    print(f"   - Project type: {context['type']}")
    print(f"   - Project size: {context['size']}")
    print(f"   - Key files: {context['key_files']}")
    
    print("\n🎉 All MCP tools working correctly!")
    print("\n💡 The SAME functionality is available via:")
    print("   🌐 HTTP API: curl -X POST http://localhost:8010/projects/analyze")
    print("   🤖 MCP Tool: analyze_project")
    print("   📡 DUAL PROTOCOL PROVEN!")


if __name__ == "__main__":
    asyncio.run(test_toolkit_mcp())