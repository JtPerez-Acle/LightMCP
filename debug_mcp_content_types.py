#!/usr/bin/env python3
"""
Debug MCP Content Types

Check what content types are available and their structure.
"""

import sys
sys.path.insert(0, "src")

from mcp.types import (
    TextContent, BlobResourceContents, TextResourceContents, 
    Content, ResourceContents, ImageContent, AudioContent
)

print("=== MCP CONTENT TYPES ===")

content_types = [
    ("TextContent", TextContent),
    ("TextResourceContents", TextResourceContents),
    ("BlobResourceContents", BlobResourceContents),
    ("Content", Content),
    ("ResourceContents", ResourceContents),
    ("ImageContent", ImageContent),
    ("AudioContent", AudioContent),
]

for name, content_type in content_types:
    print(f"\n{name}:")
    try:
        if hasattr(content_type, 'model_fields'):
            fields = content_type.model_fields
            print(f"  Fields: {list(fields.keys())}")
            for field_name, field_info in fields.items():
                print(f"    {field_name}: {field_info.annotation}")
        else:
            print(f"  No model_fields attribute")
        
        # Check if it has a 'content' attribute expectation
        print(f"  Type: {content_type}")
        
        # Try to create a simple instance to see structure
        if name == "TextContent":
            instance = TextContent(type="text", text="test")
            print(f"  Sample: {instance}")
            print(f"  Attributes: {[attr for attr in dir(instance) if not attr.startswith('_')]}")
        elif name == "ImageContent":
            # Skip this one as it likely requires more complex data
            print(f"  (Skipping complex instantiation)")
        elif name == "AudioContent":
            # Skip this one as it likely requires more complex data
            print(f"  (Skipping complex instantiation)")
        
    except Exception as e:
        print(f"  Error exploring {name}: {e}")

# Test what works with read_resource
print(f"\n=== TESTING READ_RESOURCE EXPECTATIONS ===")

# Look at the source or try different approaches
print("The MCP library expects handlers to return specific formats.")
print("Based on the error, it's looking for .content attribute on content_item.")
print("Let's see if TextContent has this...")

try:
    text_content = TextContent(type="text", text="Hello World")
    print(f"TextContent instance: {text_content}")
    print(f"Has 'content' attr: {hasattr(text_content, 'content')}")
    print(f"Has 'text' attr: {hasattr(text_content, 'text')}")
    
    if hasattr(text_content, 'content'):
        print(f"Content value: {text_content.content}")
    if hasattr(text_content, 'text'):
        print(f"Text value: {text_content.text}")
        
except Exception as e:
    print(f"Error with TextContent: {e}")