#!/usr/bin/env python3
"""Simple test to verify agent structure without network dependencies."""

import asyncio
from agent import Pipeline

async def test_agent_structure():
    """Test that the agent initializes and has correct structure."""
    print("Testing agent structure...")
    
    # Initialize the pipeline
    pipeline = Pipeline()
    print("✅ Pipeline initialized successfully")
    
    # Check that required attributes exist
    assert hasattr(pipeline, 'name'), "Pipeline missing name attribute"
    assert hasattr(pipeline, 'type'), "Pipeline missing type attribute"
    assert hasattr(pipeline, 'valves'), "Pipeline missing valves attribute"
    assert hasattr(pipeline, 'pipe'), "Pipeline missing pipe method"
    assert hasattr(pipeline, 'get_tools'), "Pipeline missing get_tools method"
    
    print("✅ All required attributes present")
    
    # Test that pipe method is callable
    assert callable(pipeline.pipe), "pipe method is not callable"
    print("✅ pipe method is callable")
    
    # Test that get_tools method is callable
    assert callable(pipeline.get_tools), "get_tools method is not callable"
    print("✅ get_tools method is callable")
    
    print("All structure tests passed! ✅")

if __name__ == "__main__":
    asyncio.run(test_agent_structure())
