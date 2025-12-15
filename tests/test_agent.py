#!/usr/bin/env python3
"""Test script to verify the cleaned agent works correctly."""

import asyncio

from agent import Pipeline


async def test_agent():
    """Test that the agent initializes and runs correctly."""
    print("Testing cleaned agent...")
    
    # Initialize the pipeline
    pipeline = Pipeline()
    print("Pipeline initialized successfully")
    
    # Test the pipe method with a simple message
    test_body = {
        "messages": [
            {"role": "user", "content": "Test message"}
        ]
    }
    
    try:
        # This should not fail with our cleaned version
        async for chunk in pipeline.pipe(test_body):
            print(f"Response: {chunk}")
        print("Agent test completed successfully")
    except Exception as e:
        print(f"Error during agent test: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_agent())
