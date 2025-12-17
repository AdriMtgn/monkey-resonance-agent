#!/usr/bin/env python3
"""Test script to verify the cleaned agent works correctly."""

import asyncio

from agent import Pipeline


def test_agent():
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
        # Run the async pipe method synchronously for testing
        async def run_pipe():
            chunks = []
            async for chunk in pipeline.pipe(test_body):
                chunks.append(chunk)
            return chunks
        
        # Run in event loop
        chunks = asyncio.run(run_pipe())
        print(f"Response chunks: {chunks}")
        print("Agent test completed successfully")
    except Exception as e:
        print(f"Error during agent test: {e}")
        raise

if __name__ == "__main__":
    test_agent()
