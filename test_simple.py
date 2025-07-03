#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from langgraph_engine.graph_builder import build_agent_graph
from langchain_openai import ChatOpenAI

def test_different_queries():
    """Test different types of queries to ensure pandas agent works correctly"""
    
    # Create the workflow
    workflow = build_agent_graph()
    
    test_queries = [
        "hello",
        "help me with data analysis",
        "what can you do with CSV files?",
        "I want to analyze my data",
        "can you describe a dataset?",
        "how do I upload an Excel file?"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing query: '{query}' ===")
        
        test_state = {
            "query": query,
            "messages": [],
            "next_agent": "",
            "current_agent": "",
            "agent_outputs": {},
            "dataframe_info": {},
            "has_data": False,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0
        }
        
        try:
            # Run the workflow
            result = workflow.invoke(test_state)
            
            # Extract final result
            final_result = result.get("final_result", "")
            print(f"✅ Success! Result: {final_result[:200]}...")
            
        except Exception as e:
            print(f"❌ Error for query '{query}': {e}")
            if "context_length_exceeded" in str(e):
                print("   This is the token limit error we're trying to fix!")
                
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    test_different_queries()
