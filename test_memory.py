#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from langgraph_engine.graph_builder import build_agent_graph
from langchain_core.messages import HumanMessage

def test_memory_chat():
    """Test the memory-enhanced chat system"""
    
    # Create the workflow
    workflow = build_agent_graph()
    
    print("🧪 Testing Memory-Enhanced Chat System")
    print("=" * 50)
    
    # Test 1: First message
    print("\n1️⃣ First interaction:")
    state1 = {
        "query": "Hello! I'm new here.",
        "messages": [HumanMessage(content="Hello! I'm new here.")],
        "next_agent": "",
        "current_agent": "",
        "agent_outputs": {},
        "dataframe_info": {},
        "has_data": False,
        "final_result": "",
        "metadata": {},
        "iteration_count": 0,
        "chat_response": {},
        "session_id": "test_session",
        "conversation_summary": ""
    }
    
    try:
        result1 = workflow.invoke(state1)
        chat_response1 = result1.get("chat_response", {})
        print(f"   Response: {chat_response1.get('message', result1.get('final_result', 'No response'))}")
        print(f"   Context aware: {chat_response1.get('context_aware', False)}")
        
        # Test 2: Follow-up message (should have context)
        print("\n2️⃣ Follow-up interaction:")
        messages2 = state1["messages"] + [
            HumanMessage(content=chat_response1.get('message', 'Hello')),
            HumanMessage(content="What can you do with CSV files?")
        ]
        
        state2 = state1.copy()
        state2.update({
            "query": "What can you do with CSV files?", 
            "messages": messages2,
            "agent_outputs": {},
            "iteration_count": 0
        })
        
        result2 = workflow.invoke(state2)
        chat_response2 = result2.get("chat_response", {})
        print(f"   Response: {chat_response2.get('message', result2.get('final_result', 'No response'))}")
        print(f"   Context aware: {chat_response2.get('context_aware', False)}")
        
        # Test 3: Another follow-up
        print("\n3️⃣ Third interaction:")
        messages3 = messages2 + [
            HumanMessage(content=chat_response2.get('message', 'I can help with CSV')),
            HumanMessage(content="Thanks! How do I get started?")
        ]
        
        state3 = state2.copy()
        state3.update({
            "query": "Thanks! How do I get started?",
            "messages": messages3,
            "agent_outputs": {},
            "iteration_count": 0
        })
        
        result3 = workflow.invoke(state3)
        chat_response3 = result3.get("chat_response", {})
        print(f"   Response: {chat_response3.get('message', result3.get('final_result', 'No response'))}")
        print(f"   Context aware: {chat_response3.get('context_aware', False)}")
        
        print("\n✅ Memory system test completed!")
        
        # Check if memory context was generated
        memory_output = result3.get("agent_outputs", {}).get("memory", {})
        if memory_output:
            print(f"\n🧠 Memory Context Generated:")
            memory_result = memory_output.get("result", {})
            print(f"   Summary: {memory_result.get('conversation_summary', 'N/A')}")
            print(f"   Relationship: {memory_result.get('query_relationship', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_chat()
