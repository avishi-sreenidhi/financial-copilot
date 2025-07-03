#!/usr/bin/env python3
"""
ChatGPT-Style Data Analytics Demo

This script demonstrates the ChatGPT-like conversation capabilities
of the enhanced data analytics agent with memory and context awareness.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from langgraph_engine.graph_builder import build_agent_graph
from langchain_core.messages import HumanMessage, AIMessage
import time

def demo_conversation():
    """Demonstrate a ChatGPT-style conversation"""
    print("🎯 ChatGPT-Style Data Analytics Assistant Demo")
    print("=" * 60)
    
    workflow = build_agent_graph()
    session_id = "demo_session"
    messages = []
    
    # Conversation scenarios
    conversation = [
        "Hello! I'm new to data analysis.",
        "What can you help me with?",
        "I have a CSV file with sales data. What can I do with it?",
        "How do I get started with analyzing it?",
        "Thanks! That's very helpful."
    ]
    
    for i, user_message in enumerate(conversation, 1):
        print(f"\n{i}️⃣ 👤 User: {user_message}")
        
        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        messages.append(user_msg)
        
        # Create state for this interaction
        state = {
            "query": user_message,
            "messages": messages.copy(),
            "next_agent": "",
            "current_agent": "",
            "agent_outputs": {},
            "dataframe_info": {},
            "has_data": False,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0,
            "chat_response": {},
            "session_id": session_id,
            "conversation_summary": ""
        }
        
        try:
            # Process the message
            start_time = time.time()
            result = workflow.invoke(state)
            end_time = time.time()
            
            # Extract response
            chat_response = result.get("chat_response", {})
            agent_response = chat_response.get("message", result.get("final_result", "Sorry, I couldn't process your request."))
            
            # Display response
            print(f"   🤖 Assistant: {agent_response}")
            
            # Show metadata
            context_aware = chat_response.get("context_aware", False)
            agent = chat_response.get("agent", "unknown")
            processing_time = end_time - start_time
            
            print(f"   📊 Agent: {agent} | Context Aware: {context_aware} | Time: {processing_time:.2f}s")
            
            # Show conversation summary from memory
            memory_output = result.get("agent_outputs", {}).get("memory", {})
            if memory_output and memory_output.get("result"):
                memory_result = memory_output["result"]
                summary = memory_result.get("conversation_summary", "")
                if summary and "New conversation" not in summary:
                    print(f"   💭 Context: {summary}")
            
            # Add assistant response to conversation
            assistant_msg = AIMessage(content=agent_response)
            messages.append(assistant_msg)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print(f"\n🎉 Demo completed! Conversation with {len(messages)} messages.")
    print("\n✨ Key Features Demonstrated:")
    print("   • Memory and context awareness across conversations")
    print("   • Natural, ChatGPT-like response formatting") 
    print("   • Conversation continuity and relationship detection")
    print("   • Context-aware response adaptation")
    print("   • Multi-agent coordination with chat interface")

def show_features():
    """Show the key features of the chat system"""
    print("\n🚀 ChatGPT-Style Features:")
    print("=" * 40)
    
    features = [
        "💭 Conversation Memory - Remembers context across messages",
        "🎯 Context Awareness - Adapts responses based on conversation history", 
        "🔄 Continuity - Maintains natural conversation flow",
        "🤖 Multi-Agent - Router, Memory, Pandas, Python, Chart, Search agents",
        "💬 Chat Formatting - Formats responses for natural conversation",
        "📊 Analytics Focus - Specialized for data analysis tasks",
        "🧠 Smart Routing - Routes queries to appropriate specialized agents",
        "⚡ Token Optimization - Prevents OpenAI context length errors"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n📝 Usage Examples:")
    print("   • 'Hello! I'm new to data analysis.'")
    print("   • 'What can you do with CSV files?'") 
    print("   • 'How do I get started?'")
    print("   • 'Can you create a chart from my data?'")
    print("   • 'Find all rows where sales > 1000'")

if __name__ == "__main__":
    show_features()
    print("\n" + "="*60)
    demo_conversation()
