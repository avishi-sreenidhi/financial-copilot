"""
Chat Interface for Data Analytics Agent

A simple ChatGPT-style interface for testing the multi-agent data analytics system
with conversation memory and context awareness.
"""

import uuid
import time
from typing import Dict, Any, List
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from langgraph_engine.graph_builder import build_agent_graph, DataAnalyticsState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

class ChatInterface:
    """Simple chat interface for the data analytics agent"""
    
    def __init__(self):
        self.workflow = build_agent_graph()
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.messages = []
        
    def send_message(self, user_message: str) -> Dict[str, Any]:
        """
        Send a message to the agent and get a response
        
        Args:
            user_message: The user's message
            
        Returns:
            Dictionary containing the agent's response and metadata
        """
        print(f"\n🤖 Processing: {user_message}")
        
        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.messages.append(user_msg)
        
        # Create state for this interaction
        state = {
            "query": user_message,
            "messages": self.messages,
            "next_agent": "",
            "current_agent": "",
            "agent_outputs": {},
            "dataframe_info": {},
            "has_data": False,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0,
            "chat_response": {},
            "session_id": self.session_id,
            "conversation_summary": ""
        }
        
        try:
            # Run the workflow
            start_time = time.time()
            result = self.workflow.invoke(state)
            end_time = time.time()
            
            # Extract chat response
            chat_response = result.get("chat_response", {})
            agent_response = chat_response.get("message", result.get("final_result", "Sorry, I couldn't process your request."))
            
            # Add agent response to conversation
            agent_msg = AIMessage(content=agent_response)
            self.messages.append(agent_msg)
            
            # Store conversation entry
            conversation_entry = {
                "user_message": user_message,
                "agent_response": agent_response,
                "timestamp": time.time(),
                "processing_time": end_time - start_time,
                "agent": chat_response.get("agent", "unknown"),
                "context_aware": chat_response.get("context_aware", False),
                "conversation_summary": chat_response.get("conversation_summary", "")
            }
            
            self.conversation_history.append(conversation_entry)
            
            return {
                "response": agent_response,
                "metadata": conversation_entry,
                "full_result": result
            }
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            print(f"❌ Error: {e}")
            
            # Add error response to conversation
            error_response = AIMessage(content=error_msg)
            self.messages.append(error_response)
            
            return {
                "response": error_msg,
                "metadata": {"error": str(e), "timestamp": time.time()},
                "full_result": {}
            }
    
    def print_conversation(self):
        """Print the conversation history in a chat-like format"""
        print("\n" + "="*60)
        print("📱 CONVERSATION HISTORY")
        print("="*60)
        
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            
            print(f"\n[{timestamp}] 👤 You:")
            print(f"  {entry['user_message']}")
            
            print(f"\n[{timestamp}] 🤖 Assistant ({entry['agent']}):")
            print(f"  {entry['agent_response']}")
            
            if entry.get("context_aware"):
                print(f"  💭 Context: {entry.get('conversation_summary', 'N/A')}")
            
            processing_time = entry.get("processing_time", 0)
            print(f"  ⏱️  Processing time: {processing_time:.2f}s")
            
            if i < len(self.conversation_history):
                print("-" * 40)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {"message": "No conversation yet"}
        
        total_messages = len(self.conversation_history)
        total_time = sum(entry.get("processing_time", 0) for entry in self.conversation_history)
        avg_time = total_time / total_messages if total_messages > 0 else 0
        
        agents_used = set(entry.get("agent", "unknown") for entry in self.conversation_history)
        context_aware_count = sum(1 for entry in self.conversation_history if entry.get("context_aware", False))
        
        return {
            "total_interactions": total_messages,
            "total_processing_time": round(total_time, 2),
            "average_processing_time": round(avg_time, 2),
            "agents_used": list(agents_used),
            "context_aware_responses": context_aware_count,
            "context_awareness_rate": round((context_aware_count / total_messages) * 100, 1) if total_messages > 0 else 0,
            "session_id": self.session_id
        }


def main():
    """Main chat interface loop"""
    print("🎯 Data Analytics Chat Assistant")
    print("Type 'exit' to quit, 'history' to see conversation, 'stats' for statistics")
    print("=" * 60)
    
    chat = ChatInterface()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Thanks for using the Data Analytics Assistant!")
                break
            
            elif user_input.lower() == 'history':
                chat.print_conversation()
                continue
            
            elif user_input.lower() == 'stats':
                stats = chat.get_stats()
                print(f"\n📊 CONVERSATION STATS:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print("🎯 Data Analytics Chat Assistant")
                print("Type 'exit' to quit, 'history' to see conversation, 'stats' for statistics")
                print("=" * 60)
                continue
            
            elif not user_input:
                print("Please enter a message or 'exit' to quit.")
                continue
            
            # Send message and get response
            result = chat.send_message(user_input)
            
            # Display response
            print(f"\n🤖 Assistant: {result['response']}")
            
            # Show processing info
            metadata = result['metadata']
            if 'processing_time' in metadata:
                print(f"⏱️  ({metadata['processing_time']:.2f}s)")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Data Analytics Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()
