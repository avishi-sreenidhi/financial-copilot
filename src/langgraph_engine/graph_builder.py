from langgraph.graph import StateGraph, END
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

# Load environment variables first
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.router_agent import RouterAgent
from agents.python_ide_agent import PythonIDEAgent
from agents.charting_agent import ChartingAgent
from agents.data_search_agent import DataSearchAgent
from agents.pandas_agent import PandasAgent, df_manager
from agents.memory_agent import ConversationMemoryAgent, ChatResponseFormatter
from agents.query_context_agent import QueryContextAgent
from langchain_openai import ChatOpenAI

# Define the state for our graph using TypedDict for more structure
class DataAnalyticsState(TypedDict):
    """Structured state for the data analytics multi-agent system with chat capabilities"""
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]
    query: str  # The original user query
    current_agent: str  # Currently active agent
    agent_outputs: Dict[str, Any]  # Results from each agent
    next_agent: str  # Next agent to call
    dataframe_info: Dict[str, Any]  # Info about loaded dataframes
    has_data: bool  # Whether data has been loaded
    final_result: str  # Final formatted result
    metadata: Dict[str, Any]  # Additional metadata about the execution and conversation
    iteration_count: int  # Count of iterations
    # Chat-specific fields
    chat_response: Dict[str, Any]  # Chat-formatted response for UI
    session_id: str  # Session identifier for conversation tracking
    conversation_summary: str  # Summary of the conversation so far

# Initialize LLM
try:
    llm = ChatOpenAI(temperature=0)
    if llm is None:
        raise ValueError("LLM initialization returned None")
except Exception as e:
    # For testing purposes, provide a fallback
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Using mock LLM for testing.")
        # Create a mock LLM for testing - this should be replaced in production
        class MockLLM:
            def invoke(self, prompt):
                return type('MockResponse', (), {'content': 'Mock response for testing'})()
        llm = MockLLM()
    else:
        print(f"Error initializing LLM: {e}")
        raise e

# Global memory manager instance
_global_memory_agent = None

def get_memory_agent():
    """Get the global memory agent instance"""
    return _global_memory_agent

def clear_memory_agent(session_id: str = None):
    """Clear memory for a specific session or all sessions"""
    global _global_memory_agent
    if _global_memory_agent:
        if session_id:
            _global_memory_agent.clear_session_memory(session_id)
        else:
            _global_memory_agent.clear_all_memory()

def build_agent_graph():
    """Create a more sophisticated agent graph with coordinator-based routing and conversation memory"""
    global _global_memory_agent
    workflow = StateGraph(DataAnalyticsState)

    # Validate LLM before proceeding
    if llm is None:
        raise ValueError("LLM is not properly initialized. Cannot create agent graph.")

    # Initialize agents
    router_agent = RouterAgent(llm)
    python_ide_agent = PythonIDEAgent(llm)
    charting_agent = ChartingAgent(llm)
    data_search_agent = DataSearchAgent(llm)
    pandas_agent = PandasAgent(llm)
    memory_agent = ConversationMemoryAgent(llm)
    _global_memory_agent = memory_agent  # Store globally for memory management
    query_context_agent = QueryContextAgent(llm)

    # Add nodes for each agent and tools
    workflow.add_node("router", router_agent.invoke)
    workflow.add_node("python", python_ide_agent.invoke)
    workflow.add_node("chart", charting_agent.invoke)
    workflow.add_node("search", data_search_agent.invoke)
    workflow.add_node("pandas", pandas_agent.invoke)
    workflow.add_node("memory", memory_agent.invoke)
    workflow.add_node("query_context", query_context_agent.invoke)
    
    # New coordinator node for orchestrating the workflow
    workflow.add_node("coordinator", coordinator_agent)
    
    # Chat response formatting node
    workflow.add_node("chat_formatter", ChatResponseFormatter.format_chat_response)
    
    # New tool execution node
    workflow.add_node("tools", tool_executor)

    # Set router as the entry point
    workflow.set_entry_point("router")

    # From router, always go to query_context first for query analysis
    workflow.add_edge("router", "query_context")
    
    # From query_context, go to memory for conversation context
    workflow.add_edge("query_context", "memory")
    
    # From memory, route to the appropriate agent based on router's decision
    workflow.add_conditional_edges(
        "memory",
        lambda x: x.get("next_agent") or x.get("agent_outputs", {}).get("router", {}).get("next_agent", "pandas"),
        {
            "python": "python",
            "chart": "chart", 
            "search": "search",
            "pandas": "pandas",
            "coordinator": "coordinator",
        },
    )

    # Each specialized agent can go to the coordinator or tools
    for agent in ["python", "chart", "search", "pandas"]:
        workflow.add_conditional_edges(
            agent,
            agent_router,
            {
                "tools": "tools",
                "coordinator": "coordinator",
                "end": END
            }
        )
    
    # Tools always return to coordinator
    workflow.add_edge("tools", "coordinator")
    
    # Coordinator decides next step
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_router,
        {
            "python": "python",
            "chart": "chart", 
            "search": "search",
            "pandas": "pandas",
            "tools": "tools",
            "chat_formatter": "chat_formatter",
            "end": END
        }
    )
    
    # Chat formatter always ends the workflow
    workflow.add_edge("chat_formatter", END)
    
    # Remove the final response formatting edge since we don't have tone agent
    # workflow.add_edge("tone", END)

    return workflow.compile()

def agent_router(state: DataAnalyticsState) -> str:
    """Route from agents to coordinator, tools, or final response"""
    # Check last message for tool requests
    messages = state.get("messages", [])
    if not messages:
        return "coordinator"
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # Check if agent needs tools (for future implementation)
    if "NEED_TOOL:" in content or "NEED_SEARCH:" in content:
        return "tools"
        
    # Check if we have a final result ready
    agent_outputs = state.get("agent_outputs", {})
    current_agent = state.get("current_agent", "")
    
    if current_agent in agent_outputs and agent_outputs[current_agent].get("status") == "completed":
        return "coordinator"  # Route to coordinator for final processing
    
    # Default to coordinator for next steps
    return "coordinator"

def coordinator_agent(state: DataAnalyticsState) -> DataAnalyticsState:
    """Coordinator agent that orchestrates the workflow between specialized agents"""
    new_state = state.copy()
    
    # Get the latest from each agent
    agent_outputs = new_state.get("agent_outputs", {})
    completed_agents = [agent for agent, output in agent_outputs.items() 
                       if output.get("status") == "completed"]
    
    # If we have results from all needed agents, go to chat formatting
    if len(completed_agents) > 0:
        # Prepare final result from agent outputs (keep basic format for now)
        result = ""
        for agent, output in agent_outputs.items():
            if agent not in ["router", "memory"] and output.get("status") == "completed":
                result += f"From {agent}: {output.get('result', '')}\n\n"
                
        new_state["final_result"] = result
        new_state["next_agent"] = "chat_formatter"  # Route to chat formatting before end
    else:
        # Determine which agent still needs to process
        if "router" in agent_outputs and "next_agent" in agent_outputs["router"]:
            new_state["next_agent"] = agent_outputs["router"]["next_agent"]
        else:
            # Default to pandas agent if we don't know what to do
            new_state["next_agent"] = "pandas"
    
    new_state["current_agent"] = "coordinator"
    new_state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    return new_state

def coordinator_router(state: DataAnalyticsState) -> str:
    """Route from coordinator to next agent"""
    next_agent = state.get("next_agent", "pandas")
    
    # If iteration count is too high, force end
    if state.get("iteration_count", 0) > 5:
        return "chat_formatter"  # Format response before ending
    
    # If next_agent is "end", go to chat formatting first
    if next_agent == "end":
        return "chat_formatter"
    
    # If next_agent is "chat_formatter", proceed to formatting
    if next_agent == "chat_formatter":
        return "chat_formatter"
    
    # Route to the specific agent
    return next_agent

def tool_executor(state: DataAnalyticsState) -> DataAnalyticsState:
    """Execute tools based on agent requests"""
    new_state = state.copy()
    
    # Get last message and current agent
    messages = state.get("messages", [])
    if not messages:
        return state
        
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    current_agent = state.get("current_agent", "")
    
    # For now, just add a placeholder tool result message
    # This would be expanded with actual tool functionality
    tool_message = AIMessage(content=f"Tool execution result for request: {content[:50]}...")
    new_state["messages"] = state.get("messages", []) + [tool_message]
    
    return new_state


