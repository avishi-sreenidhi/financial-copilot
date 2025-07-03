from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any, List
import json
import time
import time

class ConversationMemoryAgent:
    """
    Agent responsible for maintaining conversation context and memory.
    Provides ChatGPT-like conversational experience with context awareness.
    """
    
    def __init__(self, llm):
        if llm is None:
            raise ValueError("ConversationMemoryAgent requires a valid LLM instance")
        self.llm = llm
        self.session_memory = {}  # Store session-specific memory
        self.global_cache = {}    # Store global conversation cache
        self.prompt = PromptTemplate.from_template(
            """You are a conversation memory manager for a data analytics assistant. Your role is to:

1. Maintain context from previous messages in the conversation
2. Provide relevant context to help other agents understand the conversation flow
3. Summarize conversation history when needed
4. Track user preferences and data context across the session

Current conversation history:
{conversation_history}

Latest user message: {latest_message}

Previous agent responses:
{agent_responses}

Based on this context, provide:
1. A brief summary of the conversation so far
2. Relevant context that should be considered for the current query
3. Any user preferences or patterns you've noticed
4. Whether this query relates to previous queries in the session

Format your response as JSON with these keys:
- "conversation_summary": Brief summary of the conversation
- "relevant_context": Context relevant to current query
- "user_preferences": Any preferences or patterns noticed
- "query_relationship": How this query relates to previous ones
- "suggested_context": Additional context to pass to the next agent

Respond only with valid JSON."""
        )

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation history and provide context for better responses
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with conversation context and memory
        """
        updated_state = state.copy()
        
        # Set current agent in state
        updated_state["current_agent"] = "memory"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        # Extract conversation components
        messages = state.get("messages", [])
        latest_query = state.get("query", "")
        agent_outputs = state.get("agent_outputs", {})
        session_id = state.get("session_id", "default")
        
        # Get session-specific memory
        session_memory = self._get_session_memory(session_id)
        
        try:
            # Build conversation history
            conversation_history = self._build_conversation_history(messages)
            
            # Build agent responses summary
            agent_responses = self._build_agent_responses_summary(agent_outputs)
            
            # For simple queries or short conversations, provide basic context
            if len(messages) <= 2 or len(latest_query.split()) <= 5:
                memory_context = {
                    "conversation_summary": "New conversation or simple query",
                    "relevant_context": "No significant previous context",
                    "user_preferences": "None identified yet",
                    "query_relationship": "Independent query",
                    "suggested_context": "Handle as a fresh query"
                }
            else:
                # Use LLM for complex conversation analysis
                formatted_prompt = self.prompt.format(
                    conversation_history=conversation_history,
                    latest_message=latest_query,
                    agent_responses=agent_responses
                )
                
                llm_response = self.llm.invoke(formatted_prompt)
                content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                
                try:
                    memory_context = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    memory_context = {
                        "conversation_summary": "Ongoing conversation",
                        "relevant_context": "Previous data analysis interaction",
                        "user_preferences": "Analysis-focused",
                        "query_relationship": "Continuation of data exploration",
                        "suggested_context": content[:200] + "..."
                    }
            
            # Store memory context in state
            updated_state["agent_outputs"]["memory"] = {
                "status": "completed",
                "result": memory_context,
                "reasoning": "Analyzed conversation context for better response continuity"
            }
            
            # Store in session memory for persistence
            self._store_session_memory(session_id, {
                "last_context": memory_context,
                "conversation_length": len(messages),
                "last_query": latest_query
            })
            
            # Add conversation context to metadata for other agents to use
            if "metadata" not in updated_state:
                updated_state["metadata"] = {}
            
            updated_state["metadata"]["conversation_context"] = memory_context
            updated_state["metadata"]["conversation_length"] = len(messages)
            updated_state["metadata"]["has_previous_context"] = len(messages) > 2
            updated_state["metadata"]["session_id"] = session_id
            
            print(f"[ConversationMemoryAgent] Processed {len(messages)} messages for session {session_id}")
            print(f"[ConversationMemoryAgent] Context: {memory_context['conversation_summary']}")
            
            return updated_state
            
        except Exception as e:
            print(f"[ConversationMemoryAgent] Error: {e}")
            
            # Provide basic fallback context
            fallback_context = {
                "conversation_summary": "Error in context analysis",
                "relevant_context": "Limited context available",
                "user_preferences": "Unknown",
                "query_relationship": "Independent",
                "suggested_context": "Handle query independently"
            }
            
            updated_state["agent_outputs"]["memory"] = {
                "status": "error",
                "result": fallback_context,
                "error": str(e)
            }
            
            if "metadata" not in updated_state:
                updated_state["metadata"] = {}
            updated_state["metadata"]["conversation_context"] = fallback_context
            
            return updated_state
    
    def _build_conversation_history(self, messages: List) -> str:
        """Build a readable conversation history from messages"""
        if not messages:
            return "No previous conversation"
        
        history = []
        for i, msg in enumerate(messages[-10:]):  # Last 10 messages to avoid token limits
            if isinstance(msg, HumanMessage):
                history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"Assistant: {msg.content}")
            elif hasattr(msg, 'type'):
                if msg.type == 'human':
                    history.append(f"User: {msg.content}")
                elif msg.type == 'ai':
                    history.append(f"Assistant: {msg.content}")
        
        return "\n".join(history) if history else "No conversation history"
    
    def _build_agent_responses_summary(self, agent_outputs: Dict[str, Any]) -> str:
        """Build a summary of previous agent responses"""
        if not agent_outputs:
            return "No previous agent responses"
        
        summary = []
        for agent, output in agent_outputs.items():
            if agent != "memory" and output.get("status") == "completed":
                result = output.get("result", "")
                if result:
                    summary.append(f"{agent}: {str(result)[:100]}...")
        
        return "\n".join(summary) if summary else "No previous responses"
    
    def clear_session_memory(self, session_id: str = None):
        """Clear memory for a specific session or all sessions"""
        if session_id:
            if session_id in self.session_memory:
                del self.session_memory[session_id]
                print(f"[ConversationMemoryAgent] Cleared memory for session: {session_id}")
        else:
            self.session_memory.clear()
            print("[ConversationMemoryAgent] Cleared all session memory")
    
    def clear_global_cache(self):
        """Clear global conversation cache"""
        self.global_cache.clear()
        print("[ConversationMemoryAgent] Cleared global cache")
    
    def clear_all_memory(self):
        """Clear all memory and cache"""
        self.session_memory.clear()
        self.global_cache.clear()
        print("[ConversationMemoryAgent] Cleared all memory and cache")
    
    def get_session_info(self, session_id: str = None):
        """Get information about stored sessions"""
        if session_id:
            return self.session_memory.get(session_id, {})
        else:
            return {
                "total_sessions": len(self.session_memory),
                "sessions": list(self.session_memory.keys()),
                "global_cache_size": len(self.global_cache)
            }
    
    def _get_session_memory(self, session_id: str):
        """Get or create session-specific memory"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "conversation_history": [],
                "context_summary": "",
                "user_preferences": {},
                "created_at": time.time()
            }
        return self.session_memory[session_id]
    
    def _store_session_memory(self, session_id: str, memory_data: dict):
        """Store memory data for a specific session"""
        session_mem = self._get_session_memory(session_id)
        session_mem.update(memory_data)
        session_mem["updated_at"] = time.time()

class ChatResponseFormatter:
    """
    Formats agent responses into chat-like messages for a ChatGPT-style interface
    """
    
    @staticmethod
    def format_chat_response(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the final response in a chat-friendly way
        
        Args:
            state: Current state with agent outputs
            
        Returns:
            Updated state with formatted chat response
        """
        updated_state = state.copy()
        
        agent_outputs = state.get("agent_outputs", {})
        conversation_context = state.get("metadata", {}).get("conversation_context", {})
        
        # Determine the primary responding agent (not router, memory, or query_context)
        primary_agent = None
        primary_response = ""
        
        for agent, output in agent_outputs.items():
            if agent not in ["router", "memory", "query_context"] and output.get("status") == "completed":
                primary_agent = agent
                primary_response = output.get("result", "")
                break
        
        # Ensure primary_response is a string (safety check)
        if isinstance(primary_response, dict):
            # If it's a dict, convert to a meaningful string representation
            primary_response = str(primary_response)
        elif primary_response is None:
            primary_response = ""
        
        # Format response based on agent type and context
        if primary_agent == "pandas":
            formatted_response = ChatResponseFormatter._format_pandas_response(
                primary_response, conversation_context
            )
        elif primary_agent == "python":
            formatted_response = ChatResponseFormatter._format_python_response(
                primary_response, conversation_context
            )
        elif primary_agent == "chart":
            formatted_response = ChatResponseFormatter._format_chart_response(
                primary_response, conversation_context
            )
        elif primary_agent == "search":
            formatted_response = ChatResponseFormatter._format_search_response(
                primary_response, conversation_context
            )
        else:
            formatted_response = primary_response or "I'm here to help with your data analysis. How can I assist you?"
        
        # Add conversation continuity
        relationship = conversation_context.get("query_relationship", "")
        if "continuation" in relationship.lower() or "follow-up" in relationship.lower():
            formatted_response = "Continuing from our previous analysis...\n\n" + formatted_response
        
        # Store the formatted response
        updated_state["final_result"] = formatted_response
        updated_state["chat_response"] = {
            "message": formatted_response,
            "agent": primary_agent,
            "context_aware": len(state.get("messages", [])) > 2,
            "conversation_summary": conversation_context.get("conversation_summary", "")
        }
        
        return updated_state
    
    @staticmethod
    def _format_pandas_response(response: str, context: Dict[str, Any]) -> str:
        """Format pandas agent response for chat"""
        if "Hello!" in response or "upload" in response.lower():
            return response  # Keep greeting responses as-is
        
        # Add context-aware formatting
        if "No dataframe loaded" in response:
            return "I don't see any data loaded yet. Would you like to upload a CSV or Excel file so I can help you analyze it?"
        
        return response
    
    @staticmethod
    def _format_python_response(response: str, context: Dict[str, Any]) -> str:
        """Format python agent response for chat"""
        if "```" in response:
            return f"I've executed your Python code:\n\n{response}\n\nIs there anything else you'd like me to calculate or analyze?"
        return response
    
    @staticmethod
    def _format_chart_response(response: str, context: Dict[str, Any]) -> str:
        """Format chart agent response for chat"""
        if "Chart generated successfully" in response:
            return "I've created your visualization! The chart has been generated based on your data. Would you like me to create any other charts or modify this one?"
        return response
    
    @staticmethod
    def _format_search_response(response: str, context: Dict[str, Any]) -> str:
        """Format search agent response for chat"""
        if "rows found" in response.lower() or "results" in response.lower():
            return f"Here's what I found in your data:\n\n{response}\n\nWould you like me to search for anything else or analyze these results further?"
        return response
