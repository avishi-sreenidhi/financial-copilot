from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import ast
from typing import Dict, Any, TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class RouterAgent:
    def __init__(self, llm):
        if llm is None:
            raise ValueError("RouterAgent requires a valid LLM instance")
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """You are an intelligent router for a data analytics system. Analyze the user's intent and route to the most appropriate agent.

Available agents:
- 'pandas': Data analysis, statistics, descriptive stats, dataset exploration, data summaries, correlations, missing values, general data questions
- 'python': Explicit Python programming, code writing, custom algorithms, script development  
- 'chart': Data visualization requests (plots, charts, graphs, visualizations)
- 'search': Searching or filtering for specific data points, finding specific records

Query: {input}

Based on the user's intent, which agent should handle this query? Respond with only the agent name.
"""
        )

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently route the user's query to the appropriate agent using LLM
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with routing decision
        """
        # Create a copy of the state to modify and return
        updated_state = state.copy()
        
        # Extract query from either state or messages
        query = state.get("query", "")
        if not query and state.get("messages"):
            # Get the last user message if query isn't directly available
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
                elif hasattr(msg, 'type') and msg.type == 'human':
                    query = msg.content
                    break
                    
        # Default to empty string if we still don't have a query
        query = query or ""
        
        # Set current agent in state
        updated_state["current_agent"] = "router"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[RouterAgent] Routing query: {query}")
        
        try:
            # Use LLM for intelligent routing
            routing_prompt = f"""You are a router agent for a data analytics system. Analyze the user's query and determine which agent should handle it.

Query: "{query}"

Route to:
- 'pandas': For data analysis, statistics, descriptive stats, exploring datasets, data summaries, correlations, missing values, data insights, general data questions
- 'python': ONLY for explicit Python programming requests, code writing, custom algorithms
- 'chart': For visualization requests (plots, charts, graphs) 
- 'search': For searching/filtering specific data points

For the query "{query}", which agent is most appropriate?

Respond with ONLY the agent name: pandas, python, chart, or search"""

            # Get LLM response
            llm_response = self.llm.invoke(routing_prompt)
            
            # Extract the agent name from response
            if hasattr(llm_response, 'content'):
                next_agent = llm_response.content.strip().lower()
            else:
                next_agent = str(llm_response).strip().lower()
            
            # Validate the agent name
            valid_agents = ['pandas', 'python', 'chart', 'search']
            if next_agent not in valid_agents:
                # Fallback logic for invalid responses
                query_lower = query.lower()
                if any(kw in query_lower for kw in ['plot', 'chart', 'visualize', 'graph']):
                    next_agent = 'chart'
                elif any(kw in query_lower for kw in ['python code', 'write code', 'script']):
                    next_agent = 'python'
                elif any(kw in query_lower for kw in ['search', 'find', 'filter']):
                    next_agent = 'search'
                else:
                    next_agent = 'pandas'  # Default to pandas for data analysis
            
            print(f"[RouterAgent] LLM routed to: {next_agent}")
            
        except Exception as e:
            print(f"[RouterAgent] LLM routing error: {e}")
            # Fallback to simple rule-based routing
            query_lower = query.lower()
            if any(kw in query_lower for kw in ['plot', 'chart', 'visualize', 'graph']):
                next_agent = 'chart'
            elif any(kw in query_lower for kw in ['python code', 'write code', 'script', 'programming']):
                next_agent = 'python'
            elif any(kw in query_lower for kw in ['search', 'find specific', 'filter data']):
                next_agent = 'search'
            else:
                next_agent = 'pandas'  # Default to pandas for most data queries
            
            print(f"[RouterAgent] Fallback routed to: {next_agent}")
        
        # Update state with routing decision
        updated_state["agent_outputs"]["router"] = {
            "next_agent": next_agent,
            "status": "completed",
            "result": f"Intelligently routed query to {next_agent} agent",
            "reasoning": f"LLM determined {next_agent} is most appropriate for query: '{query}'"
        }
        
        # Set next_agent in state for coordinator
        updated_state["next_agent"] = next_agent
        
        return updated_state
    
def safe_parse_action_input(action_input):
    # If already a dict, return as is
    if isinstance(action_input, dict):
        return action_input
    # Try to parse as dict from string
    try:
        return ast.literal_eval(action_input)
    except Exception:
        # Optionally, try to parse as JSON
        import json
        try:
            return json.loads(action_input)
        except Exception:
            return {}

# In your tool handler:
# parsed_input = safe_parse_action_input(action_input)
# Now use parsed_input['chart_type'], etc.



