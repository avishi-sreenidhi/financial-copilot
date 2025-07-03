
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import pandas as pd
import numpy as np
import json
import os
import ast
from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def safe_parse_action_input(action_input):
    """Safely parse action input from various formats (dict, JSON string, Python dict string)"""
    # If already a dict, return as is
    if isinstance(action_input, dict):
        return action_input
    # Try to parse as dict from string
    try:
        return ast.literal_eval(action_input)
    except Exception:
        # Optionally, try to parse as JSON
        try:
            return json.loads(action_input)
        except Exception:
            return {}

# Enhanced Python code execution with pre-loaded data analytics libraries
@tool
def execute_python_code(code: str) -> str:
    """Executes Python code and returns the output. Use this for data analysis, statistics, calculations, and data transformations.
    Pre-loaded libraries: pandas (pd), numpy (np), json, os
    Sample dataset is available at 'src/data/sample.csv'
    """
    try:
        # Pre-load common data science libraries and sample data
        global_vars = {
            'pd': pd, 
            'np': np, 
            'json': json,
            'os': os,
            'df': None  # Will be loaded when needed
        }
        
        # Load sample data if available
        sample_path = 'src/data/sample.csv'
        if os.path.exists(sample_path):
            global_vars['sample_df'] = pd.read_csv(sample_path)
        
        local_vars = {}
        exec(code, global_vars, local_vars)
        
        # Return result if explicitly set, otherwise return success message
        result = local_vars.get('result', 'Code executed successfully')
        return str(result)
    except Exception as e:
        return f"Error executing code: {e}"

@tool
def load_dataset(file_path: str) -> str:
    """Loads a dataset from the specified file path and returns basic information about it.
    Supports CSV files. Returns column names, shape, and first few rows.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            info = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'head': df.head().to_dict('records'),
                'description': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else "No numeric columns for description"
            }
            return json.dumps(info, indent=2, default=str)
        else:
            return "Unsupported file format. Currently supports CSV files."
    except Exception as e:
        return f"Error loading dataset: {e}"

class PythonIDEAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [execute_python_code, load_dataset]
        # Create the LangGraph React agent
        self.agent = create_react_agent(self.llm, self.tools)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Python code based on user requests
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with Python code execution results
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
        updated_state["current_agent"] = "python"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[PythonIDEAgent] Processing query: {query}")
        try:
            # LangGraph agents expect messages in a specific format
            result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
            
            # Extract the final response from the agent - handle both dict and AIMessage format
            if isinstance(result, dict) and "messages" in result:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    final_message = final_message.content
                elif isinstance(final_message, dict) and 'content' in final_message:
                    final_message = final_message['content']
                else:
                    final_message = str(final_message)
            else:
                # Handle direct AIMessage response
                if hasattr(result, 'content'):
                    final_message = result.content
                else:
                    final_message = str(result)
            
            print(f"[PythonIDEAgent] Result: {final_message[:200]}...")
            
            # Update state with Python code execution results
            updated_state["agent_outputs"]["python"] = {
                "status": "completed",
                "result": final_message,
                "reasoning": "Completed Python code execution"
            }
            
            return updated_state
            
        except Exception as e:
            print(f"[PythonIDEAgent] Error: {e}")
            error_message = f"Error in Python IDE agent: {e}"
            
            # Update state with error information
            updated_state["agent_outputs"]["python"] = {
                "status": "error",
                "result": error_message,
                "error": str(e)
            }
            
            return updated_state


