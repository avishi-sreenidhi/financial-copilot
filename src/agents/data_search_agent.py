from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import pandas as pd
import numpy as np
import json
import os
import ast
from datetime import datetime
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

@tool
def search_data(query: str) -> str:
    """Searches the sample.csv dataset for information relevant to the query.
    Supports text search, numerical filtering, and date range queries.
    Returns relevant rows or a summary of the search results.
    """
    try:
        df = pd.read_csv("src/data/sample.csv")
        
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Enhanced search logic
        results = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(query_lower, case=False, na=False).any(), axis=1)]
        
        if not results.empty:
            # Limit results to at most 20 rows to save tokens
            max_rows = min(20, len(results))
            limited_results = results.head(max_rows)
            
            # For wide dataframes, limit columns to save tokens
            limited_results_dict = limited_results.to_dict('records')
            
            # If results are large, truncate long string values
            if len(limited_results_dict) > 10 or len(limited_results.columns) > 10:
                for row in limited_results_dict:
                    for key, value in row.items():
                        if isinstance(value, str) and len(value) > 50:
                            row[key] = value[:50] + "..."
            
            return json.dumps({
                "found_rows": len(results),
                "displayed_rows": len(limited_results),
                "data": limited_results_dict,
                "summary": f"Found {len(results)} rows matching '{query}'" + 
                          (f" (showing first {max_rows})" if len(results) > max_rows else "")
            }, indent=2, default=str)
        else:
            return json.dumps({
                "found_rows": 0,
                "data": [],
                "summary": f"No data found matching '{query}'"
            })
    except Exception as e:
        return f"Error searching data: {e}"

@tool
def filter_data(column: str, operator: str, value: str) -> str:
    """Filters the dataset based on specified criteria.
    Operators: '>', '<', '>=', '<=', '==', '!=', 'contains'
    Example: filter_data('revenue', '>', '1000')
    """
    try:
        df = pd.read_csv("src/data/sample.csv")
        
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
        
        # Convert value to appropriate type
        if df[column].dtype in ['int64', 'float64']:
            value = float(value)
        
        # Apply filter based on operator
        if operator == '>':
            filtered_df = df[df[column] > value]
        elif operator == '<':
            filtered_df = df[df[column] < value]
        elif operator == '>=':
            filtered_df = df[df[column] >= value]
        elif operator == '<=':
            filtered_df = df[df[column] <= value]
        elif operator == '==':
            filtered_df = df[df[column] == value]
        elif operator == '!=':
            filtered_df = df[df[column] != value]
        elif operator == 'contains':
            filtered_df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            return f"Unsupported operator: {operator}. Use: >, <, >=, <=, ==, !=, contains"
        
        # Limit results to at most 20 rows to save tokens
        max_rows = min(15, len(filtered_df))
        limited_results = filtered_df.head(max_rows)
        
        # For wide dataframes, limit columns to save tokens
        limited_results_dict = limited_results.to_dict('records')
        
        # If results are large, truncate long string values
        if len(limited_results_dict) > 10 or len(limited_results.columns) > 10:
            for row in limited_results_dict:
                for key, value in row.items():
                    if isinstance(value, str) and len(value) > 50:
                        row[key] = value[:50] + "..."
        
        return json.dumps({
            "filtered_rows": len(filtered_df),
            "displayed_rows": len(limited_results),
            "data": limited_results_dict,
            "summary": f"Found {len(filtered_df)} rows where {column} {operator} {value}" +
                      (f" (showing first {max_rows})" if len(filtered_df) > max_rows else "")
        }, indent=2, default=str)
        
    except Exception as e:
        return f"Error filtering data: {e}"

@tool 
def get_data_summary() -> str:
    """Returns a comprehensive summary of the sample dataset including statistics and structure."""
    try:
        df = pd.read_csv("src/data/sample.csv")
        
        summary = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},  # Convert dtypes to strings
            "missing_values": df.isnull().sum().to_dict(),
        }
        
        # Add sample data with truncated strings for token efficiency
        sample_data = df.head(3).to_dict('records')
        for row in sample_data:
            for key, value in row.items():
                if isinstance(value, str) and len(value) > 50:
                    row[key] = value[:50] + "..."
        summary["sample_data"] = sample_data
        
        # Add statistics for numeric columns (limited to conserve tokens)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Limit to 5 numeric columns for token efficiency
            if len(numeric_cols) > 5:
                numeric_cols = numeric_cols[:5]
            summary["statistics"] = df[numeric_cols].describe().to_dict()
        
        return json.dumps(summary, indent=2, default=str)
        
    except Exception as e:
        return f"Error getting data summary: {e}"

class DataSearchAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [search_data, filter_data, get_data_summary]
        self.prompt = PromptTemplate.from_template(
            """You are an advanced Data Search and Query agent. Your goal is to help users find, filter, and explore structured datasets efficiently.

            Available tools:
            - search_data: Search for text patterns across all columns in the dataset
            - filter_data: Apply specific filters based on column values and operators
            - get_data_summary: Get comprehensive overview of dataset structure and statistics
            
            Dataset information:
            - Primary dataset: sample.csv (located at src/data/sample.csv)
            - Contains columns: date, revenue, expenses
            - Sample data with financial information over time
            
            Capabilities:
            - Text-based search across all columns
            - Numerical filtering with comparison operators (>, <, >=, <=, ==, !=)
            - String pattern matching with 'contains' operator
            - Dataset structure analysis and statistics
            - Missing value detection
            
            Best practices:
            - Use get_data_summary first for unknown datasets
            - Use appropriate operators for different data types
            - Provide clear, structured results with context

            IMPORTANT: When using tools, always format your Action Input as valid JSON. For example:
            Action Input: {{"query": "revenue"}}
            Action Input: {{"column": "revenue", "operator": ">", "value": "1000"}}

            TOOLS:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action (MUST be valid JSON format)
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought: {agent_scratchpad}
            """
        )
        # Create the LangGraph React agent
        self.agent = create_react_agent(self.llm, self.tools)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search and query structured datasets based on user requests with query context awareness
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with search results
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
        
        # Use expanded query if available from QueryContextAgent
        expanded_query = state.get("expanded_query", query)
        query_context = state.get("query_context", {})
        
        # Default to empty string if we still don't have a query
        query = query or ""
        expanded_query = expanded_query or query
        
        # Set current agent in state
        updated_state["current_agent"] = "search"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[DataSearchAgent] Processing query: {query}")
        if expanded_query != query:
            print(f"[DataSearchAgent] Using expanded query: {expanded_query}")
        if query_context:
            print(f"[DataSearchAgent] Query context available: {query_context.get('query_type', 'unknown')}")
        
        try:
            # Use the expanded query with context hints for better search
            search_query = self._enhance_search_with_context(expanded_query, query_context)
            print(f"[DataSearchAgent] Enhanced search query: {search_query}")
            
            # LangGraph agents expect messages in a specific format
            result = self.agent.invoke({"messages": [{"role": "user", "content": search_query}]})
            
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
            
            print(f"[DataSearchAgent] Result: {final_message[:200]}...")
            
            # Update state with search results
            updated_state["agent_outputs"]["search"] = {
                "status": "completed",
                "result": final_message,
                "reasoning": "Completed data search with query context enhancement"
            }
            
            return updated_state
            
        except Exception as e:
            print(f"[DataSearchAgent] Error: {e}")
            error_message = f"Error in data search agent: {e}"
            
            # Update state with error information
            updated_state["agent_outputs"]["search"] = {
                "status": "error",
                "result": error_message,
                "error": str(e)
            }
            
            return updated_state

    def _enhance_search_with_context(self, query: str, query_context: Dict[str, Any]) -> str:
        """Enhance the search query using context information"""
        enhanced_query = query
        
        # Use column mapping to enhance search
        column_mapping = query_context.get("column_mapping", {})
        for abbrev, full_column in column_mapping.items():
            enhanced_query = enhanced_query.replace(abbrev, full_column)
        
        # Add context hints
        context_hints = query_context.get("context_hints", [])
        relevant_columns = query_context.get("relevant_columns", [])
        query_type = query_context.get("query_type", "general")
        
        # Enhance based on query type
        if query_type == "ranking_max" and relevant_columns:
            enhanced_query += f". Focus on finding the maximum value in columns: {relevant_columns}"
        elif query_type == "ranking_min" and relevant_columns:
            enhanced_query += f". Focus on finding the minimum value in columns: {relevant_columns}"
        elif query_type == "comparison" and relevant_columns:
            enhanced_query += f". Compare values in columns: {relevant_columns}"
        elif query_type == "aggregation" and relevant_columns:
            enhanced_query += f". Calculate aggregations for columns: {relevant_columns}"
        
        # Add relevant columns hint
        if relevant_columns:
            enhanced_query += f". Pay special attention to these columns: {', '.join(relevant_columns)}"
        
        return enhanced_query


