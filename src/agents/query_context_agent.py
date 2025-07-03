from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd

# Import the shared DataFrameManager
try:
    from .pandas_agent import df_manager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pandas_agent import df_manager

class QueryContextAgent:
    """
    Agent that provides context understanding for user queries by:
    1. Analyzing the current dataset structure and content
    2. Expanding abbreviations and domain-specific terms
    3. Suggesting better query formulations
    4. Providing column mapping and context hints
    """
    
    def __init__(self, llm):
        if llm is None:
            raise ValueError("QueryContextAgent requires a valid LLM instance")
        self.llm = llm
        
        # Common abbreviations and mappings that can be expanded based on context
        self.common_abbreviations = {
            'rev': ['revenue', 'sales'],
            'exp': ['expenses', 'expenditure'],
            'inc': ['income'],
            'roi': ['return_on_investment'],
            'cogs': ['cost_of_goods_sold'],
            'ebitda': ['earnings_before_interest_taxes_depreciation_amortization'],
            'np': ['net_profit'],
            'gp': ['gross_profit'],
            'cf': ['cash_flow'],
            'bal': ['balance', 'account_balance'],
            'eq': ['equity'],
            'liab': ['liabilities'],
            'fin': ['financial', 'finance'],
            'amt': ['amount'],
            'qty': ['quantity', 'units'],
            'pct': ['percent', 'percentage'],
            'avg': ['average', 'mean'],
            'min': ['minimum'],
            'max': ['maximum'],
            'std': ['standard_deviation'],
            'id': ['identifier', 'transaction_id'],
            'num': ['number'],
            'vol': ['volume', 'transaction_volume'],
            'yr': ['year'],
            'mth': ['month'],
            'qtr': ['quarter'],
            'acct': ['account'],
            'txn': ['transaction'],
            'int': ['interest'],
            'inv': ['investment'],
            'cap': ['capital'],
            'apr': ['annual_percentage_rate'],
            'apy': ['annual_percentage_yield'],
            'fx': ['foreign_exchange'],
            'usd': ['us_dollars'],
            'eur': ['euro'],
        }

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user query and provide context enrichment
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with query context analysis
        """
        # Create a copy of the state to modify and return
        updated_state = state.copy()
        
        # Extract query from either state or messages
        query = state.get("query", "")
        if not query and state.get("messages"):
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
                elif hasattr(msg, 'type') and msg.type == 'human':
                    query = msg.content
                    break
        
        query = query or ""
        
        # Set current agent in state
        updated_state["current_agent"] = "query_context"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[QueryContextAgent] Processing query: {query}")
        
        try:
            # Get current dataframe for context
            current_df = df_manager.get_current_dataframe()
            
            if current_df is None:
                result = {
                    "original_query": query,
                    "expanded_query": query,
                    "context_hints": [],
                    "column_suggestions": [],
                    "reasoning": "No dataset loaded for context analysis"
                }
            else:
                result = self._analyze_query_context(query, current_df)
            
            # Update state with context analysis
            updated_state["agent_outputs"]["query_context"] = {
                "status": "completed",
                "result": result,
                "reasoning": "Completed query context analysis"
            }
            
            # Also add the expanded query to the state for other agents to use
            updated_state["expanded_query"] = result.get("expanded_query", query)
            updated_state["query_context"] = result
            
            return updated_state
            
        except Exception as e:
            print(f"[QueryContextAgent] Error: {e}")
            error_result = {
                "original_query": query,
                "expanded_query": query,
                "context_hints": [],
                "column_suggestions": [],
                "reasoning": f"Error in context analysis: {e}"
            }
            
            updated_state["agent_outputs"]["query_context"] = {
                "status": "error",
                "result": error_result,
                "error": str(e)
            }
            
            return updated_state

    def _analyze_query_context(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze query in the context of the available dataset
        """
        print(f"[QueryContextAgent] Analyzing query: {query}")
        print(f"[QueryContextAgent] Dataset shape: {df.shape}")
        print(f"[QueryContextAgent] Columns: {list(df.columns)[:10]}")
        
        # Analyze dataset structure
        columns = list(df.columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Get sample of data for better context understanding
        sample_data = {}
        for col in columns[:10]:  # Limit to first 10 columns
            try:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:
                    sample_data[col] = unique_vals.tolist()
                else:
                    sample_data[col] = unique_vals[:5].tolist() + ["..."]
            except:
                sample_data[col] = ["<analysis_error>"]
        
        # Create context analysis prompt
        context_prompt = f"""
            You are a financial query context analyzer. Your job is to interpret the user's query in the context of a financial or business dataset and provide a detailed breakdown.

            USER QUERY: "{query}"

            DATASET CONTEXT:
            - Total columns: {len(columns)}
            - Column names: {columns}
            - Numeric columns: {numeric_cols}
            - Categorical columns: {categorical_cols}
            - Sample data: {sample_data}

            TASK: Provide detailed query context analysis including:
            1. Detect any abbreviations, acronyms, or domain-specific financial terms
            2. Map abbreviations or vague terms to actual column names in the dataset (ONLY if they exist)
            3. Suggest the most relevant columns for answering the query
            4. Rewrite the query with full terms for clarity
            5. Provide high-level context to help downstream agents handle the query appropriately

            ANALYSIS GUIDELINES:
            - Use domain knowledge from finance, accounting, and business analytics
            - Look for abbreviations like rev=revenue, exp=expenses, roi=return on investment, etc.
            - Only map terms to dataset columns **if they exist**
            - If an abbreviation or term is recognized but not present in the dataset, explain its likely meaning and note that it doesn't match any column
            - If query contains keywords like "highest", "lowest", "top", "most profitable", assume a ranking or comparison intent on numeric columns
            - If query refers to categories like departments, account types, cost centers, etc., inspect categorical columns
            - If query uses vague terms (e.g., "performance", "efficiency"), try to infer intent based on dataset structure
            - Clearly state if there's a domain mismatch (e.g., asking about marketing metrics in a financial dataset)

            Respond in the following structured format:

            ABBREVIATIONS_FOUND: [list any abbreviations or unclear terms in the query]
            COLUMN_MAPPING: [map detected terms to dataset columns — only if they exist]
            RELEVANT_COLUMNS: [most relevant columns for this query, based on dataset]
            EXPANDED_QUERY: [a clearer, more explicit version of the query]
            CONTEXT_HINTS: [any helpful insights about intent, domain relevance, or mismatches]
            QUERY_TYPE: [one of: "ranking", "filtering", "comparison", "aggregation", "forecasting", "domain_mismatch"]
            DOMAIN_MATCH: [yes/no — does the query domain align with the dataset]
"""

        # Get LLM analysis
        try:
            response = self.llm.invoke(context_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            parsed_result = self._parse_context_analysis(analysis_text, query, df)
            
            print(f"[QueryContextAgent] Analysis result: {parsed_result}")
            return parsed_result
            
        except Exception as e:
            print(f"[QueryContextAgent] LLM analysis error: {e}")
            # Fallback to simple analysis
            return self._simple_context_analysis(query, df)

    def _parse_context_analysis(self, analysis_text: str, original_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse the LLM context analysis response
        """
        result = {
            "original_query": original_query,
            "expanded_query": original_query,
            "abbreviations_found": [],
            "column_mapping": {},
            "relevant_columns": [],
            "context_hints": [],
            "query_type": "general",
            "domain_match": True,  # New field
            "reasoning": "LLM-based context analysis"
        }
        
        try:
            lines = analysis_text.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == 'ABBREVIATIONS_FOUND':
                        # Parse list format
                        if value and value != 'None' and '[' in value:
                            try:
                                result["abbreviations_found"] = eval(value)
                            except:
                                result["abbreviations_found"] = [v.strip() for v in value.replace('[', '').replace(']', '').split(',') if v.strip()]
                    
                    elif key == 'COLUMN_MAPPING':
                        # Parse mapping
                        if value and value != 'None':
                            try:
                                # Handle different formats
                                if '{' in value:
                                    result["column_mapping"] = eval(value)
                                else:
                                    # Simple format: term->column
                                    mappings = {}
                                    parts = value.split(',')
                                    for part in parts:
                                        if '->' in part or '=' in part:
                                            separator = '->' if '->' in part else '='
                                            k, v = part.split(separator, 1)
                                            mappings[k.strip()] = v.strip()
                                    result["column_mapping"] = mappings
                            except:
                                result["column_mapping"] = {}
                    
                    elif key == 'RELEVANT_COLUMNS':
                        if value and value != 'None':
                            try:
                                if '[' in value:
                                    result["relevant_columns"] = eval(value)
                                else:
                                    result["relevant_columns"] = [v.strip() for v in value.split(',') if v.strip()]
                            except:
                                result["relevant_columns"] = [v.strip() for v in value.split(',') if v.strip()]
                    
                    elif key == 'EXPANDED_QUERY':
                        if value and value != 'None':
                            result["expanded_query"] = value
                    
                    elif key == 'CONTEXT_HINTS':
                        if value and value != 'None':
                            try:
                                if '[' in value:
                                    result["context_hints"] = eval(value)
                                else:
                                    result["context_hints"] = [v.strip() for v in value.split(',') if v.strip()]
                            except:
                                result["context_hints"] = [v.strip() for v in value.split(',') if v.strip()]
                    
                    elif key == 'QUERY_TYPE':
                        if value and value != 'None':
                            result["query_type"] = value.lower()
            
            # Validate columns exist in dataset
            valid_columns = []
            for col in result["relevant_columns"]:
                if col in df.columns:
                    valid_columns.append(col)
                else:
                    # Try fuzzy matching
                    similar = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                    if similar:
                        valid_columns.append(similar[0])
            
            result["relevant_columns"] = valid_columns
            
        except Exception as e:
            print(f"[QueryContextAgent] Parse error: {e}")
            result["reasoning"] = f"Parse error, using fallback: {e}"
        
        return result

    def _simple_context_analysis(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple fallback context analysis without LLM
        """
        result = {
            "original_query": query,
            "expanded_query": query,
            "abbreviations_found": [],
            "column_mapping": {},
            "relevant_columns": [],
            "context_hints": [],
            "query_type": "general",
            "reasoning": "Simple fallback analysis"
        }
        
        query_lower = query.lower()
        columns = list(df.columns)
        columns_lower = [col.lower() for col in columns]
        
        # Check for abbreviations
        for abbrev, expansions in self.common_abbreviations.items():
            if abbrev in query_lower:
                result["abbreviations_found"].append(abbrev)
                
                # Try to find matching columns
                found_match = False
                for expansion in expansions:
                    matching_cols = [col for col in columns if expansion in col.lower() or col.lower() in expansion]
                    if matching_cols:
                        result["column_mapping"][abbrev] = matching_cols[0]
                        result["relevant_columns"].extend(matching_cols)
                        # Expand the query
                        result["expanded_query"] = result["expanded_query"].replace(abbrev, expansion)
                        found_match = True
                        break
                
                # If no match found, add context hint about the mismatch
                if not found_match:
                    result["context_hints"].append(f"'{abbrev}' (likely '{expansions[0]}') not found in dataset")
                    result["context_hints"].append(f"Available columns: {', '.join(columns)}")
                    # Don't expand the query if we can't map it
                    result["expanded_query"] = query
        
        # Determine query type
        if any(word in query_lower for word in ['highest', 'maximum', 'top', 'largest', 'most']):
            result["query_type"] = "ranking_max"
        elif any(word in query_lower for word in ['lowest', 'minimum', 'bottom', 'smallest', 'least']):
            result["query_type"] = "ranking_min"
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            result["query_type"] = "comparison"
        elif any(word in query_lower for word in ['average', 'mean', 'sum', 'total', 'count']):
            result["query_type"] = "aggregation"
        
        # Add context hints
        if result["query_type"].startswith("ranking"):
            result["context_hints"].append("This is a ranking query - look for numeric columns to sort by")
        
        if result["relevant_columns"]:
            result["context_hints"].append(f"Focus analysis on columns: {result['relevant_columns']}")
        
        return result

@tool
def expand_query_context(query: str) -> str:
    """
    Expand a user query with better context and column mapping
    
    query: The original user query that might contain abbreviations or unclear terms
    """
    try:
        # Get current dataframe
        current_df = df_manager.get_current_dataframe()
        
        if current_df is None:
            return f"**Original query:** {query}\n\n**Note:** No dataset loaded for context expansion."
        
        # Create a temporary QueryContextAgent for this analysis
        from langchain_openai import ChatOpenAI
        import os
        
        llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=os.environ.get('OPENAI_API_KEY'))
        context_agent = QueryContextAgent(llm)
        
        # Analyze the query
        result = context_agent._analyze_query_context(query, current_df)
        
        # Format the response
        response = f"### Query Context Analysis\n\n"
        response += f"**Original query:** {query}\n\n"
        response += f"**Expanded query:** {result['expanded_query']}\n\n"
        
        if result['abbreviations_found']:
            response += f"**Abbreviations found:** `{', '.join(result['abbreviations_found'])}`\n\n"
        
        if result['column_mapping']:
            response += f"**Column mapping:**\n"
            for k, v in result['column_mapping'].items():
                response += f"- `{k}` → `{v}`\n"
            response += f"\n"
        
        if result['relevant_columns']:
            response += f"**Relevant columns:** `{', '.join(result['relevant_columns'])}`\n\n"
        
        if result['context_hints']:
            response += f"**Context hints:**\n"
            for hint in result['context_hints']:
                response += f"- {hint}\n"
            response += f"\n"
        
        response += f"**Query type:** `{result['query_type']}`"
        
        return response
        
    except Exception as e:
        return f"**Error expanding query context:** {e}\n\n**Original query:** {query}"