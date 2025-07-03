from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import pandas as pd
import numpy as np
import json
import os
import ast
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def safe_parse_action_input(action_input):
    """Safely parse action input from various formats (dict, JSON string, Python dict string)"""
    if isinstance(action_input, dict):
        return action_input
    try:
        return ast.literal_eval(action_input)
    except Exception:
        try:
            return json.loads(action_input)
        except Exception:
            return {}

class DataFrameManager:
    """Singleton class to manage uploaded dataframes across the application"""
    _instance = None
    _dataframes = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataFrameManager, cls).__new__(cls)
        return cls._instance
    
    def store_dataframe(self, name: str, df: pd.DataFrame):
        """Store a dataframe with a given name"""
        self._dataframes[name] = df
    
    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """Retrieve a dataframe by name"""
        return self._dataframes.get(name)
    
    def list_dataframes(self) -> list:
        """List all available dataframe names"""
        return list(self._dataframes.keys())
    
    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the most recently uploaded dataframe"""
        if self._dataframes:
            return list(self._dataframes.values())[-1]
        return None

# Global dataframe manager
df_manager = DataFrameManager()

@tool
def analyze_dataframe(analysis_type: str, column: str = None, additional_params: str = None) -> str:
    """Analyze the uploaded dataframe with various analysis types.
    
    analysis_types: 'summary', 'describe', 'head', 'tail', 'info', 'columns', 'shape', 
                   'unique_values', 'missing_values', 'correlation', 'value_counts'
    column: specific column to analyze (optional)
    additional_params: JSON string with additional parameters like max_rows or detailed
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Parse additional parameters
        max_rows = 5
        detailed = False
        if additional_params:
            try:
                params = json.loads(additional_params)
                max_rows = params.get('max_rows', 5)
                detailed = params.get('detailed', False)
            except:
                pass
            
        # For large dataframes, sample data unless detailed analysis is requested
        sample_df = df
        if df.shape[0] > 10000 and not detailed:
            sample_df = df.sample(10000, random_state=42)
            sample_notice = f"Note: Using {10000} random rows from {df.shape[0]} total rows for efficiency. " + \
                            f"For full analysis, add 'detailed':true to additional_params."
        else:
            sample_notice = ""
        
        if analysis_type == "summary":
            # Return a concise summary
            return f"{sample_notice}Dataset shape: {df.shape}\nColumns: {list(df.columns)}\nData types summary: " + \
                   f"{df.dtypes.value_counts().to_dict()}"
        
        elif analysis_type == "describe":
            if column:
                if column in df.columns:
                    return f"{sample_notice}{df[column].describe().to_string()}"
                else:
                    return f"Column '{column}' not found. Available columns: {list(df.columns)}"
            
            # For large datasets with many columns, show only numeric columns to save tokens
            if df.shape[1] > 10 and not detailed:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                return f"{sample_notice}Describing numeric columns only:\n{df[numeric_cols].describe().to_string()}"
            return f"{sample_notice}{df.describe().to_string()}"
        
        elif analysis_type == "head":
            rows = min(max_rows, 10)  # Cap at 10 rows to save tokens
            if additional_params:
                try:
                    params = json.loads(additional_params)
                    rows = min(params.get('rows', rows), 10)  # Cap at 10 rows
                except:
                    pass
            
            # For dataframes with many columns, show only a subset
            if df.shape[1] > 15 and not detailed:
                important_cols = list(df.columns)[:15]
                return f"{sample_notice}First {rows} rows (showing 15/{df.shape[1]} columns):\n{df[important_cols].head(rows).to_string()}"
            return f"{sample_notice}First {rows} rows:\n{df.head(rows).to_string()}"
        
        elif analysis_type == "tail":
            rows = min(max_rows, 10)  # Cap at 10 rows to save tokens
            if additional_params:
                try:
                    params = json.loads(additional_params)
                    rows = min(params.get('rows', rows), 10)  # Cap at 10 rows
                except:
                    pass
                    
            # For dataframes with many columns, show only a subset
            if df.shape[1] > 15 and not detailed:
                important_cols = list(df.columns)[:15]
                return f"{sample_notice}Last {rows} rows (showing 15/{df.shape[1]} columns):\n{df[important_cols].tail(rows).to_string()}"
            return f"{sample_notice}Last {rows} rows:\n{df.tail(rows).to_string()}"
        
        elif analysis_type == "info":
            buffer = io.StringIO()
            sample_df.info(buf=buffer, verbose=detailed)
            return f"{sample_notice}{buffer.getvalue()}"
        
        elif analysis_type == "columns":
            # For dataframes with many columns, show counts by type
            if df.shape[1] > 20 and not detailed:
                type_counts = df.dtypes.value_counts().to_dict()
                return f"Column count by type: {type_counts}\nTotal columns: {df.shape[1]}"
            return f"Columns: {list(df.columns)}"
        
        elif analysis_type == "shape":
            return f"Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})"
        
        elif analysis_type == "unique_values":
            if column:
                if column in df.columns:
                    # Limit number of unique values shown
                    unique_count = df[column].nunique()
                    max_values = 15 if detailed else 10
                    unique_sample = df[column].unique()[:max_values]
                    return f"Unique values in '{column}': {unique_count}\nValues: {unique_sample}" + \
                           (f" (showing {max_values}/{unique_count})" if unique_count > max_values else "")
                else:
                    return f"Column '{column}' not found. Available columns: {list(df.columns)}"
            return "Please specify a column for unique values analysis"
        
        elif analysis_type == "missing_values":
            missing = df.isnull().sum()
            
            # Only show columns with missing values to save tokens
            missing = missing[missing > 0]
            
            if missing.empty:
                return "No missing values found in the dataset."
                
            # Sort by most missing and show percentage
            missing_df = pd.DataFrame({
                'Missing Count': missing,
                'Missing %': (missing / len(df) * 100).round(2)
            }).sort_values('Missing Count', ascending=False)
            
            # Limit to top N columns with missing values
            if len(missing_df) > 10 and not detailed:
                top_missing = missing_df.head(10)
                return f"Top 10 columns with missing values (out of {len(missing_df)}):\n{top_missing.to_string()}"
            return f"Missing values:\n{missing_df.to_string()}"
        
        elif analysis_type == "correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # For dataframes with many numeric columns, limit correlation matrix
                if len(numeric_cols) > 12 and not detailed:
                    # Find columns with highest variance (likely most informative)
                    variances = df[numeric_cols].var()
                    top_cols = variances.nlargest(12).index
                    corr_matrix = df[top_cols].corr()
                    return f"{sample_notice}Correlation matrix (top 12 numeric columns by variance):\n{corr_matrix.round(2).to_string()}"
                return f"{sample_notice}Correlation matrix:\n{df[numeric_cols].corr().round(2).to_string()}"
            else:
                return "Not enough numeric columns for correlation analysis"
        
        elif analysis_type == "value_counts":
            if column:
                if column in df.columns:
                    # Limit number of values shown
                    count_limit = 15 if detailed else 10
                    value_counts = df[column].value_counts().head(count_limit)
                    total_values = df[column].nunique()
                    return f"Value counts for '{column}' (showing {len(value_counts)}/{total_values}):\n{value_counts.to_string()}"
                else:
                    return f"Column '{column}' not found. Available columns: {list(df.columns)}"
            return "Please specify a column for value counts"
        
        else:
            return f"Unknown analysis type: {analysis_type}"
    
    except Exception as e:
        return f"Error analyzing dataframe: {e}"

@tool
def filter_dataframe(filter_expression: str) -> str:
    """Filter the dataframe using pandas query syntax or column operations.
    
    Examples:
    - "column_name > 100"
    - "column_name.isin(['value1', 'value2'])"
    - "column_name.str.contains('search_term')"
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Try to evaluate the filter expression
        filtered_df = df.query(filter_expression)
        
        result = f"Filter applied: {filter_expression}\n"
        result += f"Original shape: {df.shape}\n"
        result += f"Filtered shape: {filtered_df.shape}\n"
        result += f"First 10 rows of filtered data:\n{filtered_df.head(10).to_string()}"
        
        return result
    
    except Exception as e:
        return f"Error filtering dataframe: {e}\nTip: Use pandas query syntax like 'column_name > value'"

@tool
def group_and_aggregate(group_by_column: str, agg_column: str, agg_function: str) -> str:
    """Group the dataframe by a column and apply aggregation.
    
    agg_function: 'sum', 'mean', 'count', 'min', 'max', 'std', 'median'
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        if group_by_column not in df.columns:
            return f"Group by column '{group_by_column}' not found. Available columns: {list(df.columns)}"
        
        if agg_column not in df.columns:
            return f"Aggregation column '{agg_column}' not found. Available columns: {list(df.columns)}"
        
        grouped = df.groupby(group_by_column)[agg_column]
        
        if agg_function == 'sum':
            result = grouped.sum()
        elif agg_function == 'mean':
            result = grouped.mean()
        elif agg_function == 'count':
            result = grouped.count()
        elif agg_function == 'min':
            result = grouped.min()
        elif agg_function == 'max':
            result = grouped.max()
        elif agg_function == 'std':
            result = grouped.std()
        elif agg_function == 'median':
            result = grouped.median()
        else:
            return f"Unknown aggregation function: {agg_function}"
        
        return f"Groupby {group_by_column}, {agg_function} of {agg_column}:\n{result.to_string()}"
    
    except Exception as e:
        return f"Error grouping and aggregating: {e}"

@tool
def find_extreme_values(column: str, operation: str, top_n: int = 5) -> str:
    """Find extreme values (highest/lowest) in a specific column.
    
    operation: 'highest', 'lowest', 'maximum', 'minimum', 'top', 'bottom'
    top_n: number of results to return (default 5)
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
        
        # Clean the column data
        column_data = df[column].copy()
        
        # Try to convert to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(column_data):
            # Try to extract numeric values from strings
            column_data = pd.to_numeric(column_data.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
        # Remove NaN values
        valid_data = df[column_data.notna()].copy()
        
        if valid_data.empty:
            return f"No valid numeric data found in column '{column}'"
        
        if operation.lower() in ['highest', 'maximum', 'top']:
            result_df = valid_data.nlargest(top_n, column)
        elif operation.lower() in ['lowest', 'minimum', 'bottom']:
            result_df = valid_data.nsmallest(top_n, column)
        else:
            return f"Unknown operation: {operation}. Use 'highest', 'lowest', 'maximum', 'minimum', 'top', or 'bottom'"
        
        # Format the result nicely
        result = f"Top {len(result_df)} {operation.lower()} values in '{column}':\n"
        
        # Show relevant columns (try to find name/identifier columns)
        display_columns = [column]
        name_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['name', 'title', 'organization', 'company', 'id'])]
        if name_columns:
            display_columns = name_columns[:2] + [column]  # Show up to 2 name columns plus the value column
        
        # Remove duplicates while preserving order
        display_columns = list(dict.fromkeys(display_columns))
        
        for i, (idx, row) in enumerate(result_df.iterrows(), 1):
            result += f"\n{i}. "
            for col in display_columns:
                if col in row:
                    result += f"{col}: {row[col]} | "
            result = result.rstrip(" | ")
        
        return result
    
    except Exception as e:
        return f"Error finding extreme values: {e}"

@tool
def search_data(search_term: str, column: str = None) -> str:
    """Search for specific terms in the dataset with enhanced matching capabilities.
    
    search_term: the term to search for (case insensitive, supports partial matches)
    column: specific column to search in (optional, searches all text columns if not specified)
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        search_term = str(search_term).lower().strip()
        results = []
        
        if column:
            if column not in df.columns:
                available_cols = list(df.columns)
                # Try to find similar column names
                similar_cols = [col for col in available_cols if search_term.lower() in col.lower()]
                error_msg = f"Column '{column}' not found. Available columns: {available_cols}"
                if similar_cols:
                    error_msg += f"\nDid you mean one of these similar columns? {similar_cols}"
                return error_msg
            
            # Search in specific column with multiple matching strategies
            col_data = df[column].astype(str)
            
            # Strategy 1: Exact case-insensitive match
            exact_mask = col_data.str.lower() == search_term
            exact_matches = df[exact_mask]
            
            # Strategy 2: Contains match (case insensitive)
            contains_mask = col_data.str.lower().str.contains(search_term, na=False, regex=False)
            contains_matches = df[contains_mask]
            
            # Strategy 3: Fuzzy match (split search term and look for all parts)
            search_parts = search_term.split()
            if len(search_parts) > 1:
                fuzzy_mask = col_data.str.lower().str.contains('|'.join(search_parts), na=False, regex=True)
                fuzzy_matches = df[fuzzy_mask]
            else:
                fuzzy_matches = pd.DataFrame()
            
            # Report results
            if len(exact_matches) > 0:
                results.append(f"Exact matches for '{search_term}' in column '{column}' ({len(exact_matches)} found):")
                results.append("DATAFRAME_START")
                results.append(exact_matches.head(10).to_string())
                results.append("DATAFRAME_END")
            
            if len(contains_matches) > len(exact_matches):
                additional_contains = contains_matches[~contains_matches.index.isin(exact_matches.index)]
                if len(additional_contains) > 0:
                    results.append(f"\nPartial matches containing '{search_term}' in column '{column}' ({len(additional_contains)} additional found):")
                    results.append("DATAFRAME_START")
                    results.append(additional_contains.head(10).to_string())
                    results.append("DATAFRAME_END")
            
            if len(fuzzy_matches) > len(contains_matches) and len(search_parts) > 1:
                additional_fuzzy = fuzzy_matches[~fuzzy_matches.index.isin(contains_matches.index)]
                if len(additional_fuzzy) > 0:
                    results.append(f"\nFuzzy matches for parts of '{search_term}' in column '{column}' ({len(additional_fuzzy)} additional found):")
                    results.append("DATAFRAME_START")
                    results.append(additional_fuzzy.head(5).to_string())
                    results.append("DATAFRAME_END")
            
            if len(contains_matches) == 0:
                results.append(f"No matches found for '{search_term}' in column '{column}'")
                # Show sample data from the column
                results.append(f"\nSample values in '{column}' column:")
                sample_values = df[column].dropna().head(10).tolist()
                results.append(str(sample_values))
        
        else:
            # Search in all text columns
            text_columns = df.select_dtypes(include=['object', 'string']).columns
            total_matches = 0
            
            for col in text_columns:
                col_data = df[col].astype(str)
                mask = col_data.str.lower().str.contains(search_term, na=False, regex=False)
                matching_rows = df[mask]
                
                if len(matching_rows) > 0:
                    total_matches += len(matching_rows)
                    results.append(f"\nFound {len(matching_rows)} matches in column '{col}':")
                    results.append("DATAFRAME_START")
                    results.append(matching_rows.head(5).to_string())
                    results.append("DATAFRAME_END")
            
            if total_matches == 0:
                results.append(f"No matches found for '{search_term}' in any text columns")
                results.append(f"\nSearched in columns: {list(text_columns)}")
                # Show sample data from each text column
                results.append("\nSample data from text columns:")
                for col in text_columns[:3]:  # Show sample from first 3 text columns
                    sample_values = df[col].dropna().head(3).tolist()
                    results.append(f"  {col}: {sample_values}")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error searching data: {e}"

@tool
def create_visualization(chart_type: str, x_column: str, y_column: str = None, title: str = "Chart") -> str:
    """Create visualizations from the dataframe.
    
    chart_type: 'line', 'bar', 'scatter', 'histogram', 'box', 'heatmap', 'pie'
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Sample large dataframes to reduce token usage
        sample_notice = ""
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
            sample_notice = "(Using 1000 sample rows)"
        
        # Truncate long labels
        x_label = x_column[:25] + "..." if len(x_column) > 25 else x_column
        y_label = y_column[:25] + "..." if y_column and len(y_column) > 25 else y_column
        title_label = title[:40] + "..." if len(title) > 40 else title
        
        plt.figure(figsize=(8, 6))  # Smaller figure size
        plt.style.use('seaborn-v0_8')
        
        if chart_type == "line":
            if y_column and y_column in df.columns:
                plt.plot(df[x_column], df[y_column], marker='o')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            else:
                plt.plot(df[x_column], marker='o')
                plt.ylabel(x_label)
        
        elif chart_type == "bar":
            if y_column and y_column in df.columns:
                # Limit the number of bars for large datasets
                if len(df) > 15:
                    # Group and aggregate to reduce number of bars
                    grouped = df.groupby(x_column)[y_column].mean()
                    if len(grouped) > 15:
                        grouped = grouped.nlargest(15)
                    plt.bar(grouped.index, grouped.values)
                else:
                    plt.bar(df[x_column], df[y_column])
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            else:
                value_counts = df[x_column].value_counts()
                if len(value_counts) > 15:
                    value_counts = value_counts.nlargest(15)
                plt.bar(value_counts.index, value_counts.values)
                plt.xlabel(x_label)
                plt.ylabel('Count')
        
        elif chart_type == "scatter":
            if y_column and y_column in df.columns:
                plt.scatter(df[x_column], df[y_column], alpha=0.7, s=40)  # Smaller point size
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            else:
                return "Scatter plot requires both x_column and y_column"
        
        elif chart_type == "histogram":
            plt.hist(df[x_column], bins=15, alpha=0.7, edgecolor='black')  # Fewer bins
            plt.xlabel(x_label)
            plt.ylabel('Frequency')
        
        elif chart_type == "box":
            if y_column and y_column in df.columns:
                # For box plots with too many categories, limit them
                if df[x_column].nunique() > 10:
                    # Get most frequent categories
                    top_cats = df[x_column].value_counts().nlargest(10).index
                    df_filtered = df[df[x_column].isin(top_cats)]
                    df_filtered.boxplot(column=y_column, by=x_column)
                else:
                    df.boxplot(column=y_column, by=x_column)
                plt.suptitle('')
            else:
                plt.boxplot(df[x_column])
                plt.ylabel(x_label)
        
        elif chart_type == "heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Limit to 10 columns for correlation heatmap
                if len(numeric_cols) > 10:
                    # Select columns with highest variance
                    variances = df[numeric_cols].var()
                    numeric_cols = variances.nlargest(10).index
                
                correlation_matrix = df[numeric_cols].corr()
                # For large correlation matrices, don't use annotations
                if len(numeric_cols) <= 6:
                    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
                else:
                    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
            else:
                return "Heatmap requires multiple numeric columns"
        
        elif chart_type == "pie":
            value_counts = df[x_column].value_counts().head(8)  # Top 8 values
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        
        else:
            return f"Unknown chart type: {chart_type}"
        
        plt.title(f"{title_label} {sample_notice}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save chart to base64 with lower DPI
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"Chart generated successfully: data:image/png;base64,{img_base64}"
    
    except Exception as e:
        return f"Error creating visualization: {e}"

@tool
def get_sample_data_for_display(rows: int = 10, columns: str = "all") -> str:
    """Get sample data from the uploaded dataframe for display purposes.
    
    rows: number of rows to return (default 10)
    columns: 'all' for all columns, or comma-separated list of column names
    """
    try:
        df = df_manager.get_current_dataframe()
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Limit rows
        display_df = df.head(rows)
        
        # Handle column selection
        if columns != "all":
            try:
                col_list = [col.strip() for col in columns.split(',')]
                available_cols = [col for col in col_list if col in df.columns]
                if available_cols:
                    display_df = display_df[available_cols]
            except:
                pass  # Use all columns if parsing fails
        
        # For large dataframes, limit columns to avoid display issues
        if display_df.shape[1] > 10:
            display_df = display_df.iloc[:, :10]
            column_note = f" (showing 10 of {df.shape[1]} columns)"
        else:
            column_note = ""
        
        # Return in a format that can be parsed as a dataframe
        result = f"Sample data from dataset{column_note}:\n\n"
        result += f"DATAFRAME_START\n"
        result += display_df.to_string(index=True)
        result += f"\nDATAFRAME_END\n"
        result += f"\nDataset info: {df.shape[0]} rows × {df.shape[1]} columns"
        
        return result
        
    except Exception as e:
        return f"Error retrieving sample data: {e}"

class PandasAgent:
    def __init__(self, llm):
        if llm is None:
            raise ValueError("PandasAgent requires a valid LLM instance")
        self.llm = llm
        self.tools = [analyze_dataframe, filter_dataframe, group_and_aggregate, find_extreme_values, search_data, create_visualization, get_sample_data_for_display]
        
        # Create a comprehensive system prompt for intelligent data analysis
        self.prompt = PromptTemplate.from_template(
            """You are an intelligent data analysis assistant with access to powerful pandas tools. Your primary goal is to help users analyze their datasets by using the available tools effectively.

CORE RESPONSIBILITIES:
1. Analyze user queries to understand their intent
2. Use appropriate tools to examine and manipulate data
3. Provide insightful, actionable responses based on actual data analysis
4. When users ask about specific data patterns, always use tools to investigate
5. For visualization requests, use create_visualization tool
6. For finding extremes (highest/lowest), use find_extreme_values tool
7. For searching specific terms/companies, use search_data tool

INTELLIGENT TOOL USAGE GUIDELINES:
- ALWAYS use tools when users ask specific questions about data
- For "companies that are hiring" -> use search_data tool to find hiring-related terms
- For "highest/lowest/maximum/minimum" -> use find_extreme_values tool
- For "plot/chart/visualize" -> use create_visualization tool after analyzing data
- For "show me data about X" -> use analyze_dataframe and search_data tools
- For "tell me about dataset" -> use analyze_dataframe with 'summary' and 'describe'
- For column-specific questions -> use analyze_dataframe with the specific column

BEHAVIOR EXPECTATIONS:
- Be proactive in data exploration
- Always validate your findings with actual data
- Provide specific, data-driven insights
- When creating visualizations, choose appropriate chart types
- Explain what you found in the data, don't just describe what you did

CURRENT CONTEXT:
User query: {query}

INSTRUCTIONS:
1. Analyze the user's intent
2. Determine which tools to use
3. Execute the tools to gather data insights
4. Provide a comprehensive, intelligent response based on your findings
5. If creating visualizations, ensure they're relevant to the query

Remember: You have access to real data - use it! Don't give generic responses when you can provide specific, data-driven insights."""
        )

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently process user's data analysis query using pandas operations and tools
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with pandas analysis results
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
        updated_state["current_agent"] = "pandas"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[PandasAgent] Processing query: {query}")
        
        try:
            # Check if data is available
            current_df = df_manager.get_current_dataframe()
            has_data = current_df is not None
            data_info = state.get("dataframe_info", {})
            
            print(f"[PandasAgent] Has data: {has_data}")
            
            # Handle case where no data is uploaded
            if not has_data:
                result = ("I'd love to help you analyze data! However, I don't see any dataset uploaded yet. "
                         "Please upload a CSV or Excel file using the file uploader in the sidebar, and then I can "
                         "provide detailed analysis, create visualizations, and answer specific questions about your data.")
            
            # Handle very simple greetings (only basic ones)
            elif query.lower().strip() in ['hi', 'hello', 'help']:
                filename = data_info.get('filename', 'uploaded dataset')
                shape = data_info.get('shape', current_df.shape if current_df is not None else (0, 0))
                result = (f"Hello! I can see you have '{filename}' loaded with {shape[0]:,} rows and {shape[1]} columns. "
                         f"I'm ready to help you analyze this data. What would you like to explore?")
            
            # For all other queries with data available, use React agent with intelligent tool calling
            else:
                try:
                    print(f"[PandasAgent] Using React agent for intelligent analysis")
                    
                    # Create a React agent with the available tools
                    react_agent = create_react_agent(self.llm, self.tools)
                    
                    # Prepare comprehensive context for the React agent
                    filename = data_info.get('filename', 'uploaded dataset')
                    shape = data_info.get('shape', current_df.shape if current_df is not None else (0, 0))
                    columns = data_info.get('columns', [])
                    
                    # Create an enhanced system message with full context
                    system_context = f"""You are analyzing a dataset called '{filename}' with {shape[0]:,} rows and {shape[1]} columns.

Available columns: {', '.join(columns[:20])}{'...' if len(columns) > 20 else ''}

IMPORTANT TOOL USAGE RULES:
1. For questions about "companies hiring" or "actively hiring" -> Use search_data tool to search for terms like "hiring", "recruiting", "open positions"
2. For "highest/lowest/top/bottom" questions -> Use find_extreme_values tool with appropriate column and operation
3. For "plot/chart/visualize" requests -> First analyze the data, then use create_visualization tool
4. For general data questions -> Start with analyze_dataframe tool to understand the data structure
5. For finding specific companies/organizations -> Use search_data tool
6. Always provide specific, data-driven insights based on actual tool results

User's question: {query}

Analyze this query and use the appropriate tools to provide a comprehensive, data-driven response."""
                    
                    # Invoke the React agent with the enhanced context
                    agent_result = react_agent.invoke({
                        "messages": [HumanMessage(content=system_context)]
                    })
                    
                    # Extract the result from the agent response
                    if agent_result and "messages" in agent_result:
                        last_message = agent_result["messages"][-1]
                        if hasattr(last_message, 'content'):
                            result = last_message.content
                        else:
                            result = str(last_message)
                    else:
                        result = "I processed your request but couldn't extract a clear result. Let me try a different approach."
                        
                    print(f"[PandasAgent] React agent completed successfully")
                    
                except Exception as e:
                    print(f"[PandasAgent] React agent error: {e}")
                    
                    # Intelligent fallback - try to handle the query with direct tool calls
                    try:
                        result = self._intelligent_fallback(query, data_info, current_df)
                    except Exception as fallback_error:
                        print(f"[PandasAgent] Fallback error: {fallback_error}")
                        result = (f"I encountered an issue processing your request: {str(e)}. "
                                f"However, I can see your dataset '{data_info.get('filename', 'data')}' is loaded. "
                                f"Could you try rephrasing your question? For example:\n"
                                f"• 'Show me a summary of the data'\n"
                                f"• 'What are the column names?'\n"
                                f"• 'Create a chart of [column name]'")
            
            print(f"[PandasAgent] Result length: {len(result)} characters")
            
            # Update state with pandas analysis results
            updated_state["agent_outputs"]["pandas"] = {
                "status": "completed",
                "result": result,
                "reasoning": "Completed intelligent pandas data analysis with tool usage"
            }
            
            return updated_state
            
        except Exception as e:
            print(f"[PandasAgent] Error: {e}")
            error_message = f"Error in pandas agent: {str(e)}"
            
            # Update state with error information
            updated_state["agent_outputs"]["pandas"] = {
                "status": "error", 
                "result": error_message,
                "error": str(e)
            }
            
            return updated_state
    
    def _intelligent_fallback(self, query: str, data_info: dict, current_df) -> str:
        """
        Intelligent fallback method that tries to handle queries with direct tool calls
        """
        query_lower = query.lower()
        
        # Handle hiring/recruitment queries
        if any(term in query_lower for term in ['hiring', 'recruit', 'job', 'position']):
            search_result = search_data.invoke("hiring")
            if "No matches found" not in search_result:
                return f"I found information about hiring in your dataset:\n\n{search_result}"
            else:
                # Try alternative search terms
                for term in ['recruit', 'job', 'position', 'career']:
                    alt_result = search_data.invoke(term)
                    if "No matches found" not in alt_result:
                        return f"I found job-related information in your dataset:\n\n{alt_result}"
                return "I couldn't find specific hiring information in your dataset. Let me show you the available columns and data structure instead:\n\n" + analyze_dataframe.invoke("columns")
        
        # Handle visualization requests
        elif any(term in query_lower for term in ['plot', 'chart', 'visualize', 'graph']):
            # First get data structure
            columns_info = analyze_dataframe.invoke("columns")
            return f"I'd be happy to create a visualization! Here are the available columns:\n\n{columns_info}\n\nPlease specify which column(s) you'd like to visualize and what type of chart you prefer (bar, line, scatter, etc.)."
        
        # Handle extreme value queries
        elif any(term in query_lower for term in ['highest', 'lowest', 'maximum', 'minimum', 'top', 'bottom']):
            columns_info = analyze_dataframe.invoke("columns")
            return f"I can find extreme values for you! Here are the available columns:\n\n{columns_info}\n\nPlease specify which column you'd like me to analyze for extreme values."
        
        # Default: provide data overview
        else:
            summary = analyze_dataframe.invoke("summary")
            return f"Here's an overview of your dataset:\n\n{summary}\n\nFeel free to ask specific questions about the data, request visualizations, or search for particular information!"
