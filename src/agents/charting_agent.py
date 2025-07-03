from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langgraph.prebuilt import create_react_agent
from typing import Optional, Type, Dict, Any, List, TypedDict
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import base64
from io import BytesIO
import json
import ast
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import the shared DataFrameManager - use try/except to handle import issues
try:
    from .pandas_agent import DataFrameManager, df_manager
except ImportError:
    # Fallback: create our own instance
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pandas_agent import DataFrameManager
    df_manager = DataFrameManager()

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

class GenerateChartInput(BaseModel):
    data: str = Field(description="JSON string representing the data")
    chart_type: str = Field(description="Type of chart to generate")
    x_axis: str = Field(description="Column name for x-axis")
    y_axis: str = Field(description="Column name for y-axis")
    title: str = Field(description="Chart title")

class LoadAndChartCSVInput(BaseModel):
    file_path: str = Field(description="Path to the CSV file")
    chart_type: str = Field(description="Type of chart to generate")
    x_axis: str = Field(description="Column name for x-axis")
    y_axis: str = Field(description="Column name for y-axis")
    title: str = Field(description="Chart title")

class RobustGenerateChartTool(BaseTool):
    name: str = "generate_chart"
    description: str = """Generates various types of charts from provided data and returns it as a base64 encoded PNG image.
    Supported chart types: 'line', 'bar', 'scatter', 'histogram', 'boxplot', 'heatmap', 'pie'
    The `data` parameter should be a JSON string representing the data.
    `x_axis` and `y_axis` are the column names for the axes, and `title` is the chart title."""
    args_schema: Type[BaseModel] = GenerateChartInput
    
    def _run(self, data: str, chart_type: str, x_axis: str, y_axis: str, title: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return generate_chart_impl(data, chart_type, x_axis, y_axis, title)

class RobustLoadAndChartCSVTool(BaseTool):
    name: str = "load_and_chart_csv"
    description: str = """Loads data from a CSV file and generates a chart directly.
    This tool combines data loading and charting for convenience."""
    args_schema: Type[BaseModel] = LoadAndChartCSVInput
    
    def _run(self, file_path: str, chart_type: str, x_axis: str, y_axis: str, title: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return load_and_chart_csv_impl(file_path, chart_type, x_axis, y_axis, title)

def generate_chart_impl(data: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Implementation of chart generation with robust error handling"""
    try:
        df = pd.read_json(data)
        
        # Limit dataframe size for large datasets to save tokens
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
            sample_notice = "(Using 1000 sample rows)"
        else:
            sample_notice = ""
        
        # For wide dataframes, remove unnecessary columns to save memory
        if len(df.columns) > 5:
            needed_cols = [x_axis, y_axis]
            df = df[needed_cols]
        
        # Truncate long strings in axis labels
        x_label = x_axis[:25] + "..." if len(x_axis) > 25 else x_axis
        y_label = y_axis[:25] + "..." if len(y_axis) > 25 else y_axis
        
        # Truncate long title
        if len(title) > 50:
            title = title[:47] + "..."
            
        plt.figure(figsize=(8, 5))  # Even smaller figure size
        plt.style.use('seaborn-v0_8')
        
        if chart_type == "line":
            plt.plot(df[x_axis], df[y_axis], marker='o', linewidth=2, markersize=6)
        elif chart_type == "bar":
            # Limit the number of bars shown
            if len(df) > 15:
                counts = df.groupby(x_axis)[y_axis].mean().nlargest(15)
                plt.bar(counts.index, counts.values, alpha=0.8, color='skyblue', edgecolor='navy')
            else:
                plt.bar(df[x_axis], df[y_axis], alpha=0.8, color='skyblue', edgecolor='navy')
        elif chart_type == "scatter":
            plt.scatter(df[x_axis], df[y_axis], alpha=0.7, s=40, color='coral')  # Smaller point size
        elif chart_type == "histogram":
            plt.hist(df[y_axis], bins=12, alpha=0.7, color='lightgreen', edgecolor='black')  # Further reduced bin count
            plt.xlabel(y_label)
            plt.ylabel('Frequency')
        elif chart_type == "boxplot":
            df.boxplot(column=y_axis, by=x_axis)
            plt.suptitle('')  # Remove default title
        elif chart_type == "pie":
            # For pie charts, limit to top 8 categories to save tokens
            if len(df) > 8:
                counts = df[y_axis].value_counts().nlargest(8)
                plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            else:
                plt.pie(df[y_axis], labels=df[x_axis], autopct='%1.1f%%', startangle=90)
        else:
            return f"Unsupported chart type: {chart_type}. Supported types: line, bar, scatter, histogram, boxplot, pie"
            
        if chart_type not in ["histogram", "boxplot", "pie"]:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        
        plt.title(f"{title} {sample_notice}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI from 100 to 80
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return f"Chart generated successfully: data:image/png;base64,{img_base64}"
    except Exception as e:
        return f"Error generating chart: {e}"

def load_and_chart_csv_impl(file_path: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Implementation of CSV loading and charting with robust error handling"""
    try:
        # Read only the necessary columns to save memory
        try:
            if x_axis != y_axis:
                df = pd.read_csv(file_path, usecols=[x_axis, y_axis])
            else:
                df = pd.read_csv(file_path, usecols=[x_axis])
        except:
            # If specific columns can't be read, read all and then filter
            df = pd.read_csv(file_path)
            
        # Convert to JSON with only the necessary data
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
            data_json = df_sample.to_json(orient='records')
        else:
            data_json = df.to_json(orient='records')
            
        return generate_chart_impl(data_json, chart_type, x_axis, y_axis, title)
    except Exception as e:
        return f"Error loading CSV and generating chart: {e}"

@tool
def generate_chart(data: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Generates various types of charts from provided data and returns it as a base64 encoded PNG image.
    Supported chart types: 'line', 'bar', 'scatter', 'histogram', 'boxplot', 'heatmap', 'pie'
    The `data` parameter should be a JSON string representing the data.
    `x_axis` and `y_axis` are the column names for the axes, and `title` is the chart title.
    """
    return generate_chart_impl(data, chart_type, x_axis, y_axis, title)

@tool
def load_and_chart_csv(file_path: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Loads data from a CSV file and generates a chart directly.
    This tool combines data loading and charting for convenience.
    """
    return load_and_chart_csv_impl(file_path, chart_type, x_axis, y_axis, title)

@tool
def create_chart_from_uploaded_data(chart_type: str, x_column: str, y_column: str = None, title: str = "Chart", top_n: int = 10) -> str:
    """Create a chart from the currently uploaded dataframe.
    
    chart_type: 'line', 'bar', 'scatter', 'histogram', 'box', 'heatmap', 'pie'
    x_column: column name for x-axis  
    y_column: column name for y-axis (optional for some chart types)
    title: chart title
    top_n: number of top values to show (for bar charts, pie charts)
    """
    try:
        # Try to get dataframe from pandas_agent first, then fallback
        try:
            from .pandas_agent import df_manager as pandas_df_manager
            df = pandas_df_manager.get_current_dataframe()
        except:
            df = df_manager.get_current_dataframe()
            
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        if x_column not in df.columns:
            return f"Column '{x_column}' not found. Available columns: {list(df.columns)}"
        
        if y_column and y_column not in df.columns:
            return f"Column '{y_column}' not found. Available columns: {list(df.columns)}"
        
        # Sample large dataframes to reduce processing time
        sample_notice = ""
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
            sample_notice = " (Using 1000 sample rows)"
        
        plt.figure(figsize=(10, 6))
        plt.style.use('default')
        
        if chart_type == "bar":
            if y_column:
                # Validate that y_column is numeric for aggregation
                if not pd.api.types.is_numeric_dtype(df[y_column]):
                    # Try to convert to numeric, if it fails, use value counts instead
                    try:
                        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
                        # Remove rows where conversion failed (NaN values)
                        df = df.dropna(subset=[y_column])
                        if df.empty:
                            return f"Column '{y_column}' contains no valid numeric data for charting."
                    except:
                        return f"Column '{y_column}' is not numeric and cannot be used for bar chart aggregation. Try using it as x-axis instead."
                
                # Group and aggregate for bar chart
                if df[x_column].dtype == 'object':
                    # Categorical x-axis, aggregate y values
                    grouped = df.groupby(x_column)[y_column].sum().nlargest(top_n)
                    plt.bar(range(len(grouped)), grouped.values)
                    plt.xticks(range(len(grouped)), grouped.index, rotation=45, ha='right')
                    plt.ylabel(y_column)
                else:
                    # Numeric x-axis
                    sorted_df = df.nlargest(top_n, y_column)
                    plt.bar(sorted_df[x_column], sorted_df[y_column])
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
            else:
                # Value counts bar chart
                value_counts = df[x_column].value_counts().head(top_n)
                plt.bar(range(len(value_counts)), value_counts.values)
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
                plt.ylabel('Count')
            plt.xlabel(x_column)
        
        elif chart_type == "scatter":
            if y_column:
                plt.scatter(df[x_column], df[y_column], alpha=0.7)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            else:
                return "Scatter plot requires both x_column and y_column"
        
        elif chart_type == "line":
            if y_column:
                plt.plot(df[x_column], df[y_column], marker='o')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            else:
                plt.plot(df[x_column], marker='o')
                plt.ylabel(x_column)
        
        elif chart_type == "histogram":
            plt.hist(df[x_column], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel(x_column)
            plt.ylabel('Frequency')
        
        elif chart_type == "box":
            if y_column:
                # Box plot by category
                if df[x_column].nunique() > 10:
                    # Limit categories for readability
                    top_cats = df[x_column].value_counts().head(10).index
                    df_filtered = df[df[x_column].isin(top_cats)]
                    df_filtered.boxplot(column=y_column, by=x_column)
                else:
                    df.boxplot(column=y_column, by=x_column)
                plt.suptitle('')
            else:
                plt.boxplot(df[x_column])
                plt.ylabel(x_column)
        
        elif chart_type == "pie":
            value_counts = df[x_column].value_counts().head(top_n)
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        
        elif chart_type == "heatmap":
            # Correlation heatmap for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            else:
                return "Heatmap requires multiple numeric columns"
        
        else:
            return f"Unknown chart type: {chart_type}. Supported types: line, bar, scatter, histogram, box, heatmap, pie"
        
        plt.title(f"{title}{sample_notice}")
        plt.tight_layout()
        
        # Save chart to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"Chart generated successfully: data:image/png;base64,{img_base64}"
    
    except Exception as e:
        return f"Error creating chart: {e}"

class ChartingAgent:
    def __init__(self, llm):
        if llm is None:
            raise ValueError("ChartingAgent requires a valid LLM instance")
        self.llm = llm
        # Include the new dynamic code generation tool as the primary tool
        self.tools = [
            generate_and_execute_chart_code,  # Primary tool for dynamic code generation
            create_chart_from_uploaded_data,
            RobustGenerateChartTool(),
            RobustLoadAndChartCSVTool()
        ]
        self.prompt = PromptTemplate.from_template(
            """You are an expert data visualization agent that generates custom Python plotting code on demand.

AVAILABLE TOOLS:
- generate_and_execute_chart_code: PREFERRED - Generates custom Python code for any visualization request
- create_chart_from_uploaded_data: Fallback for simple predefined chart types
- generate_chart: For JSON data visualization
- load_and_chart_csv: For CSV file visualization

INTELLIGENT APPROACH:
1. ALWAYS try generate_and_execute_chart_code FIRST for maximum flexibility
2. This tool can create any type of visualization by generating custom Python code
3. It can handle complex requests like multiple variables, subplots, custom styling
4. Use fallback tools only if dynamic generation fails

DYNAMIC CODE CAPABILITIES:
- Custom chart types (violin plots, swarm plots, 3D plots, etc.)
- Multiple subplots and complex layouts
- Advanced statistical visualizations
- Custom color schemes and styling
- Interactive elements and annotations
- Data transformations and aggregations

EXAMPLES OF DYNAMIC REQUESTS:
- "Create a violin plot showing distribution by category"
- "Make a subplot with revenue and expenses side by side"
- "Show correlation matrix as a heatmap with annotations"
- "Create a stacked bar chart with percentages"
- "Plot time series with trend lines"

USER REQUEST: {input}

Use generate_and_execute_chart_code to create a custom visualization that perfectly matches the user's request."""
        )
        # Create the LangGraph React agent
        self.agent = create_react_agent(self.llm, self.tools)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently generate data visualizations based on user requests
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with chart generation results
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
        updated_state["current_agent"] = "chart"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[ChartingAgent] Processing query: {query}")
        
        try:
            # Check if data is available - be more thorough in checking
            try:
                from .pandas_agent import df_manager as pandas_df_manager
                current_df = pandas_df_manager.get_current_dataframe()
            except:
                # Fallback to our own df_manager
                current_df = df_manager.get_current_dataframe()
            
            has_data = current_df is not None
            
            print(f"[ChartingAgent] Has data: {has_data}")
            if has_data:
                print(f"[ChartingAgent] Data shape: {current_df.shape}")
                print(f"[ChartingAgent] Columns: {list(current_df.columns)[:5]}")
            
            if not has_data:
                result = ("I'd be happy to create visualizations for you! However, I don't see any dataset uploaded yet. "
                         "Please upload a CSV or Excel file using the file uploader in the sidebar, and then I can "
                         "create charts and visualizations from your data.")
            else:
                # Use intelligent analysis to create appropriate visualizations
                result = self._intelligent_chart_creation(query, current_df)
            
            print(f"[ChartingAgent] Result: {result[:200]}...")
            
            # Update state with chart generation results
            updated_state["agent_outputs"]["chart"] = {
                "status": "completed",
                "result": result,
                "reasoning": "Completed intelligent chart generation"
            }
            
            return updated_state
            
        except Exception as e:
            print(f"[ChartingAgent] Error: {e}")
            error_message = f"Error in charting agent: {e}"
            
            # Update state with error information
            updated_state["agent_outputs"]["chart"] = {
                "status": "error",
                "result": error_message,
                "error": str(e)
            }
            
            return updated_state
        
    def _intelligent_chart_creation(self, query: str, df) -> str:
        """
        Use LLM intelligence to analyze the query and create appropriate charts
        """
        print(f"[ChartingAgent] Intelligent processing: {query}")
        
        try:
            # First, try dynamic code generation for maximum flexibility
            result = generate_and_execute_chart_code.invoke({
                "user_request": query,
                "chart_description": f"Create a visualization based on the user request: {query}"
            })
            
            if "Error" not in result and "data:image/png;base64," in result:
                return f"I generated custom Python code to create exactly what you requested:\n\n{result}"
            else:
                print(f"[ChartingAgent] Dynamic generation failed, trying LLM analysis fallback")
                # Fallback to LLM analysis approach if dynamic generation fails
                return self._llm_analysis_fallback(query, df)
                
        except Exception as e:
            print(f"[ChartingAgent] Dynamic code generation error: {e}")
            # Fallback to LLM analysis approach
            return self._llm_analysis_fallback(query, df)
    
    def _llm_analysis_fallback(self, query: str, df) -> str:
        """
        LLM analysis fallback when dynamic code generation fails
        """
        query_lower = query.lower()
        columns = list(df.columns)
        
        print(f"[ChartingAgent] LLM analysis fallback: {query}")
        print(f"[ChartingAgent] Available columns: {columns[:10]}")
        
        # Let the LLM analyze the query and suggest the approach
        try:
            analysis_prompt = f"""Analyze this data visualization request and suggest the best approach:

Query: "{query}"
Available columns: {columns}

Task: Determine the most appropriate:
1. Chart type (bar, line, scatter, histogram, pie, box)
2. X-axis column 
3. Y-axis column (if needed)
4. Number of items to show (for top-N queries)

Guidelines:
- For "top N" or "highest/lowest": use bar charts with appropriate top_n
- For "over time" or "trends": use line charts with time on x-axis
- For "distribution": use histograms
- For "relationship between": use scatter plots
- For "proportion" or "percentage": use pie charts

Respond in this exact format:
CHART_TYPE: [type]
X_COLUMN: [column name]
Y_COLUMN: [column name or None]
TOP_N: [number or None]
TITLE: [descriptive title]
REASONING: [brief explanation]"""

            # Get LLM analysis
            llm_response = self.llm.invoke(analysis_prompt)
            analysis = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            print(f"[ChartingAgent] LLM Analysis: {analysis[:200]}...")
            
            # Parse the LLM response
            chart_params = self._parse_llm_analysis(analysis, df)
            
            if chart_params:
                # Use the LLM-determined parameters to create the chart
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": chart_params["chart_type"],
                    "x_column": chart_params["x_column"],
                    "y_column": chart_params["y_column"],
                    "title": chart_params["title"],
                    "top_n": chart_params["top_n"]
                })
                
                reasoning = chart_params.get("reasoning", "LLM-driven analysis")
                return f"I analyzed your request and created a {chart_params['chart_type']} chart. {reasoning}\n\n{result}"
            else:
                # Final fallback to rule-based approach
                return self._direct_chart_fallback(query, df)
                
        except Exception as e:
            print(f"[ChartingAgent] LLM analysis error: {e}")
            # Final fallback to rule-based approach
            return self._direct_chart_fallback(query, df)
    
    def _parse_llm_analysis(self, analysis: str, df) -> dict:
        """
        Parse the LLM analysis response and validate parameters
        """
        try:
            lines = analysis.split('\n')
            params = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'chart_type':
                        valid_types = ['bar', 'line', 'scatter', 'histogram', 'pie', 'box', 'heatmap']
                        if value.lower() in valid_types:
                            params['chart_type'] = value.lower()
                    elif key == 'x_column':
                        if value in df.columns:
                            params['x_column'] = value
                        else:
                            # Try to find similar column
                            similar = [col for col in df.columns if value.lower() in col.lower()]
                            if similar:
                                params['x_column'] = similar[0]
                    elif key == 'y_column':
                        if value.lower() == 'none':
                            params['y_column'] = None
                        elif value in df.columns:
                            # Validate that y_column is numeric for aggregation charts
                            if pd.api.types.is_numeric_dtype(df[value]):
                                params['y_column'] = value
                            else:
                                # Try to find a similar numeric column
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                similar = [col for col in numeric_cols if value.lower() in col.lower()]
                                if similar:
                                    params['y_column'] = similar[0]
                                else:
                                    # If no similar numeric column, set to None for value counts chart
                                    params['y_column'] = None
                        else:
                            # Try to find similar numeric column
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            similar = [col for col in numeric_cols if value.lower() in col.lower()]
                            if similar:
                                params['y_column'] = similar[0]
                            else:
                                params['y_column'] = None
                    elif key == 'top_n':
                        if value.lower() == 'none':
                            params['top_n'] = 10  # Default
                        else:
                            try:
                                params['top_n'] = int(value)
                            except:
                                params['top_n'] = 10
                    elif key == 'title':
                        params['title'] = value
                    elif key == 'reasoning':
                        params['reasoning'] = value
            
            # Validate required parameters
            if 'chart_type' in params and 'x_column' in params:
                return params
            else:
                return None
                
        except Exception as e:
            print(f"[ChartingAgent] Error parsing LLM analysis: {e}")
            return None

    def _analyze_data_structure(self, df) -> dict:
        """
        Generically analyze the dataset structure to understand what types of visualizations are possible
        """
        analysis = {
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'string']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
            'suitable_for_pie': [],
            'suitable_for_bar': [],
            'suitable_for_histogram': [],
            'suitable_for_scatter': [],
            'high_cardinality': [],
            'low_cardinality': []
        }
        
        # Analyze categorical columns for chart suitability
        for col in analysis['categorical_columns']:
            unique_count = df[col].nunique()
            non_null_count = df[col].count()
            
            if unique_count <= 10 and non_null_count > 0:
                analysis['suitable_for_pie'].append(col)
                analysis['low_cardinality'].append(col)
            elif unique_count <= 20 and non_null_count > 0:
                analysis['suitable_for_bar'].append(col)
                analysis['low_cardinality'].append(col)
            elif unique_count > 50:
                analysis['high_cardinality'].append(col)
        
        # Analyze numeric columns
        for col in analysis['numeric_columns']:
            if df[col].count() > 0:  # Has non-null values
                analysis['suitable_for_histogram'].append(col)
                if len(analysis['numeric_columns']) > 1:
                    analysis['suitable_for_scatter'].append(col)
        
        return analysis

    def _create_generic_chart(self, query: str, df) -> str:
        """
        Create charts based on generic data analysis, not specific to any domain
        """
        data_analysis = self._analyze_data_structure(df)
        query_lower = query.lower()
        
        print(f"[ChartingAgent] Data analysis: {data_analysis}")
        
        # Look for distribution/status related queries
        distribution_keywords = ['status', 'distribution', 'breakdown', 'categories', 'types', 'different']
        if any(keyword in query_lower for keyword in distribution_keywords):
            # Find the best categorical column for distribution
            if data_analysis['suitable_for_pie']:
                best_col = data_analysis['suitable_for_pie'][0]
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": "pie",
                    "x_column": best_col,
                    "title": f"Distribution of {best_col}",
                    "top_n": 10
                })
                return f"I found a categorical column '{best_col}' and created a distribution chart:\n\n{result}"
            elif data_analysis['suitable_for_bar']:
                best_col = data_analysis['suitable_for_bar'][0]
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": "bar",
                    "x_column": best_col,
                    "title": f"Distribution of {best_col}",
                    "top_n": 15
                })
                return f"I found a categorical column '{best_col}' and created a distribution chart:\n\n{result}"
        
        # Look for top/highest/lowest queries
        ranking_keywords = ['top', 'highest', 'lowest', 'best', 'worst', 'largest', 'smallest']
        if any(keyword in query_lower for keyword in ranking_keywords):
            if data_analysis['numeric_columns'] and data_analysis['categorical_columns']:
                numeric_col = data_analysis['numeric_columns'][0]
                cat_col = data_analysis['categorical_columns'][0]
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": "bar",
                    "x_column": cat_col,
                    "y_column": numeric_col,
                    "title": f"Top Values: {cat_col} by {numeric_col}",
                    "top_n": 10
                })
                return f"I created a ranking chart using '{cat_col}' and '{numeric_col}':\n\n{result}"
        
        # Fallback: suggest what's available in the dataset
        suggestions = []
        if data_analysis['suitable_for_pie']:
            suggestions.append(f"Pie charts for: {data_analysis['suitable_for_pie'][:3]}")
        if data_analysis['suitable_for_histogram']:
            suggestions.append(f"Histograms for: {data_analysis['suitable_for_histogram'][:3]}")
        if data_analysis['numeric_columns'] and data_analysis['categorical_columns']:
            suggestions.append(f"Bar charts comparing {data_analysis['categorical_columns'][0]} vs {data_analysis['numeric_columns'][0]}")
        
        if suggestions:
            return f"I analyzed your dataset and found these visualization possibilities:\n\n" + "\n".join(f"• {s}" for s in suggestions) + f"\n\nPlease specify which type of chart you'd like to see, or I can create a default visualization based on your data structure."
        else:
            return f"This dataset has {len(df.columns)} columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}. Please specify which columns you'd like to visualize and what type of chart you prefer."
    
    def _direct_chart_fallback(self, query: str, df) -> str:
        """
        Direct chart creation fallback when React agent hits token limits
        """
        query_lower = query.lower()
        columns = list(df.columns)
        
        print(f"[ChartingAgent] Fallback processing: {query}")
        print(f"[ChartingAgent] Available columns: {columns[:10]}")
        
        # Use generic data analysis instead of hardcoded keywords
        data_analysis = self._analyze_data_structure(df)
        
        try:
            # Handle distribution/status queries generically
            if any(word in query_lower for word in ['status', 'distribution', 'different', 'breakdown', 'categories']):
                return self._create_generic_chart(query, df)
            
            # Handle ranking queries generically  
            elif any(word in query_lower for word in ['top', 'highest', 'lowest', 'best', 'worst', 'ranking', 'compare']):
                if data_analysis['numeric_columns'] and data_analysis['categorical_columns']:
                    result = create_chart_from_uploaded_data.invoke({
                        "chart_type": "bar",
                        "x_column": data_analysis['categorical_columns'][0],
                        "y_column": data_analysis['numeric_columns'][0],
                        "title": f"Top Values: {data_analysis['categorical_columns'][0]} by {data_analysis['numeric_columns'][0]}",
                        "top_n": 10
                    })
                    return f"I created a ranking chart:\n\n{result}"
                else:
                    return "I need both categorical and numeric columns to create ranking charts."
            
            # Handle correlation/relationship queries
            elif any(word in query_lower for word in ['correlation', 'relationship', 'scatter', 'compare']):
                if len(data_analysis['numeric_columns']) >= 2:
                    result = create_chart_from_uploaded_data.invoke({
                        "chart_type": "scatter",
                        "x_column": data_analysis['numeric_columns'][0],
                        "y_column": data_analysis['numeric_columns'][1],
                        "title": f"Relationship: {data_analysis['numeric_columns'][0]} vs {data_analysis['numeric_columns'][1]}"
                    })
                    return f"I created a scatter plot to show relationships:\n\n{result}"
                else:
                    return "I need at least 2 numeric columns to create scatter plots or correlation charts."
            
            # Handle distribution queries for numeric data
            elif any(word in query_lower for word in ['distribution', 'histogram', 'spread']):
                if data_analysis['suitable_for_histogram']:
                    result = create_chart_from_uploaded_data.invoke({
                        "chart_type": "histogram",
                        "x_column": data_analysis['suitable_for_histogram'][0],
                        "title": f"Distribution of {data_analysis['suitable_for_histogram'][0]}"
                    })
                    return f"I created a histogram showing the distribution:\n\n{result}"
                else:
                    return "I need numeric columns to create distribution charts."
            
            # Handle general chart/graph requests
            elif any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize', 'show']):
                return self._create_generic_chart(query, df)
            
            # Default: suggest available columns and chart types
            else:
                return self._create_generic_chart(query, df)
                       
        except Exception as e:
            return f"Error creating chart: {e}. Available columns: {columns[:10]}"

@tool
def generate_and_execute_chart_code(user_request: str, chart_description: str = "") -> str:
    """Generate Python plotting code on the fly based on user request and execute it.
    
    user_request: The original user request for the chart
    chart_description: Optional description of what chart to create
    """
    try:
        # Get the current dataframe
        try:
            from .pandas_agent import df_manager as pandas_df_manager
            df = pandas_df_manager.get_current_dataframe()
        except:
            df = df_manager.get_current_dataframe()
            
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Analyze the dataframe structure
        columns = list(df.columns)
        dtypes = df.dtypes.to_dict()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Sample data for large dataframes
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
            sample_note = f"(Using 1000 random samples from {len(df)} total rows)"
        else:
            df_sample = df
            sample_note = ""
        
        # Generate intelligent plotting code using LLM
        code_generation_prompt = f"""
You are an expert Python data visualization programmer. Generate matplotlib/seaborn code to create a chart based on the user's request.

USER REQUEST: "{user_request}"
CHART DESCRIPTION: "{chart_description}"

AVAILABLE DATA:
- Dataset shape: {df.shape}
- Columns: {columns}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Data types: {dtypes}

REQUIREMENTS:
1. Write clean, executable Python code using matplotlib and/or seaborn
2. The dataframe is already available as 'df'
3. Include proper error handling
4. Use appropriate chart types for the data
5. Add titles, labels, and formatting
6. Handle large datasets appropriately (sampling, limiting categories)
7. Return the chart as base64 encoded image
8. Close the plot after saving to prevent memory issues

TEMPLATE STRUCTURE:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO

# Your chart generation code here
plt.figure(figsize=(10, 6))

# Chart creation logic
# ...

plt.title("Your Chart Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.tight_layout()

# Save to base64
buf = BytesIO()
plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

result = f"data:image/png;base64,{{img_base64}}"
```

Generate ONLY the Python code, no explanations. Make it intelligent and adaptive to the specific request.
"""

        # Get the LLM to generate the code
        from langchain_openai import ChatOpenAI
        import os
        llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=os.environ.get('OPENAI_API_KEY'))
        
        code_response = llm.invoke(code_generation_prompt)
        generated_code = code_response.content if hasattr(code_response, 'content') else str(code_response)
        
        # Clean the code (remove markdown formatting)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0]
            
        generated_code = generated_code.strip()
        
        print(f"[ChartingAgent] Generated code length: {len(generated_code)} characters")
        print(f"[ChartingAgent] Code preview: {generated_code[:200]}...")
        
        # Execute the generated code
        local_vars = {
            'df': df_sample,
            'plt': plt,
            'sns': sns,
            'pd': pd,
            'np': np,
            'base64': base64,
            'BytesIO': BytesIO
        }
        
        exec(generated_code, {}, local_vars)
        
        # Get the result
        if 'result' in local_vars:
            return f"I generated and executed custom plotting code for your request. {sample_note}\n\n{local_vars['result']}"
        else:
            return "Chart code was executed but no result was returned. The chart may have been displayed."
            
    except Exception as e:
        print(f"[ChartingAgent] Dynamic code generation error: {e}")
        return f"Error generating dynamic chart code: {e}"

@tool
def generate_dynamic_chart(query: str, chart_requirements: str = "") -> str:
    """Generate and execute custom plotting code based on user requirements.
    
    query: The user's visualization request
    chart_requirements: Additional specific requirements or chart type preferences
    """
    try:
        # Get the current dataframe
        try:
            from .pandas_agent import df_manager as pandas_df_manager
            df = pandas_df_manager.get_current_dataframe()
        except:
            df = df_manager.get_current_dataframe()
            
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Create a safe execution environment
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from io import BytesIO
        import base64
        
        # Sample large dataframes
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
            sample_notice = " (Using 1000 sample rows)"
        else:
            df_sample = df.copy()
            sample_notice = ""
        
        # Create the code generation prompt
        columns_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'string']).columns.tolist(),
            'sample_data': df.head(3).to_dict()
        }
        
        code_prompt = f"""Generate Python code to create a visualization for this request: "{query}"

Dataset Information:
- Shape: {columns_info['shape']}
- Columns: {columns_info['columns']}
- Numeric columns: {columns_info['numeric_columns']}
- Categorical columns: {columns_info['categorical_columns']}

Requirements:
{chart_requirements}

Generate Python code that:
1. Uses the variable 'df_sample' (already available)
2. Creates an appropriate matplotlib/seaborn visualization
3. Sets a descriptive title
4. Handles data preprocessing if needed (filtering, grouping, etc.)
5. Uses plt.figure(figsize=(10, 6)) for consistent sizing
6. Includes proper labels and formatting
7. Uses plt.tight_layout() before saving

Important:
- Only return the Python code, no explanations
- Use df_sample as the dataframe variable
- The code should be ready to execute
- Handle any data type conversions needed
- For top-N queries, use .nlargest() or .nsmallest()
- For categorical data, consider using .value_counts()

Example structure:
```python
plt.figure(figsize=(10, 6))
# Your visualization code here
plt.title('Your Title Here')
plt.tight_layout()
```"""

        # Get the generated code from LLM
        code_response = df_manager._ChartingAgent__llm.invoke(code_prompt)
        
        if hasattr(code_response, 'content'):
            generated_code = code_response.content
        else:
            generated_code = str(code_response)
        
        # Extract Python code from response (remove markdown formatting if present)
        import re
        code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
        if code_match:
            code_to_execute = code_match.group(1)
        else:
            # If no markdown formatting, use the entire response
            code_to_execute = generated_code.strip()
        
        print(f"[ChartingAgent] Generated code: {code_to_execute[:200]}...")
        
        # Create execution environment
        exec_globals = {
            'df_sample': df_sample,
            'df': df_sample,  # Alias for convenience
            'plt': plt,
            'sns': sns,
            'pd': pd,
            'np': np
        }
        
        # Execute the generated code
        exec(code_to_execute, exec_globals)
        
        # Save the plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"I generated custom code to create your visualization{sample_notice}:\n\n```python\n{code_to_execute}\n```\n\ndata:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"[ChartingAgent] Dynamic code generation error: {e}")
        return f"Error generating dynamic chart: {e}. Let me try a different approach."


