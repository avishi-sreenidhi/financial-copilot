# Finance Multi-Agent Analytics System

A robust, intelligent multi-agent system for comprehensive **financial data analytics** with context-aware query routing, dynamic chart generation, and flexible financial data exploration. Built with LangGraph, LangChain, and Streamlit.

## Overview

This platform leverages a sophisticated swarm of specialized agents, orchestrated to deliver advanced **financial analytics** for any structured dataset:

* **Router Agent**: Intelligently classifies and routes financial queries to the right specialist agents

* **Query Context Agent**: Expands financial abbreviations, maps terms to columns, and provides context hints

* **Memory Agent**: Maintains financial conversation context and session management

* **Pandas Agent**: Performs DataFrame operations, statistics, and financial data analysis

* **Charting Agent**: Creates dynamic financial visualizations with LLM-driven code generation

* **Data Search Agent**: Context-aware searching, filtering, and financial data exploration

* **Python IDE Agent**: Executes custom Python code for advanced financial analysis

* **Coordinator Agent**: Orchestrates multi-agent workflows and aggregates results

## Key Features

### 🧠 Intelligent Financial Query Understanding

* **Context-Aware Routing**: Automatically routes financial queries to the most appropriate agent(s)

* **Abbreviation Expansion**: Understands financial terms like "EBITDA", "ROI", "COGS", etc.

* **Column Mapping**: Maps query terms to actual financial dataset columns

* **Domain Agnostic**: Works with any financial CSV/Excel dataset structure

### 📊 Dynamic Financial Visualization

* **LLM-Driven Chart Generation**: Creates custom Python code for any financial visualization request

* **Adaptive Chart Types**: Analyzes financial data structure to suggest appropriate charts

* **Base64 Image Output**: Seamless integration with web interfaces

* **Chart Types**: Bar, line, scatter, histogram, box, pie, heatmap, violin plots, and custom financial visualizations

### 🔍 Powerful Financial Data Exploration

* **Context-Enhanced Search**: Uses query context for more accurate financial results

* **Flexible Filtering**: Supports all comparison operators and text matching for financial data

* **Statistical Analysis**: Comprehensive DataFrame operations and financial insights

* **Multi-Step Reasoning**: Chains multiple operations for complex financial analysis

### 💭 Conversation Memory

* **Session Persistence**: Maintains financial context across interactions

* **Chat-Style Interface**: Natural conversation flow for financial analysis

* **Query History**: Learns from previous financial queries

* **Multi-Turn Conversations**: Supports follow-up questions and refinements

## Quick Start

### 1\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2\. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3\. Run the Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

### 4\. Or Use Terminal Chat Interface

```bash
python chat_interface.py
```

### 5\. Or Run Demo Conversations

```bash
python demo_chat.py
```

## App

## Agent Architecture & Capabilities

### 🎯 Router Agent

* **Intent Classification**: Analyzes financial query to determine appropriate routing

* **Multi-Agent Coordination**: Can route to multiple agents for complex financial queries

* **Fallback Handling**: Graceful handling of ambiguous or unclear financial requests

### 🧠 Query Context Agent

* **Abbreviation Expansion**: e.g., EBITDA → earnings before interest, taxes, depreciation, and amortization

* **Column Mapping**: Maps query terms to actual financial dataset columns

* **Context Hints**: Provides guidance for downstream agents

* **Domain Analysis**: Understands financial dataset structure and content

### 💾 Memory & Session Management

* **Conversation Context**: Maintains financial chat history and context

* **Session Persistence**: Tracks user interactions across sessions

* **Context Formatting**: Prepares responses for UI display

### 🐼 Pandas Agent

* **DataFrame Operations**: Advanced pandas operations and financial transformations

* **Statistical Analysis**: Descriptive statistics, correlations, aggregations for financial data

* **Data Quality**: Missing value analysis, data type validation

* **Performance Optimization**: Efficient operations on large financial datasets

### 📈 Charting Agent

* **Dynamic Code Generation**: LLM creates custom Python plotting code for financial charts

* **Intelligent Chart Selection**: Analyzes financial data to suggest appropriate visualizations

* **Fallback Charts**: Predefined chart types for reliability

* **Custom Styling**: Professional financial styling and formatting

### 🔍 Data Search Agent

* **Context-Enhanced Search**: Uses QueryContext for improved accuracy on financial data

* **Advanced Filtering**: Multiple operators and complex conditions

* **Smart Summaries**: Intelligent financial data summaries and insights

* **JSON Responses**: Structured output for programmatic use

### 🐍 Python IDE Agent

* **Code Execution**: Safe execution of custom Python code for financial analysis

* **Library Access**: pandas, numpy, matplotlib, seaborn pre-loaded

* **Debugging Support**: Error handling and code analysis

* **Custom Analysis**: User-defined financial data transformations and calculations

## Example Financial Queries

### Smart Context Understanding

* "Which company had the highest EBITDA in 2023?"

* "Show me top revenue by quarter"

* "What's the average net income by year?"

* "Calculate the debt-to-equity ratio for all companies"

### Dynamic Visualizations

* "Create a line chart of revenue trends by company"

* "Show net income vs operating expenses as a scatter plot"

* "Make a stacked bar chart of cash flow by year"

### Data Exploration

* "Find records where net income > 2,000,000 and year = 2022"

* "Show me a summary of all financial metrics"

* "What are the correlations between all numeric financial columns?"

### Conversation Flow

* User: "Show me revenue trends for Company A"

* System: _Creates line chart_

* User: "Now filter for 2022 only"

* System: _Remembers context and filters previous analysis_

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                STREAMLIT FINANCE WEB INTERFACE              │
│         File Upload | Chat Input | Results Display         │
└─────────────────────┬───────────────────────────────────────┘
                      │ User Query
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                       │
│    ┌─────────────┐              ┌─────────────────┐        │
│    │ROUTER AGENT │◄────────────►│COORDINATOR AGENT│        │
│    │Query Intent │              │Workflow Mgmt    │        │
│    │Classification│              │Result Aggregation│        │
│    └─────────────┘              └─────────────────┘        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                CONTEXT & MEMORY LAYER                      │
│ ┌─────────────┐┌─────────────┐┌─────────────┐┌───────────┐ │
│ │QUERY CONTEXT││MEMORY AGENT ││CHAT FORMAT  ││TOOL EXEC  │ │
│ │Abbreviation ││Conversation ││Response     ││Tool Route │ │
│ │Expansion    ││Context      ││Formatting   ││Execution  │ │
│ └─────────────┘└─────────────┘└────────────┘└───────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               SPECIALIZED AGENTS LAYER                     │
│┌────────────┐┌────────────┐┌────────────┐┌──────────────┐  │
││DATA SEARCH ││PANDAS AGENT││CHARTING    ││PYTHON IDE    │  │
││Text Search ││DataFrame   ││Dynamic     ││Code Execution│  │
││Filtering   ││Operations  ││Charts      ││Analysis      │  │
││Context-    ││Statistics  ││LLM Code    ││Debugging     │  │
││Aware       ││Analysis    ││Generation  ││              │  │
│└────────────┘└────────────┘└────────────┘└──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA & OUTPUT LAYER                      │
│ ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│ │ DataFrame Mgmt  │ │ Visualizations  │ │ LLM Backend   │ │
│ │ CSV/Excel       │ │ Base64 Images   │ │ GPT-4o-mini   │ │
│ │ Session State   │ │ JSON Results    │ │ Intelligence  │ │
│ └─────────────────┘ └─────────────────┘ └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
finance_agent_analytics/
├── src/
│   ├── agents/                 # All agent implementations
│   │   ├── router_agent.py     # Query routing and classification
│   │   ├── query_context_agent.py  # Context analysis and expansion
│   │   ├── memory_agent.py     # Conversation memory management
│   │   ├── charting_agent.py   # Dynamic visualization generation
│   │   ├── data_search_agent.py # Context-aware data search
│   │   ├── pandas_agent.py     # DataFrame operations
│   │   └── python_ide_agent.py # Code execution
│   ├── langgraph_engine/       # Workflow orchestration
│   │   └── graph_builder.py    # Agent graph construction
│   ├── data/                   # Sample financial datasets
│   │   └── sample_finance.csv  # Financial sample data
│   └── api/                    # FastAPI backend (optional)
├── tests/                      # Comprehensive test suite
├── streamlit_app.py           # Web interface
├── chat_interface.py          # Terminal chat interface
├── demo_chat.py              # Demo conversations
├── architecture_ascii.txt     # System architecture diagram
└── requirements.txt          # Dependencies
```
