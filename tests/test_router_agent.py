import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.router_agent import RouterAgent
from typing import Dict, Any, TypedDict, List
from langchain_core.messages import HumanMessage


class TestRouterAgent(unittest.TestCase):
    """Test cases for the Router Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.router_agent = RouterAgent(self.mock_llm)
    
    def test_init(self):
        """Test RouterAgent initialization"""
        self.assertIsNotNone(self.router_agent.llm)
        self.assertIsNotNone(self.router_agent.prompt)
    
    def test_invoke_with_query(self):
        """Test invoke with query directly in state"""
        # Create a state with query
        state = {
            "query": "I want to create a bar chart of sales data",
            "next_agent": "",
            "agent_outputs": {},
            "messages": [],
            "current_agent": "",
            "dataframe_info": {},
            "has_data": True,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0
        }
        
        result = self.router_agent.invoke(state)
        
        # Check structure of result
        self.assertIn("next_agent", result)
        self.assertIn("agent_outputs", result)
        self.assertIn("router", result["agent_outputs"])
        self.assertIn("next_agent", result["agent_outputs"]["router"])
        self.assertIn("status", result["agent_outputs"]["router"])
        self.assertIn("result", result["agent_outputs"]["router"])
        self.assertIn("reasoning", result["agent_outputs"]["router"])
        
        # Given the message about charts, should route to chart agent
        self.assertEqual(result["next_agent"], "chart")
        self.assertEqual(result["agent_outputs"]["router"]["next_agent"], "chart")
    
    def test_invoke_with_messages(self):
        """Test invoke with message in messages list"""
        # Create a state with messages
        state = {
            "query": "",
            "next_agent": "",
            "agent_outputs": {},
            "messages": [HumanMessage(content="Hello, can you help me with Python code?")],
            "current_agent": "",
            "dataframe_info": {},
            "has_data": False,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0
        }
        
        result = self.router_agent.invoke(state)
        
        # Check routing for Python question
        self.assertEqual(result["next_agent"], "python")
        self.assertEqual(result["agent_outputs"]["router"]["next_agent"], "python")
    
    def test_invoke_conversation(self):
        """Test invoke with conversation message"""
        # Create a state with conversational query
        state = {
            "query": "Hello there! How are you today?",
            "next_agent": "",
            "agent_outputs": {},
            "messages": [],
            "current_agent": "",
            "dataframe_info": {},
            "has_data": False,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0
        }
        
        result = self.router_agent.invoke(state)
        
        # Should route to pandas agent for conversations
        self.assertEqual(result["next_agent"], "pandas")
        self.assertEqual(result["agent_outputs"]["router"]["next_agent"], "pandas")
    
    def test_invoke_data_analysis(self):
        """Test invoke with data analysis query"""
        # Create a state with data analysis query
        state = {
            "query": "Analyze the sales dataframe and show me the summary statistics",
            "next_agent": "",
            "agent_outputs": {},
            "messages": [],
            "current_agent": "",
            "dataframe_info": {"has_data": True},
            "has_data": True,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0
        }
        
        result = self.router_agent.invoke(state)
        
        # Should route to pandas agent for data analysis
        self.assertEqual(result["next_agent"], "pandas")
        self.assertEqual(result["agent_outputs"]["router"]["next_agent"], "pandas")
    
    def test_invoke_search_query(self):
        """Test invoke with search query"""
        # Create a state with search query
        state = {
            "query": "Find all records where sales are over 1000",
            "next_agent": "",
            "agent_outputs": {},
            "messages": [],
            "current_agent": "",
            "dataframe_info": {},
            "has_data": True,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0
        }
        
        result = self.router_agent.invoke(state)
        
        # Should route to search agent for search queries
        self.assertEqual(result["next_agent"], "search")
        self.assertEqual(result["agent_outputs"]["router"]["next_agent"], "search")


if __name__ == '__main__':
    unittest.main()
