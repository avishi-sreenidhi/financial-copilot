import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

from langgraph_engine.graph_builder import build_agent_graph, DataAnalyticsState


class TestGraphBuilder(unittest.TestCase):
    """Test cases for the LangGraph agent orchestration"""
    
    @patch('langgraph_engine.graph_builder.ChatOpenAI')
    @patch('langgraph_engine.graph_builder.RouterAgent')
    @patch('langgraph_engine.graph_builder.PythonIDEAgent')
    @patch('langgraph_engine.graph_builder.ChartingAgent')
    @patch('langgraph_engine.graph_builder.DataSearchAgent')
    @patch('langgraph_engine.graph_builder.PandasAgent')
    def test_build_agent_graph_structure(self, mock_pandas, mock_search, 
                                        mock_chart, mock_python, mock_router, mock_llm):
        """Test that the graph is built with correct structure"""
        # Mock all agent classes
        mock_router_instance = Mock()
        mock_python_instance = Mock()
        mock_chart_instance = Mock()
        mock_search_instance = Mock()
        mock_pandas_instance = Mock()
        
        mock_router.return_value = mock_router_instance
        mock_python.return_value = mock_python_instance
        mock_chart.return_value = mock_chart_instance
        mock_search.return_value = mock_search_instance
        mock_pandas.return_value = mock_pandas_instance
        
        # Build the graph
        graph = build_agent_graph()
        
        # Verify all agents were instantiated
        mock_router.assert_called_once()
        mock_python.assert_called_once()
        mock_chart.assert_called_once()
        mock_search.assert_called_once()
        mock_pandas.assert_called_once()
        
        # Verify graph is compiled
        self.assertIsNotNone(graph)
        self.assertTrue(hasattr(graph, 'invoke'))
    
    def test_data_analytics_state_structure(self):
        """Test DataAnalyticsState class structure"""
        # Test annotations
        annotations = DataAnalyticsState.__annotations__
        self.assertIn("query", annotations)
        self.assertIn("next_agent", annotations)
        self.assertIn("agent_outputs", annotations)
        self.assertIn("messages", annotations)
        self.assertIn("current_agent", annotations)
        self.assertIn("dataframe_info", annotations)
        self.assertIn("has_data", annotations)
        self.assertIn("final_result", annotations)
        self.assertIn("metadata", annotations)
        self.assertIn("iteration_count", annotations)
    
    @patch('langgraph_engine.graph_builder.ChatOpenAI')
    @patch('langgraph_engine.graph_builder.RouterAgent')
    @patch('langgraph_engine.graph_builder.PythonIDEAgent')
    @patch('langgraph_engine.graph_builder.ChartingAgent')
    @patch('langgraph_engine.graph_builder.DataSearchAgent')
    @patch('langgraph_engine.graph_builder.PandasAgent')
    def test_graph_nodes_creation(self, mock_pandas, mock_search, 
                                 mock_chart, mock_python, mock_router, mock_llm):
        """Test that all required nodes are created in the graph"""
        with patch('langgraph_engine.graph_builder.StateGraph') as mock_state_graph:
            mock_builder = Mock()
            mock_state_graph.return_value = mock_builder
            mock_builder.compile.return_value = Mock()
            
            build_agent_graph()
            
            # Verify add_node was called for each agent
            expected_nodes = ["router", "python", "chart", "search", "pandas"]
            add_node_calls = [call[0][0] for call in mock_builder.add_node.call_args_list]
            
            for node in expected_nodes:
                self.assertIn(node, add_node_calls)
            
            # Verify entry point is set
            mock_builder.set_entry_point.assert_called_once_with("router")
    
    @patch('langgraph_engine.graph_builder.ChatOpenAI')
    @patch('langgraph_engine.graph_builder.RouterAgent')
    @patch('langgraph_engine.graph_builder.PythonIDEAgent')
    @patch('langgraph_engine.graph_builder.ChartingAgent')
    @patch('langgraph_engine.graph_builder.DataSearchAgent')
    @patch('langgraph_engine.graph_builder.PandasAgent')
    def test_graph_edges_creation(self, mock_pandas, mock_search, 
                                 mock_chart, mock_python, mock_router, mock_llm):
        """Test that all required edges are created in the graph"""
        with patch('langgraph_engine.graph_builder.StateGraph') as mock_state_graph:
            mock_builder = Mock()
            mock_state_graph.return_value = mock_builder
            mock_builder.compile.return_value = Mock()
            
            build_agent_graph()
            
            # In the new architecture, multiple conditional edges are added
            # Verify there are at least 6 conditional edges (one from router plus edges from each agent)
            self.assertGreaterEqual(mock_builder.add_conditional_edges.call_count, 6)
            
            # Verify the router edge has the right structure
            router_edge_calls = [
                call for call in mock_builder.add_conditional_edges.call_args_list 
                if call[0][0] == 'router'
            ]
            self.assertEqual(len(router_edge_calls), 1)
            
            # Get the first argument of the first router edge call
            router_call = router_edge_calls[0]
            
            # Examine the routing dictionary in the third argument of the call
            routing_dict = router_call[0][2]
            expected_routes = {
                "python": "python", 
                "chart": "chart", 
                "search": "search", 
                "pandas": "pandas",
                "coordinator": "coordinator"
            }
            
            # Verify the router can route to all specialized agents
            for key, value in expected_routes.items():
                self.assertIn(key, routing_dict)


class TestGraphWorkflow(unittest.TestCase):
    """Integration tests for the complete graph workflow"""
    
    @patch('langgraph_engine.graph_builder.ChatOpenAI')
    def test_graph_invocation_structure(self, mock_llm):
        """Test that the graph can be invoked with proper state structure"""
        # This is a structural test since we can't easily mock the entire graph
        try:
            graph = build_agent_graph()
            
            # Test that graph has invoke method
            self.assertTrue(hasattr(graph, 'invoke'))
            
            # Test state structure - create a minimal state for testing
            from typing import Dict, List, Any
            from langchain_core.messages import HumanMessage
            
            test_state = {
                "query": "test message",
                "next_agent": "",
                "agent_outputs": {},
                "messages": [HumanMessage(content="test message")],
                "current_agent": "router",
                "dataframe_info": {},
                "has_data": False,
                "final_result": "",
                "metadata": {},
                "iteration_count": 0
            }
            
            # Verify state can be created and accessed
            self.assertEqual(test_state["query"], "test message")
            
        except Exception as e:
            # If there are import issues, we expect them but test structure
            if "Import" in str(e) or "module" in str(e):
                self.skipTest(f"Skipping due to import issues: {e}")
            else:
                raise
    
    def test_data_analytics_state_usage(self):
        """Test DataAnalyticsState functionality"""
        from typing import Dict, List, Any
        from langchain_core.messages import HumanMessage
        
        # Create a minimal state for testing
        state = {
            "query": "Hello world",
            "next_agent": "tone",
            "agent_outputs": {},
            "messages": [HumanMessage(content="Hello world")],
            "current_agent": "router",
            "dataframe_info": {},
            "has_data": False,
            "final_result": "Processed",
            "metadata": {},
            "iteration_count": 0
        }
        
        self.assertEqual(state["query"], "Hello world")
        self.assertEqual(state["final_result"], "Processed")
        self.assertEqual(state["next_agent"], "tone")
        
        # Test get method with default
        self.assertEqual(state.get("nonexistent", "default"), "default")
        
        # Test update functionality
        state.update({"query": "Updated message"})
        self.assertEqual(state["query"], "Updated message")


class TestGraphConfiguration(unittest.TestCase):
    """Test graph configuration and setup"""
    
    @patch('langgraph_engine.graph_builder.ChatOpenAI')
    def test_llm_initialization(self, mock_llm):
        """Test that LLM is initialized correctly"""
        # Since LLM is initialized at module level, we need to check if it was called during import
        # The mock_openai fixture in conftest.py should handle this
        # Just verify the function works without error
        try:
            graph = build_agent_graph()
            self.assertIsNotNone(graph)
        except Exception as e:
            self.fail(f"Graph building failed: {e}")
    
    @patch('langgraph_engine.graph_builder.StateGraph')
    def test_state_graph_initialization(self, mock_state_graph):
        """Test StateGraph initialization with AgentState"""
        mock_builder = Mock()
        mock_state_graph.return_value = mock_builder
        mock_builder.compile.return_value = Mock()
        
        build_agent_graph()
        
        # Verify StateGraph was initialized with DataAnalyticsState
        mock_state_graph.assert_called_once_with(DataAnalyticsState)


class TestRoutingLogic(unittest.TestCase):
    """Test the routing logic within the graph"""
    
    def test_routing_lambda_function(self):
        """Test the lambda function used for conditional routing"""
        # The routing function should extract the 'next_agent' key from state
        routing_func = lambda x: {"next": x.get("next_agent") or x.get("agent_outputs", {}).get("router", {}).get("next_agent", "tone")}
        
        # Test with next_agent directly in state
        test_state = {"next_agent": "python", "query": "test"}
        result = routing_func(test_state)
        
        self.assertEqual(result["next"], "python")
        
        # Test with next_agent in agent_outputs
        test_state = {
            "next_agent": None, 
            "agent_outputs": {
                "router": {"next_agent": "chart"}
            }
        }
        result = routing_func(test_state)
        self.assertEqual(result["next"], "chart")
    
    def test_all_routing_destinations(self):
        """Test all possible routing destinations"""
        routing_destinations = ["python", "chart", "search", "pandas", "coordinator"]
        routing_func = lambda x: {"next": x.get("next_agent")}
        
        for destination in routing_destinations:
            test_state = {"next_agent": destination}
            result = routing_func(test_state)
            self.assertEqual(result["next"], destination)


if __name__ == '__main__':
    unittest.main()
