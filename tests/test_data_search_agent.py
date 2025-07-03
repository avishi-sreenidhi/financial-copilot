import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.data_search_agent import DataSearchAgent, search_data, filter_data, get_data_summary


class TestDataSearchAgent(unittest.TestCase):
    """Test cases for DataSearchAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.search_agent = DataSearchAgent(self.mock_llm)
    
    def test_init(self):
        """Test DataSearchAgent initialization"""
        self.assertIsNotNone(self.search_agent.llm)
        self.assertIsNotNone(self.search_agent.tools)
        self.assertIsNotNone(self.search_agent.prompt)
        self.assertIsNotNone(self.search_agent.agent)
        self.assertIsNotNone(self.search_agent.agent_executor)
        self.assertEqual(len(self.search_agent.tools), 3)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_search_data_found(self, mock_read_csv):
        """Test successful data search with results"""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'revenue': [1000, 1200, 1100],
            'expenses': [500, 600, 550]
        })
        mock_read_csv.return_value = mock_df
        
        result = search_data.invoke({"query": "1200"})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['found_rows'], 1)
        self.assertIn('2023-01-02', str(parsed_result['data']))
        self.assertIn("Found 1 rows matching '1200'", parsed_result['summary'])
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_search_data_not_found(self, mock_read_csv):
        """Test data search with no results"""
        mock_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'revenue': [1000, 1200],
            'expenses': [500, 600]
        })
        mock_read_csv.return_value = mock_df
        
        result = search_data.invoke({"query": "nonexistent"})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['found_rows'], 0)
        self.assertEqual(parsed_result['data'], [])
        self.assertIn("No data found matching 'nonexistent'", parsed_result['summary'])
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_search_data_case_insensitive(self, mock_read_csv):
        """Test case-insensitive search"""
        mock_df = pd.DataFrame({
            'product': ['Widget A', 'Widget B', 'Gadget C'],
            'sales': [100, 200, 150]
        })
        mock_read_csv.return_value = mock_df
        
        result = search_data.invoke({"query": "WIDGET"})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['found_rows'], 2)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_search_data_error_handling(self, mock_read_csv):
        """Test error handling in search"""
        mock_read_csv.side_effect = Exception("File not found")
        
        result = search_data.invoke({"query": "test"})
        self.assertIn("Error searching data:", result)
        self.assertIn("File not found", result)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_filter_data_greater_than(self, mock_read_csv):
        """Test filtering with greater than operator"""
        mock_df = pd.DataFrame({
            'revenue': [1000, 1200, 1100, 1300],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
        })
        mock_read_csv.return_value = mock_df
        
        result = filter_data.invoke({"column": "revenue", "operator": ">", "value": "1150"})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['filtered_rows'], 2)
        self.assertIn("Found 2 rows where revenue > 1150.0", parsed_result['summary'])
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_filter_data_equals(self, mock_read_csv):
        """Test filtering with equals operator"""
        mock_df = pd.DataFrame({
            'status': ['active', 'inactive', 'active', 'pending'],
            'id': [1, 2, 3, 4]
        })
        mock_read_csv.return_value = mock_df
        
        result = filter_data.invoke({"column": "status", "operator": "==", "value": "active"})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['filtered_rows'], 2)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_filter_data_contains(self, mock_read_csv):
        """Test filtering with contains operator"""
        mock_df = pd.DataFrame({
            'description': ['Product A description', 'Product B info', 'Service A description'],
            'id': [1, 2, 3]
        })
        mock_read_csv.return_value = mock_df
        
        result = filter_data.invoke({"column": "description", "operator": "contains", "value": "description"})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['filtered_rows'], 2)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_filter_data_invalid_column(self, mock_read_csv):
        """Test filtering with invalid column name"""
        mock_df = pd.DataFrame({
            'revenue': [1000, 1200],
            'date': ['2023-01-01', '2023-01-02']
        })
        mock_read_csv.return_value = mock_df
        
        result = filter_data.invoke({"column": "nonexistent_column", "operator": ">", "value": "500"})
        self.assertIn("Column 'nonexistent_column' not found", result)
        self.assertIn("Available columns:", result)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_filter_data_unsupported_operator(self, mock_read_csv):
        """Test filtering with unsupported operator"""
        mock_df = pd.DataFrame({'revenue': [1000, 1200]})
        mock_read_csv.return_value = mock_df
        
        result = filter_data.invoke({"column": "revenue", "operator": "unsupported_op", "value": "1000"})
        self.assertIn("Unsupported operator: unsupported_op", result)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_filter_data_error_handling(self, mock_read_csv):
        """Test error handling in filter"""
        mock_read_csv.side_effect = Exception("File error")
        
        result = filter_data.invoke({"column": "column", "operator": ">", "value": "100"})
        self.assertIn("Error filtering data:", result)
        self.assertIn("File error", result)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_get_data_summary_complete(self, mock_read_csv):
        """Test comprehensive data summary"""
        mock_df = Mock()
        mock_df.shape = (3, 3)
        mock_df.columns = ['revenue', 'date', 'region']
        mock_df.dtypes = pd.Series({'revenue': 'int64', 'date': 'object', 'region': 'object'})
        mock_df.isnull.return_value.sum.return_value = pd.Series({'revenue': 0, 'date': 0, 'region': 0})
        mock_df.head.return_value.to_dict.return_value = [{'revenue': 1000, 'date': '2023-01-01', 'region': 'North'}]
        
        # Mock numeric column selection
        numeric_cols_mock = Mock()
        numeric_cols_mock.__len__ = Mock(return_value=1)  # len(numeric_cols) > 0
        numeric_cols_mock.__iter__ = Mock(return_value=iter(['revenue']))  # for iteration
        mock_df.select_dtypes.return_value.columns = numeric_cols_mock
        
        mock_read_csv.return_value = mock_df
        
        # Mock numeric dataframe and statistics  
        numeric_df = Mock()
        numeric_df.describe.return_value.to_dict.return_value = {'revenue': {'mean': 1100, 'std': 100}}
        mock_df.__getitem__ = Mock(return_value=numeric_df)  # df[numeric_cols]
        
        mock_read_csv.return_value = mock_df
        
        result = get_data_summary.invoke({})
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result['shape'], {'rows': 3, 'columns': 3})
        self.assertEqual(parsed_result['columns'], ['revenue', 'date', 'region'])
        self.assertIn('statistics', parsed_result)
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_get_data_summary_error_handling(self, mock_read_csv):
        """Test error handling in data summary"""
        mock_read_csv.side_effect = Exception("Cannot read file")
        
        result = get_data_summary.invoke({})
        self.assertIn("Error getting data summary:", result)
        self.assertIn("Cannot read file", result)
    
    def test_invoke_structure(self):
        """Test the invoke method structure"""
        with patch('langchain.agents.AgentExecutor.invoke') as mock_invoke:
            mock_invoke.return_value = {"output": "search results"}
            
            # Create a state that matches the new DataAnalyticsState structure
            state = {
                "query": "find revenue data",
                "next_agent": "",
                "agent_outputs": {},
                "messages": [],
                "current_agent": "search",
                "dataframe_info": {},
                "has_data": True,
                "final_result": "",
                "metadata": {},
                "iteration_count": 0
            }
            
            result = self.search_agent.invoke(state)
            
            # Check that the agent_outputs is populated correctly
            self.assertIn("search", result["agent_outputs"])
            self.assertEqual(result["agent_outputs"]["search"]["result"], "search results")
            self.assertEqual(result["next_agent"], "tone")
            
            # Verify the executor was called with the right input
            mock_invoke.assert_called_once_with({"input": "find revenue data"})


class TestDataSearchTools(unittest.TestCase):
    """Test cases for individual data search tools"""
    
    @patch('agents.data_search_agent.pd.read_csv')
    def test_multiple_operators_numeric(self, mock_read_csv):
        """Test various numeric operators"""
        mock_df = pd.DataFrame({
            'score': [85, 92, 78, 95, 88],
            'name': ['A', 'B', 'C', 'D', 'E']
        })
        mock_read_csv.return_value = mock_df
        
        test_cases = [
            ('>', '90', 2),  # 92, 95
            ('<', '80', 1),  # 78
            ('>=', '88', 3), # 88, 92, 95
            ('<=', '85', 2), # 85, 78
            ('==', '88', 1), # 88
            ('!=', '88', 4)  # all except 88
        ]
        
        for operator, value, expected_count in test_cases:
            with self.subTest(operator=operator, value=value):
                result = filter_data.invoke({
                    "column": "score", 
                    "operator": operator, 
                    "value": value
                })
                parsed_result = json.loads(result)
                self.assertEqual(parsed_result['filtered_rows'], expected_count)


if __name__ == '__main__':
    unittest.main()
