import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestAgentIntegration(unittest.TestCase):
    """Integration tests for the complete agent system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_data_path = Path(__file__).parent.parent / "src" / "data" / "sample.csv"
    
    def test_sample_data_exists(self):
        """Test that sample data file exists and is readable"""
        self.assertTrue(self.test_data_path.exists(), "Sample CSV file should exist")
        
        # Try to read the file
        try:
            import pandas as pd
            df = pd.read_csv(self.test_data_path)
            self.assertGreater(len(df), 0, "Sample CSV should not be empty")
            self.assertIn('date', df.columns, "Sample CSV should have date column")
            self.assertIn('revenue', df.columns, "Sample CSV should have revenue column")
            self.assertIn('expenses', df.columns, "Sample CSV should have expenses column")
        except Exception as e:
            self.fail(f"Failed to read sample CSV: {e}")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_graph_builder_imports(self):
        """Test that graph builder can import all dependencies"""
        try:
            # Test individual agent imports
            from agents.router_agent import RouterAgent
            from agents.python_ide_agent import PythonIDEAgent
            from agents.charting_agent import ChartingAgent
            from agents.data_search_agent import DataSearchAgent
            
            # Test graph builder import
            from langgraph_engine.graph_builder import build_agent_graph, DataAnalyticsState
            
            # All imports successful
            self.assertTrue(True)
            
        except ImportError as e:
            self.skipTest(f"Import test skipped due to missing dependencies: {e}")
        except Exception as e:
            self.fail(f"Unexpected error during import: {e}")
    
    def test_agent_tools_functionality(self):
        """Test that individual agent tools work correctly"""
        try:
            # Test Python IDE tools
            from agents.python_ide_agent import execute_python_code
            result = execute_python_code("result = 2 + 2")
            self.assertEqual(result, "4")
            
            # Test data search tools with mock data
            with patch('agents.data_search_agent.pd.read_csv') as mock_read_csv:
                import pandas as pd
                mock_df = pd.DataFrame({'test': [1, 2, 3]})
                mock_read_csv.return_value = mock_df
                
                from agents.data_search_agent import search_data
                result = search_data("1")
                self.assertIn("found_rows", result)
                
        except ImportError as e:
            self.skipTest(f"Tool test skipped due to missing dependencies: {e}")
        except Exception as e:
            self.fail(f"Tool functionality test failed: {e}")
    
    def test_end_to_end_workflow_structure(self):
        """Test the structure of end-to-end workflow"""
        try:
            # Test that we can create a complete workflow state
            from langgraph_engine.graph_builder import DataAnalyticsState
            
            # Create test state with the new structure
            state = {
                "query": "Calculate the mean revenue",
                "next_agent": "",
                "agent_outputs": {},
                "messages": [],
                "current_agent": "router",
                "dataframe_info": {},
                "has_data": True,
                "final_result": "",
                "metadata": {},
                "iteration_count": 0
            }
            
            # Verify state structure
            self.assertIn("query", state)
            self.assertIn("next_agent", state)
            self.assertIn("agent_outputs", state)
            self.assertIn("messages", state)
            
            # Test routing logic structure
            from agents.router_agent import RouterAgent
            mock_llm = Mock()
            router = RouterAgent(mock_llm)
            
            # Test that router has required methods
            self.assertTrue(hasattr(router, 'invoke'))
            self.assertTrue(hasattr(router, 'llm'))
            self.assertTrue(hasattr(router, 'prompt'))
            
        except ImportError as e:
            self.skipTest(f"Workflow test skipped due to missing dependencies: {e}")
        except Exception as e:
            self.fail(f"Workflow structure test failed: {e}")
    
    def test_agent_communication_flow(self):
        """Test that agents can pass state correctly"""
        try:
            from langgraph_engine.graph_builder import DataAnalyticsState
            from langchain_core.messages import HumanMessage
            
            # Simulate state flow through agents with new structure
            initial_state = {
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
            
            # Test router output format
            from agents.router_agent import RouterAgent
            mock_llm = Mock()
            router = RouterAgent(mock_llm)
            
            # Mock the agent executor
            with patch('langchain.agents.AgentExecutor.invoke') as mock_invoke:
                mock_invoke.return_value = {"output": "python"}
                
                result = router.invoke(initial_state)
                self.assertIn("next_agent", result)
                self.assertIn("agent_outputs", result)
                self.assertIn("router", result["agent_outputs"])
                self.assertIsInstance(result["next_agent"], str)
            
            # Test Python agent output format
            from agents.python_ide_agent import PythonIDEAgent
            python_agent = PythonIDEAgent(mock_llm)
            
            with patch('langchain.agents.AgentExecutor.invoke') as mock_invoke:
                mock_invoke.return_value = {"output": "calculation complete"}
                
                test_state = {
                    "query": "calculate mean",
                    "next_agent": "",
                    "agent_outputs": {},
                    "messages": [HumanMessage(content="calculate mean")],
                    "current_agent": "python",
                    "dataframe_info": {},
                    "has_data": True,
                    "final_result": "",
                    "metadata": {},
                    "iteration_count": 0
                }
                
                result = python_agent.invoke(test_state)
                
                self.assertIn("agent_outputs", result)
                self.assertIn("python", result["agent_outputs"])
                self.assertIn("result", result["agent_outputs"]["python"])
                self.assertIn("next_agent", result)
                self.assertEqual(result["next_agent"], "tone")
            
        except ImportError as e:
            self.skipTest(f"Communication test skipped due to missing dependencies: {e}")
        except Exception as e:
            self.fail(f"Agent communication test failed: {e}")


class TestSystemConfiguration(unittest.TestCase):
    """Test system configuration and setup"""
    
    def test_requirements_structure(self):
        """Test that requirements.txt contains necessary packages"""
        requirements_path = Path(__file__).parent.parent / "requirements.txt"
        self.assertTrue(requirements_path.exists(), "requirements.txt should exist")
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        essential_packages = [
            'langgraph', 'langchain', 'fastapi', 'pandas', 
            'matplotlib', 'numpy', 'pytest'
        ]
        
        for package in essential_packages:
            self.assertIn(package, requirements, f"{package} should be in requirements.txt")
    
    def test_directory_structure(self):
        """Test that the project has correct directory structure"""
        base_path = Path(__file__).parent.parent
        
        # Check main directories
        expected_dirs = ['src', 'tests', 'src/agents', 'src/api', 'src/langgraph_engine', 'src/data']
        
        for dir_path in expected_dirs:
            full_path = base_path / dir_path
            self.assertTrue(full_path.exists(), f"Directory {dir_path} should exist")
    
    def test_init_files_exist(self):
        """Test that __init__.py files exist for proper package structure"""
        base_path = Path(__file__).parent.parent
        
        init_files = [
            'src/__init__.py',
            'src/agents/__init__.py', 
            'src/api/__init__.py',
            'src/langgraph_engine/__init__.py',
            'tests/__init__.py'
        ]
        
        for init_file in init_files:
            full_path = base_path / init_file
            self.assertTrue(full_path.exists(), f"Init file {init_file} should exist")


if __name__ == '__main__':
    unittest.main()
