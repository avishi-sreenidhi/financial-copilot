import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import base64
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.charting_agent import ChartingAgent, generate_chart, load_and_chart_csv


class TestChartingAgent(unittest.TestCase):
    """Test cases for the Charting Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.llm = Mock()
        self.charting_agent = ChartingAgent(self.llm)
        
        # Sample data for testing
        self.sample_data = json.dumps([
            {"date": "2023-01-01", "revenue": 1000, "expenses": 500},
            {"date": "2023-01-02", "revenue": 1200, "expenses": 600},
            {"date": "2023-01-03", "revenue": 1100, "expenses": 550}
        ])
    
    def test_init(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.charting_agent.llm)
        self.assertIsNotNone(self.charting_agent.tools)
        self.assertIsNotNone(self.charting_agent.agent_executor)
        self.assertEqual(len(self.charting_agent.tools), 2)
    
    @patch('agents.charting_agent.plt')
    @patch('agents.charting_agent.BytesIO')
    @patch('agents.charting_agent.base64.b64encode')
    def test_generate_chart_line(self, mock_b64encode, mock_bytesio, mock_plt):
        """Test line chart generation"""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_b64encode.return_value.decode.return_value = "encoded_image_data"
        
        result = generate_chart.invoke({
            "data": self.sample_data, 
            "chart_type": "line", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Revenue Trends"
        })
        
        self.assertIn("Chart generated successfully", result)
        self.assertIn("data:image/png;base64,encoded_image_data", result)
        mock_plt.plot.assert_called_once()
        mock_plt.title.assert_called_once_with("Revenue Trends", fontsize=14, fontweight='bold')
    
    @patch('agents.charting_agent.plt')
    @patch('agents.charting_agent.BytesIO')
    @patch('agents.charting_agent.base64.b64encode')
    def test_generate_chart_bar(self, mock_b64encode, mock_bytesio, mock_plt):
        """Test bar chart generation"""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_b64encode.return_value.decode.return_value = "encoded_image_data"
        
        result = generate_chart.invoke({
            "data": self.sample_data, 
            "chart_type": "bar", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Revenue by Date"
        })
        
        self.assertIn("Chart generated successfully", result)
        mock_plt.bar.assert_called_once()
    
    @patch('agents.charting_agent.plt')
    @patch('agents.charting_agent.BytesIO')
    @patch('agents.charting_agent.base64.b64encode')
    def test_generate_chart_scatter(self, mock_b64encode, mock_bytesio, mock_plt):
        """Test scatter plot generation"""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_b64encode.return_value.decode.return_value = "encoded_image_data"
        
        result = generate_chart.invoke({
            "data": self.sample_data, 
            "chart_type": "scatter", 
            "x_axis": "revenue", 
            "y_axis": "expenses", 
            "title": "Revenue vs Expenses"
        })
        
        self.assertIn("Chart generated successfully", result)
        mock_plt.scatter.assert_called_once()
    
    @patch('agents.charting_agent.plt')
    @patch('agents.charting_agent.BytesIO')
    @patch('agents.charting_agent.base64.b64encode')
    def test_generate_chart_histogram(self, mock_b64encode, mock_bytesio, mock_plt):
        """Test histogram generation"""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_b64encode.return_value.decode.return_value = "encoded_image_data"
        
        result = generate_chart.invoke({
            "data": self.sample_data, 
            "chart_type": "histogram", 
            "x_axis": "", 
            "y_axis": "revenue", 
            "title": "Revenue Distribution"
        })
        
        self.assertIn("Chart generated successfully", result)
        mock_plt.hist.assert_called_once()
    
    @patch('agents.charting_agent.plt')
    @patch('agents.charting_agent.BytesIO')
    @patch('agents.charting_agent.base64.b64encode')
    def test_generate_chart_pie(self, mock_b64encode, mock_bytesio, mock_plt):
        """Test pie chart generation"""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_b64encode.return_value.decode.return_value = "encoded_image_data"
        
        result = generate_chart.invoke({
            "data": self.sample_data, 
            "chart_type": "pie", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Revenue Distribution"
        })
        
        self.assertIn("Chart generated successfully", result)
        mock_plt.pie.assert_called_once()
    
    def test_generate_chart_unsupported_type(self):
        """Test unsupported chart type"""
        result = generate_chart.invoke({
            "data": self.sample_data, 
            "chart_type": "unsupported", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Test"
        })
        
        self.assertIn("Unsupported chart type: unsupported", result)
    
    def test_generate_chart_invalid_data(self):
        """Test chart generation with invalid JSON data"""
        invalid_data = "invalid json"
        result = generate_chart.invoke({
            "data": invalid_data, 
            "chart_type": "line", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Test"
        })
        
        self.assertIn("Error generating chart:", result)
    
    @patch('agents.charting_agent.pd.read_csv')
    @patch('agents.charting_agent.generate_chart')
    def test_load_and_chart_csv_success(self, mock_generate_chart, mock_read_csv):
        """Test successful CSV loading and charting"""
        # Mock DataFrame
        mock_df = Mock()
        mock_df.to_json.return_value = self.sample_data
        mock_read_csv.return_value = mock_df
        mock_generate_chart.return_value = "Chart generated successfully"
        
        result = load_and_chart_csv.invoke({
            "file_path": "test.csv", 
            "chart_type": "line", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Test Chart"
        })
        
        self.assertIn("Chart generated successfully", result)
        mock_read_csv.assert_called_once_with("test.csv")
        mock_generate_chart.assert_called_once()
    
    @patch('agents.charting_agent.pd.read_csv')
    def test_load_and_chart_csv_error(self, mock_read_csv):
        """Test CSV loading error handling"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        result = load_and_chart_csv.invoke({
            "file_path": "nonexistent.csv", 
            "chart_type": "line", 
            "x_axis": "date", 
            "y_axis": "revenue", 
            "title": "Test"
        })
        
        self.assertIn("Error loading CSV and generating chart:", result)
        self.assertIn("File not found", result)
    
    def test_invoke_structure(self):
        """Test the invoke method structure"""
        with patch('agents.charting_agent.AgentExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.invoke.return_value = {"output": "chart generated"}
            mock_executor_class.return_value = mock_executor
            
            # Create a new agent instance for testing
            test_agent = ChartingAgent(Mock())
            
            # Create a state that matches the new DataAnalyticsState structure
            state = {
                "query": "create a line chart",
                "next_agent": "",
                "agent_outputs": {},
                "messages": [],
                "current_agent": "chart",
                "dataframe_info": {},
                "has_data": True,
                "final_result": "",
                "metadata": {},
                "iteration_count": 0
            }
            
            result = test_agent.invoke(state)
            
            # Check that the agent_outputs is populated correctly
            self.assertIn("chart", result["agent_outputs"])
            self.assertEqual(result["agent_outputs"]["chart"]["result"], "chart generated")
            self.assertEqual(result["next_agent"], "tone")
            
            # Verify the executor was called with the right input
            mock_executor.invoke.assert_called_once_with({"input": "create a line chart"})


class TestChartingTools(unittest.TestCase):
    """Test cases for charting tools with real data processing"""
    
    def setUp(self):
        """Set up test fixtures with real data"""
        self.test_data = json.dumps([
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
            {"category": "C", "value": 15}
        ])
    
    @patch('agents.charting_agent.plt')
    def test_chart_styling_applied(self, mock_plt):
        """Test that chart styling is properly applied"""
        # Mock the style and other matplotlib components
        mock_plt.style.use = Mock()
        mock_plt.figure = Mock()
        mock_plt.bar = Mock()
        mock_plt.title = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.grid = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()
        
        with patch('agents.charting_agent.BytesIO'), \
             patch('agents.charting_agent.base64.b64encode') as mock_b64:
            mock_b64.return_value.decode.return_value = "test_data"
            
            generate_chart.invoke({
                "data": self.test_data,
                "chart_type": "bar",
                "x_axis": "category",
                "y_axis": "value",
                "title": "Test Chart"
            })
            
            # Verify styling was applied
            mock_plt.style.use.assert_called_once_with('seaborn-v0_8')
            mock_plt.figure.assert_called_once_with(figsize=(12, 8))
            mock_plt.title.assert_called_once_with("Test Chart", fontsize=14, fontweight='bold')


if __name__ == '__main__':
    unittest.main()
