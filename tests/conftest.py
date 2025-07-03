"""
Test configuration and fixtures for the data analytics agent system.
This file sets up environment variables and common test fixtures.
"""

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

# Set up environment variables for testing
os.environ['OPENAI_API_KEY'] = 'test_key_for_testing'
os.environ['TESTING'] = 'true'

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI for all tests to avoid API calls"""
    with patch('langchain_openai.ChatOpenAI') as mock_llm:
        mock_instance = Mock()
        # Mock the invoke method to return a proper structure with string content
        mock_response = Mock()
        mock_response.content = "Final Answer: chart"
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_data():
    """Provide sample data for testing"""
    return {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Tokyo']
    }


@pytest.fixture
def mock_dataframe():
    """Mock pandas DataFrame for testing"""
    import pandas as pd
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Tokyo']
    })
