# Unit Tests Documentation

## Overview

Comprehensive unit test suite for the Data Analytics Agent Swarm system. The tests cover all major components including individual agents, tools, integration points, and system configuration.

## Test Structure

```
tests/
├── __init__.py
├── test_router_agent.py          # Router agent routing logic tests
├── test_python_ide_agent.py      # Python execution and data analysis tests
├── test_charting_agent.py        # Data visualization tests
├── test_data_search_agent.py     # Data search and filtering tests
├── test_personality_agent.py     # Communication formatting tests
├── test_graph_builder.py         # LangGraph orchestration tests
└── test_integration.py           # End-to-end integration tests
```

## Test Categories

### 1. Router Agent Tests (`test_router_agent.py`)
- **Keyword-based routing**: Tests routing to correct agents based on message content
- **Case sensitivity**: Ensures routing works regardless of text case
- **Priority handling**: Tests behavior when multiple keywords are present
- **Default routing**: Verifies fallback to tone agent for conversational messages
- **Edge cases**: Empty messages, malformed input

**Key Test Cases:**
- Chart keywords → chart agent
- Search keywords → search agent  
- Python keywords → python agent
- Conversational → tone agent

### 2. Python IDE Agent Tests (`test_python_ide_agent.py`)
- **Code execution**: Basic Python code execution with result capture
- **Library integration**: Tests with pandas, numpy, json libraries
- **Error handling**: Malformed code, undefined variables
- **Dataset loading**: CSV file loading and validation
- **Sample data access**: Pre-loaded sample dataset functionality

**Key Test Cases:**
- Simple calculations: `result = 2 + 2`
- Pandas operations: DataFrame creation and manipulation
- NumPy calculations: Array operations and statistics
- Error scenarios: Syntax errors, runtime exceptions
- File operations: CSV loading with various formats

### 3. Charting Agent Tests (`test_charting_agent.py`)
- **Chart generation**: All supported chart types (line, bar, scatter, histogram, pie, boxplot)
- **Image encoding**: Base64 PNG output validation
- **Styling**: Professional formatting and appearance
- **Data handling**: JSON data parsing and validation
- **Error scenarios**: Invalid data, unsupported chart types

**Key Test Cases:**
- Line charts: Time series and trend visualization
- Bar charts: Categorical data comparison
- Scatter plots: Correlation analysis
- Histograms: Distribution analysis
- Error handling: Malformed JSON, missing columns

### 4. Data Search Agent Tests (`test_data_search_agent.py`)
- **Text search**: Case-insensitive pattern matching
- **Filtering**: Numeric and string-based filtering with operators
- **Data summaries**: Comprehensive dataset analysis
- **JSON output**: Structured response formatting
- **Error handling**: File not found, invalid columns

**Key Test Cases:**
- Search operations: Text pattern matching across columns
- Filter operations: `>`, `<`, `>=`, `<=`, `==`, `!=`, `contains`
- Data summary: Shape, columns, types, statistics
- Edge cases: Empty results, invalid operators

### 5. Personality Agent Tests (`test_personality_agent.py`)
- **Tone detection**: Professional, educational, casual styles
- **Context awareness**: Business-appropriate language selection
- **State handling**: Missing data graceful degradation
- **Communication quality**: Clear, actionable output formatting

**Key Test Cases:**
- Professional tone: Executive reports, business summaries
- Educational tone: Learning and tutorial contexts
- Casual tone: Quick updates and informal communication
- Default behavior: Fallback to professional tone

### 6. Graph Builder Tests (`test_graph_builder.py`)
- **Graph construction**: Node and edge creation
- **State management**: AgentState structure and flow
- **Routing logic**: Conditional edge configuration
- **Agent initialization**: All agents properly instantiated
- **Workflow orchestration**: End-to-end state transitions

**Key Test Cases:**
- Node creation: All required agents added to graph
- Edge configuration: Proper routing between agents
- State structure: AgentState annotations and functionality
- Entry point: Router as initial node

### 7. Integration Tests (`test_integration.py`)
- **System configuration**: Directory structure, dependencies
- **End-to-end workflow**: Complete agent interaction flow
- **Data file validation**: Sample CSV structure and content
- **Import resolution**: All modules can be imported
- **State flow**: Data passing between agents

**Key Test Cases:**
- File system: Required directories and files exist
- Dependencies: All required packages in requirements.txt
- Sample data: CSV file structure and content validation
- Agent communication: State transformation through workflow

## Running Tests

### All Tests
```bash
# Using pytest (recommended)
pytest

# Using unittest
python -m unittest discover tests

# Using custom test runner
python run_tests.py
```

### Specific Test Files
```bash
# Single test file
pytest tests/test_router_agent.py

# Specific test class
pytest tests/test_router_agent.py::TestRouterAgent

# Specific test method
pytest tests/test_router_agent.py::TestRouterAgent::test_chart_routing_keywords
```

### With Coverage
```bash
# Coverage report
pytest --cov=src --cov-report=html

# Coverage with missing lines
pytest --cov=src --cov-report=term-missing
```

## Test Configuration

### pytest.ini
- **Test discovery**: Automatic test file and function detection
- **Coverage**: 80% minimum coverage requirement
- **Output formatting**: Verbose output with short tracebacks
- **Markers**: Unit, integration, and slow test categorization

### Key Settings
- `testpaths = tests`: Test discovery path
- `--cov-fail-under=80`: Minimum coverage threshold
- `--tb=short`: Concise traceback format
- `--disable-warnings`: Clean test output

## Mocking Strategy

### External Dependencies
- **LangChain components**: Mocked to avoid API calls
- **File operations**: Mocked pandas.read_csv for predictable data
- **LLM calls**: Mocked ChatOpenAI to avoid external API dependencies
- **Plot generation**: Mocked matplotlib for visualization tests

### Test Data
- **Sample datasets**: In-memory DataFrames for consistent testing
- **JSON responses**: Predictable structured data for validation
- **State objects**: AgentState instances with known values

## Coverage Goals

### Target Coverage: 80%+
- **Core logic**: 95%+ coverage for business logic
- **Error handling**: 90%+ coverage for exception paths
- **Integration**: 70%+ coverage for workflow orchestration
- **Configuration**: 100% coverage for setup and initialization

### Coverage Reports
- **HTML report**: Detailed line-by-line coverage analysis
- **Terminal output**: Quick overview with missing lines
- **CI/CD integration**: Automated coverage checking

## Best Practices

### Test Design
1. **Isolation**: Each test is independent and can run alone
2. **Predictability**: Mocked dependencies for consistent results
3. **Readability**: Clear test names describing the behavior being tested
4. **Maintainability**: Organized test structure matching source code

### Data Management
1. **Test fixtures**: Reusable test data in setUp methods
2. **Mock data**: Consistent datasets across related tests
3. **Edge cases**: Empty data, malformed input, boundary conditions
4. **Real scenarios**: Tests reflecting actual usage patterns

### Error Testing
1. **Exception handling**: All error paths tested
2. **Input validation**: Invalid input scenarios covered
3. **Resource failures**: File not found, network errors
4. **Graceful degradation**: System behavior under failure conditions

## Continuous Integration

### Automated Testing
- **Pre-commit hooks**: Run tests before code commits
- **CI pipeline**: Automated test execution on pull requests
- **Coverage enforcement**: Prevent coverage regression
- **Performance monitoring**: Test execution time tracking

### Quality Gates
- **All tests pass**: No failing tests allowed in main branch
- **Coverage threshold**: Minimum 80% code coverage
- **No test warnings**: Clean test execution
- **Documentation**: Tests serve as usage documentation

This comprehensive test suite ensures the reliability, maintainability, and correctness of the Data Analytics Agent Swarm system.
