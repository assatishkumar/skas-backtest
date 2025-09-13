# Testing Guide

This document explains how to run tests for the SKAS Backtest framework.

## Test Structure

```
tests/
├── test_data_manager.py              # Unit tests (with mocks)
├── test_data_manager_integration.py  # Integration tests (real data)
├── unit/                            # Additional unit tests
└── integration/                     # Additional integration tests
```

## Running Tests

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate
```

### Basic Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data_manager.py

# Run specific test class
pytest tests/test_data_manager.py::TestDataManagerBasics

# Run specific test method
pytest tests/test_data_manager.py::TestDataManagerBasics::test_initialization
```

### Coverage Reports

```bash
# Run tests with coverage (default)
pytest

# Generate HTML coverage report
pytest --cov-report=html
# Open htmlcov/index.html in browser

# Coverage report in terminal only
pytest --cov-report=term-missing

# No coverage report
pytest --no-cov
```

### Test Filtering

```bash
# Run only unit tests (exclude integration)
pytest -m "not integration"

# Run only integration tests
pytest -m integration

# Run tests matching pattern
pytest -k "test_cache"

# Run tests in specific directory
pytest tests/unit/
```

### Useful Options

```bash
# Stop on first failure
pytest -x

# Show local variables on failures
pytest -l

# Quiet output (less verbose)
pytest -q

# Show slowest tests
pytest --durations=10

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

## Test Types

### Unit Tests (`test_data_manager.py`)
- Fast execution (uses mocks)
- Test individual components in isolation
- No external dependencies
- Always safe to run

**Features:**
- Proper pytest fixtures
- Mocked external calls
- Parametrized tests
- Clear assertions
- Good coverage

### Integration Tests (`test_data_manager_integration.py`)
- May require actual NSE data
- Test complete workflows
- External dependencies
- Use with caution

**Note:** Integration tests may be slow or fail if external services are unavailable.

## Test Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=skas_backtest --cov-report=html --cov-report=term-missing"
markers = [
    "integration: marks tests as integration tests (may require external data)",
    "slow: marks tests as slow running",
]
```

## Example Test Run

```bash
$ source venv/bin/activate
$ pytest tests/test_data_manager.py -v

============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-8.4.2, pluggy-1.6.0
rootdir: /path/to/skas-backtest
configfile: pyproject.toml
plugins: mock-3.15.0, cov-6.3.0
collecting ... collected 20 items

tests/test_data_manager.py::TestDataManagerBasics::test_initialization PASSED [  5%]
tests/test_data_manager.py::TestDataManagerBasics::test_cache_functionality PASSED [ 10%]
...
================== 19 passed, 1 skipped, 0 warnings in 15.23s ==================
```

## Writing New Tests

### Unit Test Example
```python
import pytest
from unittest.mock import Mock, patch
from skas_backtest.data import DataManager

class TestMyFeature:
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_my_functionality(self, data_manager):
        # Arrange
        expected_result = "test_value"
        
        # Act
        result = data_manager.my_method()
        
        # Assert
        assert result == expected_result
```

### Mocking External Calls
```python
def test_with_mock(self, data_manager):
    with patch.object(data_manager.nse_fetcher, 'get_eod_price', return_value=100.0):
        price = data_manager.get_stock_price("RELIANCE", date(2024, 1, 15))
        assert price == 100.0
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project root and venv is active
   source venv/bin/activate
   pytest
   ```

2. **NSE Library Not Found**
   ```bash
   # Reinstall the NSE options library
   pip install -e /Users/saasrinivasan/git/saasrinivasan/personal-projects/skas-pricing/nse-options-pricing
   ```

3. **Coverage Issues**
   ```bash
   # Install coverage if missing
   pip install pytest-cov
   ```

4. **Slow Tests**
   ```bash
   # Skip integration tests
   pytest -m "not integration"
   ```

### Debug Mode
```bash
# Run tests with Python debugger
pytest --pdb

# Run specific test with maximum verbosity
pytest tests/test_data_manager.py::TestDataManagerBasics::test_initialization -vvv -s
```

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Fast tests only (no integration)
pytest -m "not integration" --no-cov

# Full test suite with coverage
pytest --cov=skas_backtest --cov-fail-under=80
```

---

**Note:** Always run tests before committing code changes to ensure everything works correctly.