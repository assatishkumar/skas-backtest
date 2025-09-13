# SKAS Backtest Package Structure

## Current Package Organization

```
skas_backtest/
├── __init__.py                    # Main package exports and version info
├── core/                          # Core backtesting framework
│   ├── __init__.py               # Core components exports
│   ├── backtest.py               # Main backtesting engine
│   ├── strategy.py               # Base strategy class and signals
│   ├── portfolio.py              # Portfolio and position management
│   ├── risk_manager.py           # Risk management and validation
│   └── metrics.py                # Performance metrics calculation
├── data/                          # Data management and fetching
│   ├── __init__.py               # Data exports (DataManager, etc.)
│   └── data_manager.py           # NSE data integration and caching
├── strategies/                    # Strategy implementations
│   └── __init__.py               # Strategy exports (currently empty)
├── utils/                         # Utility functions
│   ├── __init__.py               # Utility exports
│   └── helpers.py                # Helper functions
├── analysis/                      # Analysis and reporting
│   └── __init__.py               # Analysis exports (currently empty)
├── tests/                         # Test suite
│   ├── __init__.py               # Test package
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── test_data_manager.py      # Data manager tests
├── examples/                      # Example usage scripts
└── docs/                          # Documentation
```

## Package Responsibilities

### Core (`skas_backtest.core`)
**Purpose**: Fundamental backtesting framework components
- `backtest.py`: Main backtesting engine, orchestrates entire process
- `strategy.py`: Abstract strategy interface, signal definitions
- `portfolio.py`: Position tracking, trade execution, P&L calculation
- `risk_manager.py`: Risk controls, validation, stop losses
- `metrics.py`: Performance metrics, Sharpe ratio, drawdown analysis

### Data (`skas_backtest.data`)
**Purpose**: Data fetching, caching, and management
- `data_manager.py`: Primary data interface, NSE library integration
- Future: `cache_manager.py`, `data_validators.py`

### Strategies (`skas_backtest.strategies`)
**Purpose**: Trading strategy implementations
- Future: `middle_way_strategy.py`, `iron_condor_strategy.py`, etc.

### Utils (`skas_backtest.utils`)
**Purpose**: Shared utility functions and helpers
- `helpers.py`: Common calculations, formatting functions
- Future: `date_utils.py`, `math_utils.py`

### Analysis (`skas_backtest.analysis`)
**Purpose**: Result analysis and reporting
- Future: `performance_analyzer.py`, `report_generator.py`, `visualizer.py`

### Tests (`tests/`)
**Purpose**: Comprehensive test coverage
- `unit/`: Individual component tests
- `integration/`: End-to-end workflow tests
- `test_*.py`: Component-specific test files

## Import Guidelines

### Correct Import Patterns
```python
# Main package imports
from skas_backtest import Backtest, DataManager, Strategy

# Specific component imports
from skas_backtest.data import DataManager, MarketData
from skas_backtest.core import Strategy, Portfolio
from skas_backtest.strategies import MiddleWayStrategy  # Future

# Cross-package imports within the codebase
from ..data import DataManager  # From core to data
from ...core.strategy import Strategy  # From strategies to core
```

### Import Rules
1. **Public API**: Import main classes from package root
2. **Internal Imports**: Use relative imports between packages
3. **External Dependencies**: Import at module level, not package level
4. **Test Imports**: Always use absolute imports from project root

## Dependencies by Package

### Core Dependencies
- `pandas`, `numpy`: Data manipulation
- `datetime`, `typing`: Standard library
- Internal: `data`, `utils`

### Data Dependencies  
- `nse_options`: NSE data integration
- `pandas`, `numpy`: Data processing
- `logging`: Error handling
- Internal: None (base package)

### Strategies Dependencies
- Internal: `core`, `data`, `utils`
- External: `pandas`, `numpy`

## Future Expansions

### Near Term (Phase 2)
```
strategies/
├── __init__.py
├── middle_way_strategy.py         # Main strategy implementation
├── base_options_strategy.py       # Common options strategy utilities
└── strategy_configs.py            # Strategy configurations

utils/
├── date_utils.py                  # NSE calendar, trading days
├── options_utils.py               # Greeks, volatility calculations  
└── validation.py                  # Input validation helpers
```

### Long Term (Phase 3+)
```
analysis/
├── performance_analyzer.py        # Metrics calculation
├── report_generator.py           # HTML/PDF reports
├── visualizer.py                 # Charts and plots
└── risk_analyzer.py              # Risk decomposition

data/
├── cache_manager.py              # Advanced caching
├── data_validators.py            # Data quality checks
└── alternative_sources.py        # Multiple data providers

strategies/
├── iron_condor_strategy.py       # Additional strategies
├── calendar_spread_strategy.py   
└── momentum_strategy.py
```

## Best Practices

### Code Organization
1. Keep packages focused on single responsibility
2. Minimize cross-package dependencies  
3. Use clear, descriptive module names
4. Group related functionality together

### Import Management
1. Import from package root for public API
2. Use relative imports for internal cross-references
3. Document all public interfaces in `__init__.py`
4. Avoid circular imports through careful design

### Testing Strategy
1. Test each package independently
2. Mock external dependencies in unit tests
3. Use integration tests for cross-package workflows
4. Maintain test files parallel to source structure

---

*This structure supports the Middle Way strategy implementation and provides a scalable foundation for additional trading strategies.*