# MolecuGen Test Suite

This directory contains a comprehensive test suite for the MolecuGen molecular generation system. The tests are organized into unit tests, integration tests, and performance tests to ensure code quality, functionality, and performance.

## Test Structure

```
tests/
├── conftest.py                      # Pytest configuration and shared fixtures
├── run_tests.py                     # Test runner script
├── README.md                        # This file
│
├── Unit Tests/
├── test_smiles_processor.py         # SMILES processing functionality
├── test_feature_extractor.py        # Feature extraction functionality  
├── test_model_forward_passes.py     # Model forward passes and computations
├── test_loss_computations.py        # Loss function computations
├── test_constraint_filter.py        # Constraint filtering (existing)
├── test_molecular_evaluator.py      # Molecular evaluation (existing)
├── test_molecular_generator.py      # Molecular generation (existing)
├── test_data_processing.py          # Data processing (existing)
├── test_data_pipeline.py            # Data pipeline (existing)
├── test_config.py                   # Configuration (existing)
│
├── Integration Tests/
├── test_training_integration.py     # Complete training workflows
├── test_generation_integration.py   # Complete generation workflows
├── test_data_pipeline_integration.py # Data loading and preprocessing
│
└── Performance Tests/
└── test_performance_memory.py       # Performance and memory usage tests
```

## Test Categories

### Unit Tests
- **SMILES Processing**: Test SMILES string validation, conversion, and sanitization
- **Feature Extraction**: Test molecular feature extraction and graph conversion
- **Model Forward Passes**: Test model forward passes, gradient flow, and computations
- **Loss Computations**: Test all loss functions and their components
- **Constraint Filtering**: Test drug-likeness filtering and property calculations
- **Molecular Evaluation**: Test evaluation metrics and statistical computations

### Integration Tests
- **Training Integration**: Test complete training workflows with different models
- **Generation Integration**: Test end-to-end molecule generation pipelines
- **Data Pipeline Integration**: Test data loading, preprocessing, and batching

### Performance Tests
- **Performance**: Test execution speed and throughput
- **Memory Usage**: Test memory efficiency and leak detection
- **Scalability**: Test performance with different batch sizes and model sizes

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --unit           # Unit tests only
python tests/run_tests.py --integration    # Integration tests only
python tests/run_tests.py --performance    # Performance tests only
python tests/run_tests.py --fast          # Fast tests only (exclude slow)
```

### Advanced Usage

```bash
# Run with coverage report
python tests/run_tests.py --coverage

# Run specific test file
python tests/run_tests.py --file test_smiles_processor.py

# Run specific test function
python tests/run_tests.py --test test_smiles_to_graph_basic

# Run tests in parallel
python tests/run_tests.py --parallel 4

# Generate HTML report
python tests/run_tests.py --html-report

# Verbose output
python tests/run_tests.py --verbose
```

### Using pytest directly

```bash
# Basic usage
pytest

# Run specific markers
pytest -m "unit"
pytest -m "integration"
pytest -m "performance"
pytest -m "not slow"

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific file
pytest tests/test_smiles_processor.py

# Run with verbose output
pytest -v
```

## Test Markers

Tests are marked with the following categories:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for complete workflows
- `performance`: Performance and benchmarking tests
- `slow`: Tests that take longer to run
- `gpu`: Tests requiring GPU/CUDA
- `rdkit`: Tests requiring RDKit library

## Dependencies

### Required Dependencies
```bash
pip install pytest pytest-cov pytest-html pytest-xdist
pip install torch torch-geometric
pip install numpy psutil
```

### Optional Dependencies
```bash
pip install rdkit-pypi  # For SMILES processing tests
pip install matplotlib  # For visualization in some tests
```

### Install all test dependencies
```bash
python tests/run_tests.py --install-deps
```

## Test Configuration

### Pytest Configuration
- Configuration is in `pytest.ini` at the project root
- Shared fixtures and utilities are in `tests/conftest.py`
- Custom markers are defined for test categorization

### Environment Variables
- `PYTEST_CURRENT_TEST`: Current test being run (set by pytest)
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility for GPU tests

### Test Data
- Test uses synthetic data and mock objects where possible
- Real molecular data is used sparingly to keep tests fast
- Large datasets are mocked or generated programmatically

## Writing New Tests

### Test Structure
```python
import pytest
import torch
from src.your_module import YourClass

class TestYourClass:
    """Test cases for YourClass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.instance = YourClass()
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = self.instance.method()
        assert result is not None
        assert isinstance(result, expected_type)
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            self.instance.method(invalid_input)
```

### Best Practices
1. **Use descriptive test names**: `test_smiles_to_graph_with_invalid_input`
2. **Test edge cases**: Empty inputs, invalid data, boundary conditions
3. **Use fixtures**: Shared test data and setup in `conftest.py`
4. **Mock external dependencies**: Use `unittest.mock` for external APIs
5. **Test error conditions**: Use `pytest.raises()` for expected exceptions
6. **Keep tests fast**: Mock expensive operations, use small test data
7. **Make tests deterministic**: Set random seeds, avoid time-dependent tests

### Adding Markers
```python
@pytest.mark.unit
def test_unit_functionality():
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_slow_integration():
    pass

@pytest.mark.gpu
def test_gpu_functionality():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
```

## Continuous Integration

### GitHub Actions
The test suite is designed to work with GitHub Actions:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: python tests/run_tests.py --install-deps
    - name: Run tests
      run: python tests/run_tests.py --coverage --junit-xml=test-results.xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### Local Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks (in .pre-commit-config.yaml)
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **RDKit not available**
   ```bash
   pip install rdkit-pypi
   # or
   conda install -c conda-forge rdkit
   ```

2. **CUDA tests failing**
   ```bash
   # Skip GPU tests if no CUDA
   python tests/run_tests.py --fast
   ```

3. **Memory issues**
   ```bash
   # Run tests with smaller batch sizes
   export PYTEST_BATCH_SIZE=2
   ```

4. **Slow tests**
   ```bash
   # Run only fast tests
   python tests/run_tests.py --fast
   ```

### Debug Mode
```bash
# Run with Python debugger
pytest --pdb

# Run with verbose output and no capture
pytest -v -s

# Run single test with maximum verbosity
pytest -vvv tests/test_specific.py::test_function
```

## Coverage Reports

### Generating Coverage
```bash
# HTML coverage report
python tests/run_tests.py --coverage
# Open htmlcov/index.html in browser

# Terminal coverage report
pytest --cov=src --cov-report=term-missing
```

### Coverage Goals
- **Overall coverage**: > 90%
- **Critical modules**: > 95%
- **New code**: 100% coverage required

## Performance Benchmarks

### Running Benchmarks
```bash
# Run performance tests
python tests/run_tests.py --performance

# Run with timing information
pytest --durations=10
```

### Performance Targets
- **Forward pass**: < 1s for batch of 32 molecules
- **Training step**: < 2s per step
- **Generation**: < 5s per molecule
- **Memory usage**: < 2GB for typical workloads

## Contributing

### Adding Tests
1. Create test file following naming convention: `test_*.py`
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Include docstrings and comments
4. Test both success and failure cases
5. Update this README if adding new test categories

### Test Review Checklist
- [ ] Tests cover new functionality
- [ ] Tests include error cases
- [ ] Tests are deterministic
- [ ] Tests run quickly (< 1s each for unit tests)
- [ ] Tests use appropriate fixtures
- [ ] Tests have descriptive names and docstrings

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/notes/testing.html)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Test-Driven Development](https://testdriven.io/test-driven-development/)