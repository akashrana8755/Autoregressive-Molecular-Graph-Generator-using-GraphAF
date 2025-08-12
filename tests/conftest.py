"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import torch
import numpy as np
import warnings
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch


def pytest_configure(config):
    """Configure pytest settings."""
    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure torch settings for testing
    torch.set_num_threads(1)  # Use single thread for consistent testing
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        'model': {
            'node_dim': 16,
            'edge_dim': 8,
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1,
            'max_nodes': 10,
            'num_timesteps': 50,
            'beta_schedule': 'linear',
            'num_flow_layers': 2,
            'num_node_types': 5
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-3,
            'num_epochs': 2,
            'gradient_clip': 1.0
        },
        'data': {
            'max_nodes': 10,
            'train_split': 0.8,
            'val_split': 0.2
        }
    }


@pytest.fixture(scope="session")
def test_smiles():
    """Provide test SMILES strings."""
    return [
        'C',           # Methane
        'CC',          # Ethane
        'CCO',         # Ethanol
        'c1ccccc1',    # Benzene
        'CC(=O)O',     # Acetic acid
        'CCN(CC)CC',   # Triethylamine
        'C1CCC(CC1)O', # Cyclohexanol
    ]


@pytest.fixture(scope="session")
def invalid_smiles():
    """Provide invalid SMILES strings for testing."""
    return [
        '',            # Empty string
        'X',           # Invalid atom
        'C(',          # Unmatched parenthesis
        'invalid',     # Nonsense string
        'C1CC1C1',     # Invalid ring closure
    ]


@pytest.fixture
def temp_dir():
    """Provide temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_rdkit():
    """Mock RDKit for tests that don't require actual RDKit functionality."""
    with patch('src.data.smiles_processor.RDKIT_AVAILABLE', True):
        mock_chem = patch('src.data.smiles_processor.Chem')
        mock_descriptors = patch('src.data.smiles_processor.Descriptors')
        
        with mock_chem as chem_mock, mock_descriptors as desc_mock:
            # Configure basic mocks
            mock_mol = chem_mock.MolFromSmiles.return_value
            mock_mol.GetNumAtoms.return_value = 5
            mock_mol.GetNumBonds.return_value = 4
            
            yield chem_mock, desc_mock


@pytest.fixture
def sample_molecular_graphs(test_config):
    """Provide sample molecular graphs for testing."""
    from torch_geometric.data import Data
    
    graphs = []
    for i in range(5):
        num_nodes = 3 + i
        
        x = torch.randn(num_nodes, test_config['model']['node_dim'])
        
        # Create simple chain connectivity
        if num_nodes > 1:
            edge_list = []
            for j in range(num_nodes - 1):
                edge_list.extend([[j, j+1], [j+1, j]])  # Bidirectional
            edge_index = torch.tensor(edge_list).t()
            edge_attr = torch.randn(edge_index.size(1), test_config['model']['edge_dim'])
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, test_config['model']['edge_dim'])
        
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        graphs.append(graph)
    
    return graphs


@pytest.fixture
def sample_batch(sample_molecular_graphs):
    """Provide sample batch for testing."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(sample_molecular_graphs)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def skip_if_no_rdkit():
    """Skip test if RDKit is not available."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles('C')
        if mol is None:
            pytest.skip("RDKit not properly configured")
    except ImportError:
        pytest.skip("RDKit not available")


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def test_properties():
    """Provide test molecular properties."""
    return {
        'logp': [0.5, 1.0, -0.3, 2.1, 0.2, 1.5, 1.2],
        'molecular_weight': [16.0, 30.0, 46.0, 78.0, 60.0, 101.0, 100.0],
        'qed': [0.3, 0.4, 0.7, 0.6, 0.5, 0.4, 0.6]
    }


# Custom markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "rdkit: mark test as requiring RDKit"
    )


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        if "test_integration" in item.nodeid or "test_training_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        if "rdkit" in item.name.lower() or "smiles" in item.name.lower():
            item.add_marker(pytest.mark.rdkit)
        
        # Mark unit tests
        if any(name in item.nodeid for name in [
            "test_smiles_processor", "test_feature_extractor", 
            "test_model_forward_passes", "test_loss_computations"
        ]):
            item.add_marker(pytest.mark.unit)


# Fixtures for mocking external dependencies
@pytest.fixture
def mock_dataset_download():
    """Mock dataset download functionality."""
    with patch('src.data.dataset_downloader.download_zinc15') as mock_zinc, \
         patch('src.data.dataset_downloader.download_qm9') as mock_qm9:
        
        mock_zinc.return_value = ["C", "CC", "CCO"]
        mock_qm9.return_value = ["C", "CC", "CCO"]
        
        yield mock_zinc, mock_qm9


@pytest.fixture
def mock_property_calculation():
    """Mock property calculation functionality."""
    with patch('src.data.property_calculator.calculate_properties') as mock_calc:
        mock_calc.return_value = {
            'logp': 1.0,
            'molecular_weight': 100.0,
            'qed': 0.5
        }
        yield mock_calc


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Utility for timing test operations."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Memory testing utilities
@pytest.fixture
def memory_monitor():
    """Utility for monitoring memory usage."""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
        
        def start(self):
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def current_usage(self):
            return self.process.memory_info().rss / 1024 / 1024  # MB
        
        def memory_increase(self):
            if self.initial_memory is None:
                return None
            return self.current_usage() - self.initial_memory
    
    return MemoryMonitor()