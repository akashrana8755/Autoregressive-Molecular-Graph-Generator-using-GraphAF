"""
Tests for molecular generation engine.
"""

import pytest
import torch
from torch_geometric.data import Data
from unittest.mock import Mock, patch
import numpy as np

from src.generate.molecular_generator import MolecularGenerator
from src.generate.constraint_filter import ConstraintFilter
from src.data.smiles_processor import SMILESProcessor
from src.models.base_model import BaseGenerativeModel


class MockGenerativeModel(BaseGenerativeModel):
    """Mock generative model for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(1, 1)  # Dummy parameter
    
    def forward(self, batch, **kwargs):
        return {"output": torch.randn(batch.num_graphs, 10)}
    
    def training_step(self, batch, **kwargs):
        return torch.tensor(1.0)
    
    def sample(self, num_samples, **kwargs):
        """Generate mock molecular graphs."""
        graphs = []
        for i in range(num_samples):
            # Create a simple mock graph (benzene-like)
            x = torch.tensor([[6.0], [6.0], [6.0], [6.0], [6.0], [6.0]], dtype=torch.float)  # Carbon atoms
            edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],  # Ring structure
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
            ], dtype=torch.long)
            edge_attr = torch.ones(12, 1, dtype=torch.float)  # Single bonds
            
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph.smiles = "c1ccccc1"  # Benzene SMILES
            graphs.append(graph)
        
        return graphs


class TestMolecularGenerator:
    """Test cases for MolecularGenerator."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock generative model."""
        config = {
            'node_dim': 1,
            'edge_dim': 1,
            'hidden_dim': 64,
            'num_layers': 2
        }
        return MockGenerativeModel(config)
    
    @pytest.fixture
    def smiles_processor(self):
        """Create SMILES processor."""
        return SMILESProcessor()
    
    @pytest.fixture
    def constraint_filter(self):
        """Create constraint filter."""
        return ConstraintFilter()
    
    @pytest.fixture
    def generator(self, mock_model, smiles_processor, constraint_filter):
        """Create molecular generator."""
        return MolecularGenerator(
            model=mock_model,
            smiles_processor=smiles_processor,
            constraint_filter=constraint_filter
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.model is not None
        assert generator.smiles_processor is not None
        assert generator.constraint_filter is not None
        assert generator.device is not None
    
    def test_basic_generation(self, generator):
        """Test basic molecule generation."""
        smiles_list = generator.generate(num_molecules=5, max_attempts=1)
        
        assert isinstance(smiles_list, list)
        assert len(smiles_list) <= 5  # May be less due to validation failures
        
        for smiles in smiles_list:
            assert isinstance(smiles, str)
            assert len(smiles) > 0
    
    def test_batch_generation(self, generator):
        """Test batch generation."""
        smiles_list, graphs = generator.generate_batch(batch_size=3)
        
        assert isinstance(smiles_list, list)
        assert isinstance(graphs, list)
        assert len(smiles_list) == len(graphs)
        assert len(smiles_list) <= 3
    
    def test_constraint_generation(self, generator):
        """Test constraint-aware generation."""
        constraints = {
            'lipinski': True,
            'qed_threshold': 0.3,
            'mw_range': [100, 400]
        }
        
        smiles_list = generator.generate_with_constraints(
            num_molecules=3,
            constraints=constraints,
            max_attempts=5
        )
        
        assert isinstance(smiles_list, list)
        assert len(smiles_list) <= 3
        
        # Verify constraints are applied
        for smiles in smiles_list:
            assert generator.constraint_filter.passes_lipinski_filter(smiles)
    
    def test_iterative_generation(self, generator):
        """Test iterative constraint generation."""
        constraints = {
            'lipinski': True,
            'qed_threshold': 0.4
        }
        
        smiles_list = generator.iterative_constraint_generation(
            num_molecules=2,
            constraints=constraints,
            max_iterations=3,
            batch_size=2
        )
        
        assert isinstance(smiles_list, list)
        assert len(smiles_list) <= 2
    
    def test_validation(self, generator):
        """Test molecule validation."""
        test_smiles = ["c1ccccc1", "CCO", "invalid_smiles", "CC(C)C"]
        
        stats = generator.validate_molecules(test_smiles)
        
        assert 'total_molecules' in stats
        assert 'valid_molecules' in stats
        assert 'validity_rate' in stats
        assert 'unique_molecules' in stats
        assert 'uniqueness_rate' in stats
        
        assert stats['total_molecules'] == 4
        assert stats['valid_molecules'] >= 3  # At least the valid ones
    
    def test_statistics(self, generator):
        """Test generation statistics."""
        # Reset statistics
        generator.reset_statistics()
        stats = generator.get_generation_statistics()
        
        assert stats['total_generated'] == 0
        assert stats['valid_molecules'] == 0
        assert stats['validity_rate'] == 0.0
        
        # Generate some molecules
        generator.generate(num_molecules=2, max_attempts=1)
        
        # Check updated statistics
        stats = generator.get_generation_statistics()
        assert stats['total_generated'] >= 0
    
    def test_properties_to_constraints(self, generator):
        """Test property to constraint conversion."""
        target_properties = {
            'mw': 300.0,
            'logp': 2.5,
            'qed': 0.7
        }
        
        constraints = generator._properties_to_constraints(target_properties, tolerance=0.1)
        
        assert 'mw_range' in constraints
        assert 'logp_range' in constraints
        assert 'qed_threshold' in constraints
        
        assert constraints['mw_range'][0] < 300.0 < constraints['mw_range'][1]
        assert constraints['logp_range'][0] < 2.5 < constraints['logp_range'][1]
        assert constraints['qed_threshold'] <= 0.7
    
    def test_constraint_checking(self, generator):
        """Test constraint checking for molecules."""
        # Test with benzene (should pass most constraints)
        smiles = "c1ccccc1"
        
        constraints = {
            'lipinski': True,
            'qed_threshold': 0.1,  # Low threshold
            'mw_range': [50, 200]
        }
        
        passes = generator._molecule_passes_constraints(smiles, constraints)
        assert isinstance(passes, bool)
    
    def test_duplicate_removal(self, generator):
        """Test duplicate removal."""
        smiles_list = ["c1ccccc1", "CCO", "c1ccccc1", "CCC", "CCO"]
        graphs_list = [Data(x=torch.randn(3, 1)) for _ in smiles_list]
        
        unique_smiles, unique_graphs = generator._remove_duplicates(smiles_list, graphs_list)
        
        assert len(unique_smiles) == 3  # Should have 3 unique SMILES
        assert len(unique_graphs) == 3
        assert "c1ccccc1" in unique_smiles
        assert "CCO" in unique_smiles
        assert "CCC" in unique_smiles
    
    @patch('src.generate.molecular_generator.torch.load')
    def test_from_checkpoint(self, mock_load):
        """Test loading generator from checkpoint."""
        # Mock checkpoint data
        mock_load.return_value = {
            'model_name': 'MockGenerativeModel',
            'config': {'node_dim': 1, 'edge_dim': 1, 'hidden_dim': 64, 'num_layers': 2},
            'model_state_dict': {}
        }
        
        # This would normally fail because MockGenerativeModel isn't a real model class
        # but we can test the structure
        with pytest.raises((ValueError, AttributeError)):
            MolecularGenerator.from_checkpoint('dummy_path.pt')
    
    def test_model_info(self, generator):
        """Test getting model information."""
        info = generator.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info


if __name__ == "__main__":
    pytest.main([__file__])