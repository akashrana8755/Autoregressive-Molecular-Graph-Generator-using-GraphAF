"""
Tests for molecular data processing components.
"""

import pytest
import torch
from src.data import SMILESProcessor, FeatureExtractor, MolecularDataset


class TestSMILESProcessor:
    """Test SMILES processing functionality."""
    
    def test_smiles_to_graph_basic(self):
        """Test basic SMILES to graph conversion."""
        processor = SMILESProcessor()
        
        # Test simple molecules
        test_smiles = [
            'C',      # Methane
            'CC',     # Ethane
            'CCO',    # Ethanol
            'c1ccccc1'  # Benzene
        ]
        
        for smiles in test_smiles:
            graph = processor.smiles_to_graph(smiles)
            assert graph is not None, f"Failed to convert {smiles}"
            assert hasattr(graph, 'x'), "Graph missing node features"
            assert hasattr(graph, 'edge_index'), "Graph missing edge indices"
            assert hasattr(graph, 'edge_attr'), "Graph missing edge features"
            assert graph.num_nodes > 0, "Graph has no nodes"
            
    def test_validate_molecule(self):
        """Test molecule validation."""
        processor = SMILESProcessor()
        
        # Valid molecules
        valid_smiles = ['C', 'CC', 'CCO', 'c1ccccc1']
        for smiles in valid_smiles:
            assert processor.validate_molecule(smiles), f"{smiles} should be valid"
            
        # Invalid molecules
        invalid_smiles = ['', 'X', 'C(', 'invalid']
        for smiles in invalid_smiles:
            assert not processor.validate_molecule(smiles), f"{smiles} should be invalid"
            
    def test_sanitize_smiles(self):
        """Test SMILES sanitization."""
        processor = SMILESProcessor()
        
        # Test canonicalization
        smiles = 'CCO'
        canonical = processor.sanitize_smiles(smiles)
        assert canonical is not None
        assert processor.validate_molecule(canonical)


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def test_feature_dimensions(self):
        """Test feature dimension consistency."""
        extractor = FeatureExtractor()
        
        atom_dim = extractor.get_atom_features_dim()
        bond_dim = extractor.get_bond_features_dim()
        
        assert atom_dim > 0, "Atom features should have positive dimension"
        assert bond_dim > 0, "Bond features should have positive dimension"
        
    def test_graph_features(self):
        """Test graph feature extraction."""
        from rdkit import Chem
        
        extractor = FeatureExtractor()
        
        # Test with simple molecule
        mol = Chem.MolFromSmiles('CCO')
        features = extractor.get_graph_features(mol)
        
        assert features is not None
        assert 'x' in features
        assert 'edge_attr' in features
        assert 'edge_index' in features
        
        # Check tensor shapes
        assert features['x'].dim() == 2
        assert features['edge_attr'].dim() == 2
        assert features['edge_index'].dim() == 2
        assert features['edge_index'].size(0) == 2
        
    def test_feature_names(self):
        """Test feature name extraction."""
        extractor = FeatureExtractor()
        
        feature_names = extractor.get_feature_names()
        assert 'atom_features' in feature_names
        assert 'bond_features' in feature_names
        assert len(feature_names['atom_features']) > 0
        assert len(feature_names['bond_features']) > 0


class TestMolecularDataset:
    """Test molecular dataset functionality."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        smiles_list = ['C', 'CC', 'CCO', 'c1ccccc1']
        
        dataset = MolecularDataset(
            smiles_list=smiles_list,
            use_cache=False  # Disable caching for tests
        )
        
        assert len(dataset) == len(smiles_list)
        
        # Test data access
        for i in range(len(dataset)):
            data = dataset[i]
            if data is not None:  # Some molecules might fail processing
                assert hasattr(data, 'x')
                assert hasattr(data, 'edge_index')
                assert hasattr(data, 'edge_attr')
                assert hasattr(data, 'smiles')
                
    def test_dataset_with_properties(self):
        """Test dataset with molecular properties."""
        smiles_list = ['C', 'CC', 'CCO']
        properties = {
            'logp': [0.5, 1.0, -0.3],
            'mw': [16.0, 30.0, 46.0]
        }
        
        dataset = MolecularDataset(
            smiles_list=smiles_list,
            properties=properties,
            use_cache=False
        )
        
        for i in range(len(dataset)):
            data = dataset[i]
            if data is not None:
                assert hasattr(data, 'logp')
                assert hasattr(data, 'mw')
                
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        smiles_list = ['C', 'CC', 'CCO', 'invalid_smiles']
        
        dataset = MolecularDataset(
            smiles_list=smiles_list,
            use_cache=False
        )
        
        stats = dataset.get_statistics()
        assert 'total_molecules' in stats
        assert 'valid_molecules' in stats
        assert 'invalid_molecules' in stats
        assert 'validity_rate' in stats
        assert stats['total_molecules'] == len(smiles_list)


if __name__ == "__main__":
    # Run basic tests
    test_processor = TestSMILESProcessor()
    test_processor.test_smiles_to_graph_basic()
    test_processor.test_validate_molecule()
    
    test_extractor = TestFeatureExtractor()
    test_extractor.test_feature_dimensions()
    test_extractor.test_graph_features()
    
    test_dataset = TestMolecularDataset()
    test_dataset.test_dataset_creation()
    
    print("All basic tests passed!")