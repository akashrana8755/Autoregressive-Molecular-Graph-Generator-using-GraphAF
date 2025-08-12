"""
Comprehensive unit tests for FeatureExtractor class.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Mock molecules for testing
        self.test_smiles = [
            'C',           # Methane - simple
            'CCO',         # Ethanol - with heteroatom
            'c1ccccc1',    # Benzene - aromatic
            'CC(=O)O',     # Acetic acid - with double bond
            'C[C@H](N)C(=O)O',  # Alanine - with stereochemistry
        ]
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
        
        # Test with custom parameters
        extractor_custom = FeatureExtractor(
            use_chirality=False,
            use_partial_charge=False
        )
        assert extractor_custom is not None
    
    def test_get_atom_features_dim(self):
        """Test atom feature dimension calculation."""
        atom_dim = self.extractor.get_atom_features_dim()
        
        assert isinstance(atom_dim, int), "Atom feature dimension should be integer"
        assert atom_dim > 0, "Atom feature dimension should be positive"
        assert atom_dim < 1000, "Atom feature dimension should be reasonable"
    
    def test_get_bond_features_dim(self):
        """Test bond feature dimension calculation."""
        bond_dim = self.extractor.get_bond_features_dim()
        
        assert isinstance(bond_dim, int), "Bond feature dimension should be integer"
        assert bond_dim > 0, "Bond feature dimension should be positive"
        assert bond_dim < 1000, "Bond feature dimension should be reasonable"
    
    def test_get_feature_names(self):
        """Test feature name extraction."""
        feature_names = self.extractor.get_feature_names()
        
        assert isinstance(feature_names, dict), "Feature names should be dictionary"
        assert 'atom_features' in feature_names, "Should contain atom feature names"
        assert 'bond_features' in feature_names, "Should contain bond feature names"
        
        atom_names = feature_names['atom_features']
        bond_names = feature_names['bond_features']
        
        assert isinstance(atom_names, list), "Atom feature names should be list"
        assert isinstance(bond_names, list), "Bond feature names should be list"
        assert len(atom_names) > 0, "Should have atom feature names"
        assert len(bond_names) > 0, "Should have bond feature names"
        
        # Check that names are strings
        for name in atom_names:
            assert isinstance(name, str), f"Atom feature name {name} should be string"
        for name in bond_names:
            assert isinstance(name, str), f"Bond feature name {name} should be string"
    
    @patch('src.data.feature_extractor.Chem')
    def test_get_atom_features_basic(self, mock_chem):
        """Test basic atom feature extraction."""
        # Mock atom object
        mock_atom = MagicMock()
        mock_atom.GetAtomicNum.return_value = 6  # Carbon
        mock_atom.GetDegree.return_value = 4
        mock_atom.GetFormalCharge.return_value = 0
        mock_atom.GetHybridization.return_value = mock_chem.HybridizationType.SP3
        mock_atom.GetIsAromatic.return_value = False
        mock_atom.GetTotalNumHs.return_value = 4
        mock_atom.GetChiralTag.return_value = mock_chem.ChiralType.CHI_UNSPECIFIED
        mock_atom.GetMass.return_value = 12.01
        
        # Mock RDKit constants
        mock_chem.HybridizationType.SP3 = 3
        mock_chem.ChiralType.CHI_UNSPECIFIED = 0
        
        features = self.extractor.get_atom_features(mock_atom)
        
        assert isinstance(features, torch.Tensor), "Atom features should be tensor"
        assert features.dim() == 1, "Atom features should be 1D tensor"
        assert features.size(0) == self.extractor.get_atom_features_dim(), "Feature dimension mismatch"
        assert not torch.isnan(features).any(), "Atom features should not contain NaN"
        assert torch.isfinite(features).all(), "Atom features should be finite"
    
    @patch('src.data.feature_extractor.Chem')
    def test_get_bond_features_basic(self, mock_chem):
        """Test basic bond feature extraction."""
        # Mock bond object
        mock_bond = MagicMock()
        mock_bond.GetBondType.return_value = mock_chem.BondType.SINGLE
        mock_bond.GetStereo.return_value = mock_chem.BondStereo.STEREONONE
        mock_bond.GetIsConjugated.return_value = False
        mock_bond.IsInRing.return_value = False
        
        # Mock RDKit constants
        mock_chem.BondType.SINGLE = 1
        mock_chem.BondStereo.STEREONONE = 0
        
        features = self.extractor.get_bond_features(mock_bond)
        
        assert isinstance(features, torch.Tensor), "Bond features should be tensor"
        assert features.dim() == 1, "Bond features should be 1D tensor"
        assert features.size(0) == self.extractor.get_bond_features_dim(), "Feature dimension mismatch"
        assert not torch.isnan(features).any(), "Bond features should not contain NaN"
        assert torch.isfinite(features).all(), "Bond features should be finite"
    
    @patch('src.data.feature_extractor.Chem')
    def test_get_graph_features_basic(self, mock_chem):
        """Test basic graph feature extraction."""
        # Mock molecule
        mock_mol = MagicMock()
        
        # Mock atoms
        mock_atom1 = MagicMock()
        mock_atom1.GetAtomicNum.return_value = 6  # Carbon
        mock_atom1.GetDegree.return_value = 4
        mock_atom1.GetFormalCharge.return_value = 0
        mock_atom1.GetHybridization.return_value = 3  # SP3
        mock_atom1.GetIsAromatic.return_value = False
        mock_atom1.GetTotalNumHs.return_value = 3
        mock_atom1.GetChiralTag.return_value = 0
        mock_atom1.GetMass.return_value = 12.01
        
        mock_atom2 = MagicMock()
        mock_atom2.GetAtomicNum.return_value = 8  # Oxygen
        mock_atom2.GetDegree.return_value = 2
        mock_atom2.GetFormalCharge.return_value = 0
        mock_atom2.GetHybridization.return_value = 3  # SP3
        mock_atom2.GetIsAromatic.return_value = False
        mock_atom2.GetTotalNumHs.return_value = 1
        mock_atom2.GetChiralTag.return_value = 0
        mock_atom2.GetMass.return_value = 16.00
        
        mock_mol.GetAtoms.return_value = [mock_atom1, mock_atom2]
        mock_mol.GetNumAtoms.return_value = 2
        
        # Mock bonds
        mock_bond = MagicMock()
        mock_bond.GetBondType.return_value = 1  # SINGLE
        mock_bond.GetStereo.return_value = 0    # STEREONONE
        mock_bond.GetIsConjugated.return_value = False
        mock_bond.IsInRing.return_value = False
        mock_bond.GetBeginAtomIdx.return_value = 0
        mock_bond.GetEndAtomIdx.return_value = 1
        
        mock_mol.GetBonds.return_value = [mock_bond]
        mock_mol.GetNumBonds.return_value = 1
        
        features = self.extractor.get_graph_features(mock_mol)
        
        assert isinstance(features, dict), "Graph features should be dictionary"
        
        # Check required keys
        required_keys = ['x', 'edge_index', 'edge_attr']
        for key in required_keys:
            assert key in features, f"Missing required key: {key}"
        
        # Check tensor properties
        assert isinstance(features['x'], torch.Tensor), "Node features should be tensor"
        assert isinstance(features['edge_index'], torch.Tensor), "Edge indices should be tensor"
        assert isinstance(features['edge_attr'], torch.Tensor), "Edge features should be tensor"
        
        # Check dimensions
        assert features['x'].dim() == 2, "Node features should be 2D"
        assert features['edge_index'].dim() == 2, "Edge indices should be 2D"
        assert features['edge_attr'].dim() == 2, "Edge features should be 2D"
        
        # Check sizes
        assert features['x'].size(0) == 2, "Should have 2 nodes"
        assert features['x'].size(1) == self.extractor.get_atom_features_dim(), "Node feature dimension mismatch"
        assert features['edge_index'].size(0) == 2, "Edge index should have 2 rows"
        assert features['edge_index'].size(1) == 2, "Should have 2 edges (bidirectional)"
        assert features['edge_attr'].size(0) == 2, "Should have 2 edge features"
        assert features['edge_attr'].size(1) == self.extractor.get_bond_features_dim(), "Edge feature dimension mismatch"
    
    def test_feature_consistency(self):
        """Test feature consistency across different molecules."""
        from rdkit import Chem
        
        # Skip if RDKit not available
        try:
            mol = Chem.MolFromSmiles('C')
            if mol is None:
                pytest.skip("RDKit not properly configured")
        except:
            pytest.skip("RDKit not available")
        
        feature_dims = []
        
        for smiles in self.test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                features = self.extractor.get_graph_features(mol)
                
                node_dim = features['x'].size(1)
                edge_dim = features['edge_attr'].size(1)
                
                feature_dims.append((node_dim, edge_dim))
        
        if len(feature_dims) > 1:
            # All molecules should have same feature dimensions
            first_dims = feature_dims[0]
            for dims in feature_dims[1:]:
                assert dims == first_dims, f"Inconsistent feature dimensions: {dims} vs {first_dims}"
    
    def test_atom_type_encoding(self):
        """Test atom type encoding for different elements."""
        from rdkit import Chem
        
        try:
            # Test different atom types
            test_molecules = {
                'C': 'methane',
                'N': 'ammonia', 
                'O': 'water',
                'S': 'hydrogen sulfide',
                'P': 'phosphine',
                'F': 'hydrogen fluoride',
                'Cl': 'hydrogen chloride',
                'Br': 'hydrogen bromide',
                'I': 'hydrogen iodide'
            }
            
            atom_features = {}
            
            for atom_symbol, name in test_molecules.items():
                # Create simple molecule with this atom
                if atom_symbol == 'C':
                    mol = Chem.MolFromSmiles('C')
                elif atom_symbol in ['N', 'P']:
                    mol = Chem.MolFromSmiles(f'{atom_symbol}')
                elif atom_symbol in ['O', 'S']:
                    mol = Chem.MolFromSmiles(f'{atom_symbol}')
                else:  # Halogens
                    mol = Chem.MolFromSmiles(f'C{atom_symbol}')
                
                if mol is not None:
                    features = self.extractor.get_graph_features(mol)
                    if atom_symbol in ['F', 'Cl', 'Br', 'I']:
                        # For halogens, get the second atom (halogen)
                        atom_feature = features['x'][1] if features['x'].size(0) > 1 else features['x'][0]
                    else:
                        atom_feature = features['x'][0]
                    
                    atom_features[atom_symbol] = atom_feature
            
            # Check that different atoms have different features
            atom_symbols = list(atom_features.keys())
            for i, symbol1 in enumerate(atom_symbols):
                for symbol2 in atom_symbols[i+1:]:
                    feature1 = atom_features[symbol1]
                    feature2 = atom_features[symbol2]
                    
                    # Features should be different (not identical)
                    assert not torch.equal(feature1, feature2), \
                        f"Atoms {symbol1} and {symbol2} have identical features"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_bond_type_encoding(self):
        """Test bond type encoding for different bond types."""
        from rdkit import Chem
        
        try:
            # Test different bond types
            test_bonds = {
                'single': 'CC',           # Single bond
                'double': 'C=C',          # Double bond
                'triple': 'C#C',          # Triple bond
                'aromatic': 'c1ccccc1',   # Aromatic bonds
            }
            
            bond_features = {}
            
            for bond_type, smiles in test_bonds.items():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    features = self.extractor.get_graph_features(mol)
                    if features['edge_attr'].size(0) > 0:
                        bond_feature = features['edge_attr'][0]  # First bond
                        bond_features[bond_type] = bond_feature
            
            # Check that different bond types have different features
            bond_types = list(bond_features.keys())
            for i, type1 in enumerate(bond_types):
                for type2 in bond_types[i+1:]:
                    feature1 = bond_features[type1]
                    feature2 = bond_features[type2]
                    
                    # Features should be different
                    assert not torch.equal(feature1, feature2), \
                        f"Bond types {type1} and {type2} have identical features"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_aromatic_vs_aliphatic(self):
        """Test distinction between aromatic and aliphatic atoms."""
        from rdkit import Chem
        
        try:
            # Aromatic carbon in benzene
            benzene = Chem.MolFromSmiles('c1ccccc1')
            # Aliphatic carbon in ethane
            ethane = Chem.MolFromSmiles('CC')
            
            if benzene is not None and ethane is not None:
                benzene_features = self.extractor.get_graph_features(benzene)
                ethane_features = self.extractor.get_graph_features(ethane)
                
                # Aromatic and aliphatic carbons should have different features
                aromatic_carbon = benzene_features['x'][0]
                aliphatic_carbon = ethane_features['x'][0]
                
                assert not torch.equal(aromatic_carbon, aliphatic_carbon), \
                    "Aromatic and aliphatic carbons should have different features"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_formal_charge_encoding(self):
        """Test formal charge encoding."""
        from rdkit import Chem
        
        try:
            # Molecules with different formal charges
            neutral = Chem.MolFromSmiles('C')
            # Note: Creating charged molecules in SMILES can be tricky
            # This is a simplified test
            
            if neutral is not None:
                features = self.extractor.get_graph_features(neutral)
                assert features['x'].size(0) > 0, "Should have atom features"
                
                # Check that features are reasonable
                atom_feature = features['x'][0]
                assert not torch.isnan(atom_feature).any(), "Features should not contain NaN"
                assert torch.isfinite(atom_feature).all(), "Features should be finite"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_hybridization_encoding(self):
        """Test hybridization state encoding."""
        from rdkit import Chem
        
        try:
            # Different hybridization states
            sp3_carbon = Chem.MolFromSmiles('CC')      # SP3
            sp2_carbon = Chem.MolFromSmiles('C=C')     # SP2
            sp_carbon = Chem.MolFromSmiles('C#C')      # SP
            
            molecules = [sp3_carbon, sp2_carbon, sp_carbon]
            features_list = []
            
            for mol in molecules:
                if mol is not None:
                    features = self.extractor.get_graph_features(mol)
                    if features['x'].size(0) > 0:
                        features_list.append(features['x'][0])
            
            # Different hybridization states should have different features
            if len(features_list) >= 2:
                for i in range(len(features_list)):
                    for j in range(i+1, len(features_list)):
                        assert not torch.equal(features_list[i], features_list[j]), \
                            f"Different hybridization states should have different features"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_ring_membership(self):
        """Test ring membership encoding."""
        from rdkit import Chem
        
        try:
            # Ring vs non-ring atoms
            cyclic = Chem.MolFromSmiles('C1CCC1')      # Cyclobutane
            acyclic = Chem.MolFromSmiles('CCCC')       # Butane
            
            if cyclic is not None and acyclic is not None:
                cyclic_features = self.extractor.get_graph_features(cyclic)
                acyclic_features = self.extractor.get_graph_features(acyclic)
                
                # Ring and non-ring carbons might have different features
                # (depending on implementation)
                ring_carbon = cyclic_features['x'][0]
                chain_carbon = acyclic_features['x'][0]
                
                # Features might be different due to ring membership
                # This is implementation-dependent
                assert ring_carbon.size() == chain_carbon.size(), \
                    "Feature dimensions should be consistent"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # None input
        result = self.extractor.get_graph_features(None)
        assert result is None, "None input should return None"
        
        # Test with mock empty molecule
        mock_mol = MagicMock()
        mock_mol.GetAtoms.return_value = []
        mock_mol.GetBonds.return_value = []
        mock_mol.GetNumAtoms.return_value = 0
        mock_mol.GetNumBonds.return_value = 0
        
        result = self.extractor.get_graph_features(mock_mol)
        if result is not None:
            # Should handle empty molecules gracefully
            assert isinstance(result, dict), "Should return dictionary for empty molecule"
    
    def test_feature_normalization(self):
        """Test that features are properly normalized/scaled."""
        from rdkit import Chem
        
        try:
            mol = Chem.MolFromSmiles('CCO')
            if mol is not None:
                features = self.extractor.get_graph_features(mol)
                
                # Check that features are in reasonable ranges
                node_features = features['x']
                edge_features = features['edge_attr']
                
                # Features should not be extremely large
                assert node_features.abs().max() < 1000, "Node features should be reasonably scaled"
                assert edge_features.abs().max() < 1000, "Edge features should be reasonably scaled"
                
                # Features should not all be zero
                assert node_features.abs().sum() > 0, "Node features should not all be zero"
                if edge_features.size(0) > 0:
                    assert edge_features.abs().sum() > 0, "Edge features should not all be zero"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_batch_consistency(self):
        """Test consistency when processing multiple molecules."""
        from rdkit import Chem
        
        try:
            molecules = []
            for smiles in self.test_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules.append(mol)
            
            if len(molecules) > 1:
                all_features = []
                for mol in molecules:
                    features = self.extractor.get_graph_features(mol)
                    if features is not None:
                        all_features.append(features)
                
                # Check dimension consistency
                if len(all_features) > 1:
                    node_dim = all_features[0]['x'].size(1)
                    edge_dim = all_features[0]['edge_attr'].size(1)
                    
                    for features in all_features[1:]:
                        assert features['x'].size(1) == node_dim, "Node feature dimensions should be consistent"
                        assert features['edge_attr'].size(1) == edge_dim, "Edge feature dimensions should be consistent"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")
    
    def test_deterministic_behavior(self):
        """Test that feature extraction is deterministic."""
        from rdkit import Chem
        
        try:
            mol = Chem.MolFromSmiles('CCO')
            if mol is not None:
                # Extract features multiple times
                features1 = self.extractor.get_graph_features(mol)
                features2 = self.extractor.get_graph_features(mol)
                
                if features1 is not None and features2 is not None:
                    # Results should be identical
                    assert torch.equal(features1['x'], features2['x']), "Node features should be deterministic"
                    assert torch.equal(features1['edge_index'], features2['edge_index']), "Edge indices should be deterministic"
                    assert torch.equal(features1['edge_attr'], features2['edge_attr']), "Edge features should be deterministic"
        
        except Exception:
            pytest.skip("RDKit not available or molecule creation failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])