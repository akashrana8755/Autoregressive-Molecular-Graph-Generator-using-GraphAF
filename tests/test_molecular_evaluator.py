"""
Tests for the MolecularEvaluator class.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Test data
VALID_SMILES = [
    'CCO',  # ethanol
    'CC(=O)O',  # acetic acid
    'c1ccccc1',  # benzene
    'CCN(CC)CC',  # triethylamine
    'CC(C)O'  # isopropanol
]

INVALID_SMILES = [
    'invalid_smiles',
    'C[C@H](C)O[C@H]1C[C@@H]2C[C@H]',  # incomplete structure
    '',  # empty string
    'XYZ123'  # nonsense
]

DUPLICATE_SMILES = [
    'CCO',
    'CCO',  # duplicate
    'OCC',  # same as CCO in canonical form
    'CC(=O)O',
    'CC(=O)O'  # duplicate
]


class TestMolecularEvaluator:
    """Test cases for MolecularEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock RDKit availability
        self.rdkit_patcher = patch('src.evaluate.molecular_evaluator.RDKIT_AVAILABLE', True)
        self.rdkit_patcher.start()
        
        # Import after patching
        from src.evaluate.molecular_evaluator import MolecularEvaluator
        self.MolecularEvaluator = MolecularEvaluator
    
    def teardown_method(self):
        """Clean up after tests."""
        self.rdkit_patcher.stop()
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_init_without_reference(self, mock_chem):
        """Test initialization without reference molecules."""
        evaluator = self.MolecularEvaluator()
        assert evaluator.reference_molecules == set()
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_init_with_reference(self, mock_chem):
        """Test initialization with reference molecules."""
        reference = ['CCO', 'CC(=O)O']
        evaluator = self.MolecularEvaluator(reference_molecules=reference)
        assert evaluator.reference_molecules == set(reference)
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_validity_all_valid(self, mock_chem):
        """Test validity computation with all valid molecules."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 5
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        
        evaluator = self.MolecularEvaluator()
        validity = evaluator.compute_validity(VALID_SMILES)
        
        assert validity == 1.0
        assert mock_chem.MolFromSmiles.call_count == len(VALID_SMILES)
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_validity_mixed(self, mock_chem):
        """Test validity computation with mixed valid/invalid molecules."""
        def mock_mol_from_smiles(smiles):
            if smiles in VALID_SMILES:
                mock_mol = MagicMock()
                mock_mol.GetNumAtoms.return_value = 5
                return mock_mol
            return None
        
        mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles
        mock_chem.SanitizeMol.return_value = None
        
        evaluator = self.MolecularEvaluator()
        mixed_molecules = VALID_SMILES + INVALID_SMILES
        validity = evaluator.compute_validity(mixed_molecules)
        
        expected_validity = len(VALID_SMILES) / len(mixed_molecules)
        assert abs(validity - expected_validity) < 1e-6
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_validity_empty_list(self, mock_chem):
        """Test validity computation with empty molecule list."""
        evaluator = self.MolecularEvaluator()
        validity = evaluator.compute_validity([])
        assert validity == 0.0
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_uniqueness_all_unique(self, mock_chem):
        """Test uniqueness computation with all unique molecules."""
        # Mock canonicalization to return the same SMILES
        def mock_mol_to_smiles(mol, canonical=True):
            return VALID_SMILES[mock_chem.MolToSmiles.call_count - 1]
        
        mock_mol = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolToSmiles.side_effect = mock_mol_to_smiles
        
        evaluator = self.MolecularEvaluator()
        uniqueness = evaluator.compute_uniqueness(VALID_SMILES)
        
        assert uniqueness == 1.0
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_uniqueness_with_duplicates(self, mock_chem):
        """Test uniqueness computation with duplicate molecules."""
        # Mock canonicalization
        canonical_map = {
            'CCO': 'CCO',
            'OCC': 'CCO',  # Same canonical form
            'CC(=O)O': 'CC(=O)O'
        }
        
        def mock_mol_from_smiles(smiles):
            if smiles in canonical_map:
                return MagicMock()
            return None
        
        def mock_mol_to_smiles(mol, canonical=True):
            # Return based on call order
            smiles_list = ['CCO', 'CCO', 'CCO', 'CC(=O)O', 'CC(=O)O']
            return smiles_list[mock_chem.MolToSmiles.call_count - 1]
        
        mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles
        mock_chem.MolToSmiles.side_effect = mock_mol_to_smiles
        
        evaluator = self.MolecularEvaluator()
        uniqueness = evaluator.compute_uniqueness(DUPLICATE_SMILES)
        
        # Should have 2 unique molecules out of 5 total
        expected_uniqueness = 2.0 / 5.0
        assert abs(uniqueness - expected_uniqueness) < 1e-6
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_novelty_with_reference(self, mock_chem):
        """Test novelty computation with reference molecules."""
        reference = ['CCO', 'CC(=O)O']
        generated = ['CCO', 'c1ccccc1', 'CCN(CC)CC']  # 1 known, 2 novel
        
        # Mock canonicalization to return the same SMILES
        def mock_mol_from_smiles(smiles):
            return MagicMock() if smiles in reference + generated else None
        
        def mock_mol_to_smiles(mol, canonical=True):
            # Return based on call order
            all_smiles = reference + generated + generated  # reference + 2x generated
            return all_smiles[mock_chem.MolToSmiles.call_count - 1]
        
        mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles
        mock_chem.MolToSmiles.side_effect = mock_mol_to_smiles
        
        evaluator = self.MolecularEvaluator(reference_molecules=reference)
        novelty = evaluator.compute_novelty(generated)
        
        # 2 novel out of 3 generated
        expected_novelty = 2.0 / 3.0
        assert abs(novelty - expected_novelty) < 1e-6
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_compute_novelty_without_reference(self, mock_chem):
        """Test novelty computation without reference molecules."""
        evaluator = self.MolecularEvaluator()
        novelty = evaluator.compute_novelty(VALID_SMILES)
        assert novelty == 0.0
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_evaluate_comprehensive(self, mock_chem):
        """Test comprehensive evaluation with all metrics."""
        # Mock RDKit functions for comprehensive test
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 5
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        mock_chem.MolToSmiles.return_value = 'CCO'
        
        reference = ['CC(=O)O']
        generated = ['CCO', 'CCO', 'c1ccccc1']
        
        evaluator = self.MolecularEvaluator(reference_molecules=reference)
        results = evaluator.evaluate(generated)
        
        assert 'validity' in results
        assert 'uniqueness' in results
        assert 'novelty' in results
        assert 'total_molecules' in results
        assert results['total_molecules'] == len(generated)
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_evaluate_empty_list(self, mock_chem):
        """Test evaluation with empty molecule list."""
        evaluator = self.MolecularEvaluator()
        results = evaluator.evaluate([])
        
        expected = {
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': None,
            'total_molecules': 0
        }
        assert results == expected
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_get_valid_molecules(self, mock_chem):
        """Test filtering for valid molecules only."""
        def mock_mol_from_smiles(smiles):
            if smiles in VALID_SMILES:
                mock_mol = MagicMock()
                mock_mol.GetNumAtoms.return_value = 5
                return mock_mol
            return None
        
        mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles
        mock_chem.SanitizeMol.return_value = None
        
        evaluator = self.MolecularEvaluator()
        mixed_molecules = VALID_SMILES + INVALID_SMILES
        valid_molecules = evaluator.get_valid_molecules(mixed_molecules)
        
        assert len(valid_molecules) == len(VALID_SMILES)
        for smiles in valid_molecules:
            assert smiles in VALID_SMILES
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    def test_get_unique_molecules(self, mock_chem):
        """Test filtering for unique molecules only."""
        # Mock canonicalization
        canonical_map = {
            'CCO': 'CCO',
            'OCC': 'CCO',  # Same canonical form
            'CC(=O)O': 'CC(=O)O'
        }
        
        def mock_mol_from_smiles(smiles):
            return MagicMock() if smiles in canonical_map else None
        
        def mock_mol_to_smiles(mol, canonical=True):
            # Return canonical forms in order
            canonical_forms = ['CCO', 'CCO', 'CC(=O)O', 'CC(=O)O']
            return canonical_forms[mock_chem.MolToSmiles.call_count - 1]
        
        mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles
        mock_chem.MolToSmiles.side_effect = mock_mol_to_smiles
        
        evaluator = self.MolecularEvaluator()
        unique_molecules = evaluator.get_unique_molecules(DUPLICATE_SMILES)
        
        # Should return 2 unique canonical forms
        assert len(unique_molecules) == 2
        assert 'CCO' in unique_molecules
        assert 'CC(=O)O' in unique_molecules
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_compute_property_distributions(self, mock_descriptors, mock_chem):
        """Test property distribution computation."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 10
        mock_mol.GetNumBonds.return_value = 9
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        
        # Mock descriptor calculations
        mock_descriptors.MolWt.return_value = 180.0
        mock_descriptors.MolLogP.return_value = 2.5
        mock_descriptors.RingCount.return_value = 1
        mock_descriptors.TPSA.return_value = 40.0
        
        evaluator = self.MolecularEvaluator()
        properties = evaluator.compute_property_distributions(['CCO', 'CC(=O)O'])
        
        assert 'molecular_weight' in properties
        assert 'logp' in properties
        assert 'num_atoms' in properties
        assert 'num_bonds' in properties
        assert 'num_rings' in properties
        assert 'tpsa' in properties
        
        # Check that arrays have correct length
        assert len(properties['molecular_weight']) == 2
        assert len(properties['logp']) == 2
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_compare_property_distributions(self, mock_descriptors, mock_chem):
        """Test property distribution comparison."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 10
        mock_mol.GetNumBonds.return_value = 9
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        
        # Mock descriptor calculations with different values for generated vs reference
        def mock_molwt(mol):
            return 180.0 if mock_descriptors.MolWt.call_count <= 2 else 200.0
        
        def mock_logp(mol):
            return 2.5 if mock_descriptors.MolLogP.call_count <= 2 else 3.0
        
        mock_descriptors.MolWt.side_effect = mock_molwt
        mock_descriptors.MolLogP.side_effect = mock_logp
        mock_descriptors.RingCount.return_value = 1
        mock_descriptors.TPSA.return_value = 40.0
        
        evaluator = self.MolecularEvaluator()
        generated = ['CCO', 'CC(=O)O']
        reference = ['c1ccccc1', 'CCN(CC)CC']
        
        comparison = evaluator.compare_property_distributions(generated, reference)
        
        assert 'molecular_weight' in comparison
        assert 'logp' in comparison
        
        # Check that comparison contains expected keys
        mw_comparison = comparison['molecular_weight']
        assert 'generated_mean' in mw_comparison
        assert 'reference_mean' in mw_comparison
        assert 'mean_difference' in mw_comparison
        assert 'ks_statistic' in mw_comparison
        assert 'wasserstein_distance' in mw_comparison
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_generate_evaluation_report(self, mock_descriptors, mock_chem):
        """Test comprehensive evaluation report generation."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 10
        mock_mol.GetNumBonds.return_value = 9
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        mock_chem.MolToSmiles.return_value = 'CCO'
        
        # Mock descriptor calculations
        mock_descriptors.MolWt.return_value = 180.0
        mock_descriptors.MolLogP.return_value = 2.5
        mock_descriptors.RingCount.return_value = 1
        mock_descriptors.TPSA.return_value = 40.0
        
        evaluator = self.MolecularEvaluator()
        generated = ['CCO', 'CC(=O)O']
        reference = ['c1ccccc1']
        
        report = evaluator.generate_evaluation_report(generated, reference)
        
        # Check report structure
        assert 'basic_metrics' in report
        assert 'diversity_metrics' in report
        assert 'property_distributions' in report
        assert 'property_comparison' in report
        assert 'summary' in report
        
        # Check summary content
        summary = report['summary']
        assert 'total_generated' in summary
        assert 'valid_count' in summary
        assert 'unique_count' in summary
        assert 'validity_rate' in summary
        assert 'uniqueness_rate' in summary
        
        assert summary['total_generated'] == 2
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.QED')
    def test_compute_qed_scores(self, mock_qed, mock_chem):
        """Test QED score computation."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 5
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        mock_qed.qed.return_value = 0.75
        
        evaluator = self.MolecularEvaluator()
        qed_scores = evaluator.compute_qed_scores(['CCO', 'CC(=O)O'])
        
        assert len(qed_scores) == 2
        assert all(score == 0.75 for score in qed_scores)
        assert mock_qed.qed.call_count == 2
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_compute_lipinski_compliance(self, mock_descriptors, mock_chem):
        """Test Lipinski Rule of Five compliance computation."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 5
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        
        # Mock descriptor calculations (passing Lipinski rules)
        mock_descriptors.MolWt.return_value = 400.0  # <= 500
        mock_descriptors.MolLogP.return_value = 3.0  # <= 5
        mock_descriptors.NumHDonors.return_value = 2  # <= 5
        mock_descriptors.NumHAcceptors.return_value = 4  # <= 10
        
        evaluator = self.MolecularEvaluator()
        compliance = evaluator.compute_lipinski_compliance(['CCO', 'CC(=O)O'])
        
        assert 'molecular_weight_ok' in compliance
        assert 'logp_ok' in compliance
        assert 'hbd_ok' in compliance
        assert 'hba_ok' in compliance
        assert 'lipinski_pass' in compliance
        
        # All should pass with the mocked values
        assert all(compliance['molecular_weight_ok'])
        assert all(compliance['logp_ok'])
        assert all(compliance['hbd_ok'])
        assert all(compliance['hba_ok'])
        assert all(compliance['lipinski_pass'])
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.QED')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_compute_drug_likeness_metrics(self, mock_descriptors, mock_qed, mock_chem):
        """Test comprehensive drug-likeness metrics computation."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 5
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        
        # Mock QED and Lipinski calculations
        mock_qed.qed.return_value = 0.8
        mock_descriptors.MolWt.return_value = 300.0
        mock_descriptors.MolLogP.return_value = 2.0
        mock_descriptors.NumHDonors.return_value = 1
        mock_descriptors.NumHAcceptors.return_value = 3
        
        evaluator = self.MolecularEvaluator()
        metrics = evaluator.compute_drug_likeness_metrics(['CCO', 'CC(=O)O'])
        
        expected_keys = [
            'mean_qed', 'median_qed', 'lipinski_pass_rate',
            'mw_pass_rate', 'logp_pass_rate', 'hbd_pass_rate', 'hba_pass_rate'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # With mocked values, all should pass
        assert metrics['mean_qed'] == 0.8
        assert metrics['median_qed'] == 0.8
        assert metrics['lipinski_pass_rate'] == 1.0
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.QED')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_filter_drug_like_molecules(self, mock_descriptors, mock_qed, mock_chem):
        """Test filtering of drug-like molecules."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 5
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        
        # Mock QED scores: first molecule passes, second fails
        qed_values = [0.7, 0.3]  # threshold = 0.5
        mock_qed.qed.side_effect = qed_values
        
        # Mock Lipinski compliance (all pass)
        mock_descriptors.MolWt.return_value = 300.0
        mock_descriptors.MolLogP.return_value = 2.0
        mock_descriptors.NumHDonors.return_value = 1
        mock_descriptors.NumHAcceptors.return_value = 3
        
        evaluator = self.MolecularEvaluator()
        drug_like = evaluator.filter_drug_like_molecules(
            ['CCO', 'CC(=O)O'], 
            qed_threshold=0.5, 
            require_lipinski=True
        )
        
        # Only first molecule should pass (QED = 0.7 > 0.5)
        assert len(drug_like) == 1
        assert drug_like[0] == 'CCO'
    
    @patch('src.evaluate.molecular_evaluator.Chem')
    @patch('src.evaluate.molecular_evaluator.QED')
    @patch('src.evaluate.molecular_evaluator.Descriptors')
    def test_generate_comprehensive_report(self, mock_descriptors, mock_qed, mock_chem):
        """Test comprehensive report generation with drug-likeness metrics."""
        # Mock RDKit functions
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 10
        mock_mol.GetNumBonds.return_value = 9
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.SanitizeMol.return_value = None
        mock_chem.MolToSmiles.return_value = 'CCO'
        
        # Mock all descriptor calculations
        mock_qed.qed.return_value = 0.8
        mock_descriptors.MolWt.return_value = 180.0
        mock_descriptors.MolLogP.return_value = 2.5
        mock_descriptors.NumHDonors.return_value = 1
        mock_descriptors.NumHAcceptors.return_value = 2
        mock_descriptors.RingCount.return_value = 1
        mock_descriptors.TPSA.return_value = 40.0
        
        evaluator = self.MolecularEvaluator()
        generated = ['CCO', 'CC(=O)O']
        reference = ['c1ccccc1']
        
        report = evaluator.generate_comprehensive_report(generated, reference)
        
        # Check that all sections are present
        expected_sections = [
            'basic_metrics', 'diversity_metrics', 'property_distributions',
            'property_comparison', 'summary', 'drug_likeness', 'qed_scores',
            'lipinski_compliance', 'drug_likeness_comparison'
        ]
        
        for section in expected_sections:
            assert section in report, f"Missing section: {section}"
        
        # Check drug-likeness specific content
        assert 'mean_qed' in report['drug_likeness']
        assert 'lipinski_pass_rate' in report['drug_likeness']
        assert len(report['qed_scores']) == 2
        assert 'lipinski_pass' in report['lipinski_compliance']
    
    def test_rdkit_not_available(self):
        """Test behavior when RDKit is not available."""
        with patch('src.evaluate.molecular_evaluator.RDKIT_AVAILABLE', False):
            with pytest.raises(ImportError, match="RDKit is required"):
                from src.evaluate.molecular_evaluator import MolecularEvaluator
                MolecularEvaluator()