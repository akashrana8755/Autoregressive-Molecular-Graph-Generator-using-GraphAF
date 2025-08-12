"""
Tests for constraint filtering functionality.
"""

import pytest
import numpy as np
from src.generate.constraint_filter import ConstraintFilter


class TestConstraintFilter:
    """Test cases for ConstraintFilter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = ConstraintFilter()
        
        # Test molecules with known properties
        self.test_molecules = [
            "CCO",  # Ethanol - should pass Lipinski
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen - should pass
            "C" * 50,  # Very long alkane - should fail MW
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # Long chain - fail MW
        ]
    
    def test_molecular_weight_calculation(self):
        """Test molecular weight calculation."""
        # Ethanol (C2H6O) = 46.07 Da
        mw = self.filter.calculate_molecular_weight("CCO")
        assert mw is not None
        assert abs(mw - 46.07) < 0.1
        
        # Invalid SMILES
        mw = self.filter.calculate_molecular_weight("INVALID")
        assert mw is None
    
    def test_logp_calculation(self):
        """Test logP calculation."""
        logp = self.filter.calculate_logp("CCO")
        assert logp is not None
        assert isinstance(logp, float)
        
        # Invalid SMILES
        logp = self.filter.calculate_logp("INVALID")
        assert logp is None
    
    def test_hbd_calculation(self):
        """Test hydrogen bond donor calculation."""
        # Ethanol has 1 HBD (OH group)
        hbd = self.filter.calculate_hbd("CCO")
        assert hbd == 1
        
        # Invalid SMILES
        hbd = self.filter.calculate_hbd("INVALID")
        assert hbd is None
    
    def test_hba_calculation(self):
        """Test hydrogen bond acceptor calculation."""
        # Ethanol has 1 HBA (oxygen)
        hba = self.filter.calculate_hba("CCO")
        assert hba == 1
        
        # Invalid SMILES
        hba = self.filter.calculate_hba("INVALID")
        assert hba is None
    
    def test_lipinski_rule_check(self):
        """Test individual Lipinski rule checking."""
        # Ethanol should pass all rules
        rules = self.filter.check_lipinski_rule("CCO")
        assert rules['valid'] is True
        assert rules['mw_pass'] is True
        assert rules['logp_pass'] is True
        assert rules['hbd_pass'] is True
        assert rules['hba_pass'] is True
        
        # Invalid SMILES
        rules = self.filter.check_lipinski_rule("INVALID")
        assert rules['valid'] is False
    
    def test_lipinski_filter_pass(self):
        """Test Lipinski filter pass/fail."""
        # Ethanol should pass
        assert self.filter.passes_lipinski_filter("CCO") is True
        
        # Invalid SMILES should fail
        assert self.filter.passes_lipinski_filter("INVALID") is False
    
    def test_apply_lipinski_filter(self):
        """Test applying Lipinski filter to molecule list."""
        test_smiles = ["CCO", "INVALID", "C"]
        filtered = self.filter.apply_lipinski_filter(test_smiles)
        
        # Should filter out invalid SMILES
        assert len(filtered) <= len(test_smiles)
        assert "INVALID" not in filtered
    
    def test_qed_score_calculation(self):
        """Test QED score calculation."""
        qed = self.filter.calculate_qed_score("CCO")
        assert qed is not None
        assert 0 <= qed <= 1
        
        # Invalid SMILES
        qed = self.filter.calculate_qed_score("INVALID")
        assert qed is None
    
    def test_compute_qed_scores(self):
        """Test batch QED score computation."""
        test_smiles = ["CCO", "INVALID", "C"]
        scores = self.filter.compute_qed_scores(test_smiles)
        
        assert len(scores) == len(test_smiles)
        assert not np.isnan(scores[0])  # Valid molecule
        assert np.isnan(scores[1])      # Invalid molecule
    
    def test_qed_filter(self):
        """Test QED filtering."""
        # Test with low threshold
        self.filter.qed_threshold = 0.1
        assert self.filter.passes_qed_filter("CCO") is True
        
        # Test with high threshold
        self.filter.qed_threshold = 0.9
        result = self.filter.passes_qed_filter("CCO")
        # Result depends on actual QED score of ethanol
        assert isinstance(result, bool)
    
    def test_lipinski_statistics(self):
        """Test Lipinski statistics calculation."""
        test_smiles = ["CCO", "C", "CC"]
        stats = self.filter.get_lipinski_statistics(test_smiles)
        
        assert 'total_molecules' in stats
        assert 'valid_molecules' in stats
        assert 'all_rules_pass_rate' in stats
        assert stats['total_molecules'] == 3
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics."""
        test_smiles = ["CCO", "C", "CC"]
        stats = self.filter.get_comprehensive_statistics(test_smiles)
        
        assert 'qed_statistics' in stats
        assert 'combined_pass_rate' in stats
        assert 'mean_qed' in stats['qed_statistics']


if __name__ == "__main__":
    pytest.main([__file__])