"""
Constraint filtering for drug-likeness evaluation.

This module implements Lipinski's Rule of Five and QED score filtering
for molecular drug-likeness assessment.
"""

from typing import List, Dict, Tuple, Optional
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import numpy as np

logger = logging.getLogger(__name__)


class ConstraintFilter:
    """
    Applies drug-likeness and property constraints to molecular structures.
    
    Implements Lipinski's Rule of Five filtering and QED score calculation
    for drug-likeness assessment.
    """
    
    def __init__(self, 
                 mw_threshold: float = 500.0,
                 logp_threshold: float = 5.0,
                 hbd_threshold: int = 5,
                 hba_threshold: int = 10,
                 qed_threshold: float = 0.5):
        """
        Initialize constraint filter with Lipinski rule thresholds.
        
        Args:
            mw_threshold: Maximum molecular weight (Da)
            logp_threshold: Maximum logP value
            hbd_threshold: Maximum hydrogen bond donors
            hba_threshold: Maximum hydrogen bond acceptors
            qed_threshold: Minimum QED score for drug-likeness
        """
        self.mw_threshold = mw_threshold
        self.logp_threshold = logp_threshold
        self.hbd_threshold = hbd_threshold
        self.hba_threshold = hba_threshold
        self.qed_threshold = qed_threshold
        
    def calculate_molecular_weight(self, smiles: str) -> Optional[float]:
        """
        Calculate molecular weight for a SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Molecular weight in Da, or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.MolWt(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate molecular weight for {smiles}: {e}")
            return None
    
    def calculate_logp(self, smiles: str) -> Optional[float]:
        """
        Calculate logP (partition coefficient) for a SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            LogP value, or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.MolLogP(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate logP for {smiles}: {e}")
            return None
    
    def calculate_hbd(self, smiles: str) -> Optional[int]:
        """
        Calculate number of hydrogen bond donors.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Number of hydrogen bond donors, or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.NumHDonors(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate HBD for {smiles}: {e}")
            return None
    
    def calculate_hba(self, smiles: str) -> Optional[int]:
        """
        Calculate number of hydrogen bond acceptors.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Number of hydrogen bond acceptors, or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.NumHAcceptors(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate HBA for {smiles}: {e}")
            return None  
  
    def check_lipinski_rule(self, smiles: str) -> Dict[str, bool]:
        """
        Check if a molecule satisfies individual Lipinski rules.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Dictionary with boolean values for each Lipinski rule
        """
        results = {
            'mw_pass': False,
            'logp_pass': False,
            'hbd_pass': False,
            'hba_pass': False,
            'valid': False
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return results
            
            results['valid'] = True
            
            # Molecular weight < 500 Da
            mw = Descriptors.MolWt(mol)
            results['mw_pass'] = mw <= self.mw_threshold
            
            # LogP < 5
            logp = Descriptors.MolLogP(mol)
            results['logp_pass'] = logp <= self.logp_threshold
            
            # Hydrogen bond donors ≤ 5
            hbd = Descriptors.NumHDonors(mol)
            results['hbd_pass'] = hbd <= self.hbd_threshold
            
            # Hydrogen bond acceptors ≤ 10
            hba = Descriptors.NumHAcceptors(mol)
            results['hba_pass'] = hba <= self.hba_threshold
            
        except Exception as e:
            logger.warning(f"Failed to check Lipinski rules for {smiles}: {e}")
            
        return results
    
    def passes_lipinski_filter(self, smiles: str) -> bool:
        """
        Check if a molecule passes all Lipinski rules.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            True if molecule passes all four Lipinski rules
        """
        rules = self.check_lipinski_rule(smiles)
        if not rules['valid']:
            return False
            
        return all([
            rules['mw_pass'],
            rules['logp_pass'], 
            rules['hbd_pass'],
            rules['hba_pass']
        ])
    
    def apply_lipinski_filter(self, smiles_list: List[str]) -> List[str]:
        """
        Filter a list of SMILES strings using Lipinski's Rule of Five.
        
        Args:
            smiles_list: List of SMILES strings to filter
            
        Returns:
            List of SMILES strings that pass all Lipinski rules
        """
        filtered_smiles = []
        
        for smiles in smiles_list:
            if self.passes_lipinski_filter(smiles):
                filtered_smiles.append(smiles)
        
        logger.info(f"Lipinski filter: {len(filtered_smiles)}/{len(smiles_list)} molecules passed")
        return filtered_smiles
    
    def get_lipinski_statistics(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Calculate Lipinski rule pass rates for a list of molecules.
        
        Args:
            smiles_list: List of SMILES strings to analyze
            
        Returns:
            Dictionary with pass rates for each rule and overall
        """
        stats = {
            'total_molecules': len(smiles_list),
            'valid_molecules': 0,
            'mw_pass_rate': 0.0,
            'logp_pass_rate': 0.0,
            'hbd_pass_rate': 0.0,
            'hba_pass_rate': 0.0,
            'all_rules_pass_rate': 0.0
        }
        
        if len(smiles_list) == 0:
            return stats
        
        rule_counts = {
            'valid': 0,
            'mw_pass': 0,
            'logp_pass': 0,
            'hbd_pass': 0,
            'hba_pass': 0,
            'all_pass': 0
        }
        
        for smiles in smiles_list:
            rules = self.check_lipinski_rule(smiles)
            
            if rules['valid']:
                rule_counts['valid'] += 1
                rule_counts['mw_pass'] += int(rules['mw_pass'])
                rule_counts['logp_pass'] += int(rules['logp_pass'])
                rule_counts['hbd_pass'] += int(rules['hbd_pass'])
                rule_counts['hba_pass'] += int(rules['hba_pass'])
                
                if all([rules['mw_pass'], rules['logp_pass'], 
                       rules['hbd_pass'], rules['hba_pass']]):
                    rule_counts['all_pass'] += 1
        
        stats['valid_molecules'] = rule_counts['valid']
        
        if rule_counts['valid'] > 0:
            stats['mw_pass_rate'] = rule_counts['mw_pass'] / rule_counts['valid']
            stats['logp_pass_rate'] = rule_counts['logp_pass'] / rule_counts['valid']
            stats['hbd_pass_rate'] = rule_counts['hbd_pass'] / rule_counts['valid']
            stats['hba_pass_rate'] = rule_counts['hba_pass'] / rule_counts['valid']
            stats['all_rules_pass_rate'] = rule_counts['all_pass'] / rule_counts['valid']
        
        return stats
    
    def calculate_qed_score(self, smiles: str) -> Optional[float]:
        """
        Calculate QED (drug-likeness) score for a SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            QED score (0-1), or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return QED.qed(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate QED score for {smiles}: {e}")
            return None
    
    def compute_qed_scores(self, smiles_list: List[str]) -> List[float]:
        """
        Compute QED scores for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of QED scores (NaN for failed calculations)
        """
        qed_scores = []
        
        for smiles in smiles_list:
            qed_score = self.calculate_qed_score(smiles)
            if qed_score is not None:
                qed_scores.append(qed_score)
            else:
                qed_scores.append(np.nan)
        
        return qed_scores
    
    def passes_qed_filter(self, smiles: str) -> bool:
        """
        Check if a molecule passes QED threshold filter.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            True if QED score >= threshold
        """
        qed_score = self.calculate_qed_score(smiles)
        if qed_score is None:
            return False
        return qed_score >= self.qed_threshold
    
    def apply_qed_filter(self, smiles_list: List[str]) -> List[str]:
        """
        Filter molecules based on QED score threshold.
        
        Args:
            smiles_list: List of SMILES strings to filter
            
        Returns:
            List of SMILES strings with QED >= threshold
        """
        filtered_smiles = []
        
        for smiles in smiles_list:
            if self.passes_qed_filter(smiles):
                filtered_smiles.append(smiles)
        
        logger.info(f"QED filter: {len(filtered_smiles)}/{len(smiles_list)} molecules passed")
        return filtered_smiles
    
    def apply_property_filter(self, smiles_list: List[str], 
                            constraints: Optional[Dict] = None) -> List[str]:
        """
        Apply combined property constraints to filter molecules.
        
        Args:
            smiles_list: List of SMILES strings to filter
            constraints: Optional custom constraints dict
            
        Returns:
            List of SMILES strings passing all constraints
        """
        if constraints is None:
            constraints = {}
        
        # Apply Lipinski filter
        filtered_smiles = self.apply_lipinski_filter(smiles_list)
        
        # Apply QED filter if threshold specified
        qed_threshold = constraints.get('qed_threshold', self.qed_threshold)
        if qed_threshold > 0:
            old_threshold = self.qed_threshold
            self.qed_threshold = qed_threshold
            filtered_smiles = self.apply_qed_filter(filtered_smiles)
            self.qed_threshold = old_threshold
        
        return filtered_smiles
    
    def get_comprehensive_statistics(self, smiles_list: List[str]) -> Dict[str, any]:
        """
        Get comprehensive drug-likeness statistics for a molecule set.
        
        Args:
            smiles_list: List of SMILES strings to analyze
            
        Returns:
            Dictionary with detailed statistics
        """
        stats = self.get_lipinski_statistics(smiles_list)
        
        # Add QED statistics
        qed_scores = self.compute_qed_scores(smiles_list)
        valid_qed_scores = [score for score in qed_scores if not np.isnan(score)]
        
        stats['qed_statistics'] = {
            'mean_qed': np.mean(valid_qed_scores) if valid_qed_scores else 0.0,
            'std_qed': np.std(valid_qed_scores) if valid_qed_scores else 0.0,
            'median_qed': np.median(valid_qed_scores) if valid_qed_scores else 0.0,
            'qed_pass_rate': sum(1 for score in valid_qed_scores 
                               if score >= self.qed_threshold) / len(valid_qed_scores) 
                               if valid_qed_scores else 0.0
        }
        
        # Combined filter statistics
        combined_pass = 0
        for smiles in smiles_list:
            if self.passes_lipinski_filter(smiles) and self.passes_qed_filter(smiles):
                combined_pass += 1
        
        stats['combined_pass_rate'] = combined_pass / len(smiles_list) if smiles_list else 0.0
        
        return stats