"""
Molecular evaluation framework for assessing generated molecules.

This module provides comprehensive evaluation metrics for generated molecules
including validity, uniqueness, novelty, and drug-likeness assessments.
"""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from collections import Counter
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Some evaluation metrics will be disabled.")

# Optional scipy import for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MolecularEvaluator:
    """
    Comprehensive evaluation framework for generated molecules.
    
    Provides metrics for validity, uniqueness, novelty, and drug-likeness
    assessment of generated molecular structures.
    """
    
    def __init__(self, reference_molecules: Optional[List[str]] = None):
        """
        Initialize the molecular evaluator.
        
        Args:
            reference_molecules: List of reference SMILES strings for novelty computation
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular evaluation")
            
        self.reference_molecules = set(reference_molecules) if reference_molecules else set()
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, generated_molecules: List[str]) -> Dict[str, float]:
        """
        Perform comprehensive evaluation of generated molecules.
        
        Args:
            generated_molecules: List of generated SMILES strings
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if not generated_molecules:
            return {
                'validity': 0.0,
                'uniqueness': 0.0,
                'novelty': 0.0 if self.reference_molecules else None,
                'total_molecules': 0
            }
        
        # Compute basic metrics
        validity = self.compute_validity(generated_molecules)
        uniqueness = self.compute_uniqueness(generated_molecules)
        
        # Compute novelty if reference molecules are available
        novelty = None
        if self.reference_molecules:
            novelty = self.compute_novelty(generated_molecules)
        
        results = {
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'total_molecules': len(generated_molecules)
        }
        
        self.logger.info(f"Evaluation results: {results}")
        return results
    
    def compute_validity(self, molecules: List[str]) -> float:
        """
        Compute the percentage of chemically valid molecules.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Validity percentage (0.0 to 1.0)
        """
        if not molecules:
            return 0.0
        
        valid_count = 0
        for smiles in molecules:
            if self._is_valid_molecule(smiles):
                valid_count += 1
        
        validity = valid_count / len(molecules)
        self.logger.debug(f"Validity: {valid_count}/{len(molecules)} = {validity:.3f}")
        return validity
    
    def compute_uniqueness(self, molecules: List[str]) -> float:
        """
        Compute the percentage of unique molecules in the generated set.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Uniqueness percentage (0.0 to 1.0)
        """
        if not molecules:
            return 0.0
        
        # Canonicalize SMILES to ensure proper uniqueness computation
        canonical_smiles = []
        for smiles in molecules:
            canonical = self._canonicalize_smiles(smiles)
            if canonical:
                canonical_smiles.append(canonical)
        
        if not canonical_smiles:
            return 0.0
        
        unique_molecules = set(canonical_smiles)
        uniqueness = len(unique_molecules) / len(canonical_smiles)
        
        self.logger.debug(f"Uniqueness: {len(unique_molecules)}/{len(canonical_smiles)} = {uniqueness:.3f}")
        return uniqueness
    
    def compute_novelty(self, molecules: List[str]) -> float:
        """
        Compute the percentage of molecules not present in the reference set.
        
        Args:
            molecules: List of generated SMILES strings
            
        Returns:
            Novelty percentage (0.0 to 1.0)
        """
        if not molecules or not self.reference_molecules:
            return 0.0
        
        # Canonicalize generated molecules
        canonical_generated = set()
        for smiles in molecules:
            canonical = self._canonicalize_smiles(smiles)
            if canonical:
                canonical_generated.add(canonical)
        
        if not canonical_generated:
            return 0.0
        
        # Canonicalize reference molecules if not already done
        canonical_reference = set()
        for smiles in self.reference_molecules:
            canonical = self._canonicalize_smiles(smiles)
            if canonical:
                canonical_reference.add(canonical)
        
        # Compute novelty
        novel_molecules = canonical_generated - canonical_reference
        novelty = len(novel_molecules) / len(canonical_generated)
        
        self.logger.debug(f"Novelty: {len(novel_molecules)}/{len(canonical_generated)} = {novelty:.3f}")
        return novelty
    
    def get_valid_molecules(self, molecules: List[str]) -> List[str]:
        """
        Filter and return only chemically valid molecules.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            List of valid SMILES strings
        """
        valid_molecules = []
        for smiles in molecules:
            if self._is_valid_molecule(smiles):
                valid_molecules.append(smiles)
        
        return valid_molecules
    
    def get_unique_molecules(self, molecules: List[str]) -> List[str]:
        """
        Filter and return only unique molecules.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            List of unique SMILES strings (canonicalized)
        """
        seen = set()
        unique_molecules = []
        
        for smiles in molecules:
            canonical = self._canonicalize_smiles(smiles)
            if canonical and canonical not in seen:
                seen.add(canonical)
                unique_molecules.append(canonical)
        
        return unique_molecules
    
    def _is_valid_molecule(self, smiles: str) -> bool:
        """
        Check if a SMILES string represents a valid molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Additional validation checks
            # Check for reasonable atom count
            if mol.GetNumAtoms() == 0:
                return False
            
            # Check for sanitization
            Chem.SanitizeMol(mol)
            return True
            
        except Exception as e:
            self.logger.debug(f"Invalid molecule {smiles}: {e}")
            return False
    
    def _canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to canonical form.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            return Chem.MolToSmiles(mol, canonical=True)
            
        except Exception as e:
            self.logger.debug(f"Cannot canonicalize {smiles}: {e}")
            return None
    
    def compute_property_distributions(self, molecules: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute property distributions for the given molecules.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Dictionary containing property distributions
        """
        if not molecules:
            return {}
        
        # Get valid molecules only
        valid_molecules = self.get_valid_molecules(molecules)
        if not valid_molecules:
            return {}
        
        properties = {
            'molecular_weight': [],
            'logp': [],
            'num_atoms': [],
            'num_bonds': [],
            'num_rings': [],
            'tpsa': []  # Topological Polar Surface Area
        }
        
        for smiles in valid_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Calculate molecular properties
                properties['molecular_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                properties['num_atoms'].append(mol.GetNumAtoms())
                properties['num_bonds'].append(mol.GetNumBonds())
                properties['num_rings'].append(Descriptors.RingCount(mol))
                properties['tpsa'].append(Descriptors.TPSA(mol))
                
            except Exception as e:
                self.logger.debug(f"Error computing properties for {smiles}: {e}")
                continue
        
        # Convert to numpy arrays
        property_arrays = {}
        for prop_name, values in properties.items():
            if values:
                property_arrays[prop_name] = np.array(values)
            else:
                property_arrays[prop_name] = np.array([])
        
        return property_arrays
    
    def compare_property_distributions(self, 
                                     generated_molecules: List[str], 
                                     reference_molecules: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare property distributions between generated and reference molecules.
        
        Args:
            generated_molecules: List of generated SMILES strings
            reference_molecules: List of reference SMILES strings
            
        Returns:
            Dictionary containing statistical comparison results
        """
        if not generated_molecules or not reference_molecules:
            return {}
        
        gen_props = self.compute_property_distributions(generated_molecules)
        ref_props = self.compute_property_distributions(reference_molecules)
        
        comparison_results = {}
        
        for prop_name in gen_props.keys():
            if prop_name not in ref_props or len(gen_props[prop_name]) == 0 or len(ref_props[prop_name]) == 0:
                continue
            
            gen_values = gen_props[prop_name]
            ref_values = ref_props[prop_name]
            
            # Compute statistical measures
            comparison_results[prop_name] = {
                'generated_mean': float(np.mean(gen_values)),
                'generated_std': float(np.std(gen_values)),
                'reference_mean': float(np.mean(ref_values)),
                'reference_std': float(np.std(ref_values)),
                'mean_difference': float(np.mean(gen_values) - np.mean(ref_values)),
                'ks_statistic': self._compute_ks_statistic(gen_values, ref_values),
                'wasserstein_distance': self._compute_wasserstein_distance(gen_values, ref_values)
            }
        
        return comparison_results
    
    def _compute_ks_statistic(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        """
        Compute Kolmogorov-Smirnov statistic between two samples.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            KS statistic (0 = identical distributions, 1 = completely different)
        """
        try:
            from scipy import stats
            ks_stat, _ = stats.ks_2samp(sample1, sample2)
            return float(ks_stat)
        except ImportError:
            self.logger.warning("SciPy not available, using simplified KS approximation")
            # Simple approximation: compare CDFs at a few points
            combined = np.concatenate([sample1, sample2])
            min_val, max_val = np.min(combined), np.max(combined)
            test_points = np.linspace(min_val, max_val, 100)
            
            cdf1 = np.searchsorted(np.sort(sample1), test_points, side='right') / len(sample1)
            cdf2 = np.searchsorted(np.sort(sample2), test_points, side='right') / len(sample2)
            
            return float(np.max(np.abs(cdf1 - cdf2)))
    
    def _compute_wasserstein_distance(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        """
        Compute Wasserstein distance between two samples.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Wasserstein distance
        """
        try:
            from scipy import stats
            return float(stats.wasserstein_distance(sample1, sample2))
        except ImportError:
            self.logger.warning("SciPy not available, using simplified Wasserstein approximation")
            # Simple approximation: difference of means
            return float(abs(np.mean(sample1) - np.mean(sample2)))
    
    def compute_diversity_metrics(self, molecules: List[str]) -> Dict[str, float]:
        """
        Compute additional diversity metrics for the molecule set.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Dictionary with diversity metrics
        """
        if not molecules:
            return {'diversity_score': 0.0}
        
        # Get valid molecules
        valid_molecules = self.get_valid_molecules(molecules)
        if len(valid_molecules) < 2:
            return {'diversity_score': 0.0}
        
        # For now, return a simple diversity metric based on uniqueness
        # This can be extended with more sophisticated diversity measures
        uniqueness = self.compute_uniqueness(valid_molecules)
        
        return {
            'diversity_score': uniqueness,
            'valid_molecules_count': len(valid_molecules)
        }
    
    def generate_evaluation_report(self, 
                                 generated_molecules: List[str],
                                 reference_molecules: Optional[List[str]] = None) -> Dict:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            generated_molecules: List of generated SMILES strings
            reference_molecules: Optional reference molecules for comparison
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'basic_metrics': self.evaluate(generated_molecules),
            'diversity_metrics': self.compute_diversity_metrics(generated_molecules),
            'property_distributions': self.compute_property_distributions(generated_molecules)
        }
        
        if reference_molecules:
            report['property_comparison'] = self.compare_property_distributions(
                generated_molecules, reference_molecules
            )
        
        # Add summary statistics
        valid_molecules = self.get_valid_molecules(generated_molecules)
        unique_molecules = self.get_unique_molecules(valid_molecules)
        
        report['summary'] = {
            'total_generated': len(generated_molecules),
            'valid_count': len(valid_molecules),
            'unique_count': len(unique_molecules),
            'validity_rate': len(valid_molecules) / len(generated_molecules) if generated_molecules else 0.0,
            'uniqueness_rate': len(unique_molecules) / len(valid_molecules) if valid_molecules else 0.0
        }
        
        return report
    
    def compute_qed_scores(self, molecules: List[str]) -> List[float]:
        """
        Compute QED (drug-likeness) scores for molecules.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            List of QED scores (0.0 to 1.0, higher is more drug-like)
        """
        qed_scores = []
        
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    qed_scores.append(0.0)
                    continue
                
                qed_score = QED.qed(mol)
                qed_scores.append(qed_score)
                
            except Exception as e:
                self.logger.debug(f"Error computing QED for {smiles}: {e}")
                qed_scores.append(0.0)
        
        return qed_scores
    
    def compute_lipinski_compliance(self, molecules: List[str]) -> Dict[str, List[bool]]:
        """
        Compute Lipinski Rule of Five compliance for molecules.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Dictionary with compliance for each rule and overall pass/fail
        """
        results = {
            'molecular_weight_ok': [],  # MW <= 500 Da
            'logp_ok': [],             # logP <= 5
            'hbd_ok': [],              # H-bond donors <= 5
            'hba_ok': [],              # H-bond acceptors <= 10
            'lipinski_pass': []        # All rules satisfied
        }
        
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Failed molecule - all rules fail
                    for key in results.keys():
                        results[key].append(False)
                    continue
                
                # Calculate molecular properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # Check each rule
                mw_ok = mw <= 500.0
                logp_ok = logp <= 5.0
                hbd_ok = hbd <= 5
                hba_ok = hba <= 10
                
                results['molecular_weight_ok'].append(mw_ok)
                results['logp_ok'].append(logp_ok)
                results['hbd_ok'].append(hbd_ok)
                results['hba_ok'].append(hba_ok)
                results['lipinski_pass'].append(mw_ok and logp_ok and hbd_ok and hba_ok)
                
            except Exception as e:
                self.logger.debug(f"Error computing Lipinski compliance for {smiles}: {e}")
                # Failed calculation - all rules fail
                for key in results.keys():
                    results[key].append(False)
        
        return results
    
    def compute_drug_likeness_metrics(self, molecules: List[str]) -> Dict[str, float]:
        """
        Compute comprehensive drug-likeness metrics.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Dictionary with drug-likeness metrics
        """
        if not molecules:
            return {
                'mean_qed': 0.0,
                'median_qed': 0.0,
                'lipinski_pass_rate': 0.0,
                'mw_pass_rate': 0.0,
                'logp_pass_rate': 0.0,
                'hbd_pass_rate': 0.0,
                'hba_pass_rate': 0.0
            }
        
        # Get valid molecules only
        valid_molecules = self.get_valid_molecules(molecules)
        if not valid_molecules:
            return {
                'mean_qed': 0.0,
                'median_qed': 0.0,
                'lipinski_pass_rate': 0.0,
                'mw_pass_rate': 0.0,
                'logp_pass_rate': 0.0,
                'hbd_pass_rate': 0.0,
                'hba_pass_rate': 0.0
            }
        
        # Compute QED scores
        qed_scores = self.compute_qed_scores(valid_molecules)
        valid_qed_scores = [score for score in qed_scores if score > 0.0]
        
        # Compute Lipinski compliance
        lipinski_results = self.compute_lipinski_compliance(valid_molecules)
        
        # Calculate metrics
        metrics = {
            'mean_qed': np.mean(valid_qed_scores) if valid_qed_scores else 0.0,
            'median_qed': np.median(valid_qed_scores) if valid_qed_scores else 0.0,
            'lipinski_pass_rate': np.mean(lipinski_results['lipinski_pass']),
            'mw_pass_rate': np.mean(lipinski_results['molecular_weight_ok']),
            'logp_pass_rate': np.mean(lipinski_results['logp_ok']),
            'hbd_pass_rate': np.mean(lipinski_results['hbd_ok']),
            'hba_pass_rate': np.mean(lipinski_results['hba_ok'])
        }
        
        return metrics
    
    def generate_comprehensive_report(self, 
                                    generated_molecules: List[str],
                                    reference_molecules: Optional[List[str]] = None) -> Dict:
        """
        Generate a comprehensive evaluation report including drug-likeness metrics.
        
        Args:
            generated_molecules: List of generated SMILES strings
            reference_molecules: Optional reference molecules for comparison
            
        Returns:
            Comprehensive evaluation report with drug-likeness assessment
        """
        # Get basic report
        report = self.generate_evaluation_report(generated_molecules, reference_molecules)
        
        # Add drug-likeness metrics
        report['drug_likeness'] = self.compute_drug_likeness_metrics(generated_molecules)
        
        # Add detailed QED and Lipinski analysis
        valid_molecules = self.get_valid_molecules(generated_molecules)
        if valid_molecules:
            report['qed_scores'] = self.compute_qed_scores(valid_molecules)
            report['lipinski_compliance'] = self.compute_lipinski_compliance(valid_molecules)
        
        # Add drug-likeness comparison with reference if available
        if reference_molecules:
            ref_drug_likeness = self.compute_drug_likeness_metrics(reference_molecules)
            report['drug_likeness_comparison'] = {
                'generated': report['drug_likeness'],
                'reference': ref_drug_likeness,
                'qed_difference': report['drug_likeness']['mean_qed'] - ref_drug_likeness['mean_qed'],
                'lipinski_difference': report['drug_likeness']['lipinski_pass_rate'] - ref_drug_likeness['lipinski_pass_rate']
            }
        
        return report
    
    def filter_drug_like_molecules(self, 
                                 molecules: List[str], 
                                 qed_threshold: float = 0.5,
                                 require_lipinski: bool = True) -> List[str]:
        """
        Filter molecules based on drug-likeness criteria.
        
        Args:
            molecules: List of SMILES strings
            qed_threshold: Minimum QED score (0.0 to 1.0)
            require_lipinski: Whether to require Lipinski compliance
            
        Returns:
            List of drug-like molecules
        """
        if not molecules:
            return []
        
        valid_molecules = self.get_valid_molecules(molecules)
        if not valid_molecules:
            return []
        
        drug_like_molecules = []
        qed_scores = self.compute_qed_scores(valid_molecules)
        
        if require_lipinski:
            lipinski_results = self.compute_lipinski_compliance(valid_molecules)
        
        for i, smiles in enumerate(valid_molecules):
            # Check QED threshold
            if qed_scores[i] < qed_threshold:
                continue
            
            # Check Lipinski compliance if required
            if require_lipinski and not lipinski_results['lipinski_pass'][i]:
                continue
            
            drug_like_molecules.append(smiles)
        
        return drug_like_molecules