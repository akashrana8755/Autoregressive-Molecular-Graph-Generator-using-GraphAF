"""
Molecular property calculation utilities.

This module provides functions to calculate various molecular properties
including logP, QED, molecular weight, and other drug-likeness metrics
using RDKit.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED, Lipinski
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError(
        "RDKit is required for property calculation. "
        "Install with: pip install rdkit-pypi"
    )

logger = logging.getLogger(__name__)


class PropertyCalculator:
    """
    Calculator for molecular properties and drug-likeness metrics.
    
    This class provides methods to compute various molecular properties
    that are commonly used in drug discovery and molecular generation.
    """
    
    def __init__(self):
        """Initialize the property calculator."""
        self.property_functions = {
            'molecular_weight': self.calculate_molecular_weight,
            'logp': self.calculate_logp,
            'qed': self.calculate_qed,
            'num_hbd': self.calculate_hbd,
            'num_hba': self.calculate_hba,
            'tpsa': self.calculate_tpsa,
            'num_rotatable_bonds': self.calculate_rotatable_bonds,
            'num_rings': self.calculate_num_rings,
            'num_aromatic_rings': self.calculate_num_aromatic_rings,
            'fraction_csp3': self.calculate_fraction_csp3,
            'num_heteroatoms': self.calculate_num_heteroatoms,
            'num_heavy_atoms': self.calculate_num_heavy_atoms,
            'formal_charge': self.calculate_formal_charge,
            'lipinski_violations': self.calculate_lipinski_violations,
            'sa_score': self.calculate_sa_score,
            'bertz_ct': self.calculate_bertz_ct
        }
        
    def calculate_properties(self, 
                           smiles: Union[str, List[str]], 
                           properties: Optional[List[str]] = None) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate molecular properties for SMILES string(s).
        
        Args:
            smiles: SMILES string or list of SMILES strings
            properties: List of properties to calculate (default: all)
            
        Returns:
            Dictionary of properties or list of dictionaries
        """
        if isinstance(smiles, str):
            return self._calculate_single(smiles, properties)
        else:
            return [self._calculate_single(smi, properties) for smi in smiles]
            
    def _calculate_single(self, smiles: str, properties: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate properties for a single SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
                
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            
            # Calculate requested properties
            if properties is None:
                properties = list(self.property_functions.keys())
                
            results = {}
            for prop in properties:
                if prop in self.property_functions:
                    try:
                        value = self.property_functions[prop](mol)
                        results[prop] = float(value) if value is not None else np.nan
                    except Exception as e:
                        logger.warning(f"Error calculating {prop} for {smiles}: {e}")
                        results[prop] = np.nan
                else:
                    logger.warning(f"Unknown property: {prop}")
                    
            return results
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return {}
            
    def calculate_molecular_weight(self, mol: Chem.Mol) -> float:
        """Calculate molecular weight."""
        return Descriptors.MolWt(mol)
        
    def calculate_logp(self, mol: Chem.Mol) -> float:
        """Calculate octanol-water partition coefficient (logP)."""
        return Descriptors.MolLogP(mol)
        
    def calculate_qed(self, mol: Chem.Mol) -> float:
        """Calculate Quantitative Estimate of Drug-likeness (QED)."""
        return QED.qed(mol)
        
    def calculate_hbd(self, mol: Chem.Mol) -> int:
        """Calculate number of hydrogen bond donors."""
        return Descriptors.NumHDonors(mol)
        
    def calculate_hba(self, mol: Chem.Mol) -> int:
        """Calculate number of hydrogen bond acceptors."""
        return Descriptors.NumHAcceptors(mol)
        
    def calculate_tpsa(self, mol: Chem.Mol) -> float:
        """Calculate topological polar surface area."""
        return CalcTPSA(mol)
        
    def calculate_rotatable_bonds(self, mol: Chem.Mol) -> int:
        """Calculate number of rotatable bonds."""
        return CalcNumRotatableBonds(mol)
        
    def calculate_num_rings(self, mol: Chem.Mol) -> int:
        """Calculate number of rings."""
        return Descriptors.RingCount(mol)
        
    def calculate_num_aromatic_rings(self, mol: Chem.Mol) -> int:
        """Calculate number of aromatic rings."""
        return Descriptors.NumAromaticRings(mol)
        
    def calculate_fraction_csp3(self, mol: Chem.Mol) -> float:
        """Calculate fraction of sp3 carbons."""
        return Descriptors.FractionCsp3(mol)
        
    def calculate_num_heteroatoms(self, mol: Chem.Mol) -> int:
        """Calculate number of heteroatoms."""
        return Descriptors.NumHeteroatoms(mol)
        
    def calculate_num_heavy_atoms(self, mol: Chem.Mol) -> int:
        """Calculate number of heavy atoms."""
        return mol.GetNumHeavyAtoms()
        
    def calculate_formal_charge(self, mol: Chem.Mol) -> int:
        """Calculate formal charge."""
        return Chem.rdmolops.GetFormalCharge(mol)
        
    def calculate_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Calculate number of Lipinski rule violations."""
        violations = 0
        
        # Molecular weight > 500
        if self.calculate_molecular_weight(mol) > 500:
            violations += 1
            
        # LogP > 5
        if self.calculate_logp(mol) > 5:
            violations += 1
            
        # HBD > 5
        if self.calculate_hbd(mol) > 5:
            violations += 1
            
        # HBA > 10
        if self.calculate_hba(mol) > 10:
            violations += 1
            
        return violations
        
    def calculate_sa_score(self, mol: Chem.Mol) -> float:
        """
        Calculate synthetic accessibility score.
        
        Note: This is a simplified version. For full SA score,
        you would need the SAScore module from RDKit contrib.
        """
        try:
            # This is a placeholder - implement actual SA score if needed
            # For now, return a dummy value based on complexity
            num_rings = self.calculate_num_rings(mol)
            num_rotatable = self.calculate_rotatable_bonds(mol)
            complexity = num_rings + num_rotatable * 0.5
            return min(10.0, max(1.0, complexity))
        except:
            return 5.0  # Default middle value
            
    def calculate_bertz_ct(self, mol: Chem.Mol) -> float:
        """Calculate Bertz complexity index."""
        return Descriptors.BertzCT(mol)
        
    def check_lipinski_rule(self, smiles: str) -> Dict[str, Any]:
        """
        Check Lipinski's Rule of Five compliance.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with rule compliance details
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'valid': False, 'error': 'Invalid SMILES'}
                
            mw = self.calculate_molecular_weight(mol)
            logp = self.calculate_logp(mol)
            hbd = self.calculate_hbd(mol)
            hba = self.calculate_hba(mol)
            
            violations = []
            if mw > 500:
                violations.append('molecular_weight')
            if logp > 5:
                violations.append('logp')
            if hbd > 5:
                violations.append('hbd')
            if hba > 10:
                violations.append('hba')
                
            return {
                'valid': True,
                'molecular_weight': mw,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
                'violations': violations,
                'num_violations': len(violations),
                'passes_lipinski': len(violations) == 0
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def check_drug_likeness(self, smiles: str) -> Dict[str, Any]:
        """
        Comprehensive drug-likeness assessment.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with drug-likeness metrics
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'valid': False, 'error': 'Invalid SMILES'}
                
            # Calculate properties
            properties = self.calculate_properties(smiles)
            
            # Lipinski compliance
            lipinski = self.check_lipinski_rule(smiles)
            
            # Additional drug-likeness criteria
            criteria = {
                'lipinski_compliant': lipinski['passes_lipinski'],
                'qed_score': properties.get('qed', 0),
                'tpsa_ok': properties.get('tpsa', 0) <= 140,  # TPSA <= 140 Ų
                'rotatable_bonds_ok': properties.get('num_rotatable_bonds', 0) <= 10,
                'aromatic_rings_ok': properties.get('num_aromatic_rings', 0) <= 5,
                'formal_charge_ok': abs(properties.get('formal_charge', 0)) <= 2
            }
            
            # Overall drug-likeness score
            score = sum(criteria.values()) / len(criteria)
            
            return {
                'valid': True,
                'properties': properties,
                'lipinski': lipinski,
                'criteria': criteria,
                'drug_likeness_score': score,
                'is_drug_like': score >= 0.7  # Threshold for drug-likeness
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def filter_drug_like_molecules(self, 
                                 smiles_list: List[str],
                                 min_qed: float = 0.5,
                                 max_violations: int = 1) -> List[str]:
        """
        Filter molecules for drug-likeness.
        
        Args:
            smiles_list: List of SMILES strings
            min_qed: Minimum QED score
            max_violations: Maximum Lipinski violations allowed
            
        Returns:
            List of drug-like SMILES
        """
        drug_like = []
        
        for smiles in smiles_list:
            try:
                assessment = self.check_drug_likeness(smiles)
                
                if (assessment['valid'] and 
                    assessment['properties']['qed'] >= min_qed and
                    assessment['lipinski']['num_violations'] <= max_violations):
                    drug_like.append(smiles)
                    
            except Exception as e:
                logger.warning(f"Error filtering {smiles}: {e}")
                continue
                
        return drug_like
        
    def calculate_diversity_metrics(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Calculate diversity metrics for a set of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary of diversity metrics
        """
        if not smiles_list:
            return {}
            
        # Calculate properties for all molecules
        all_properties = []
        for smiles in smiles_list:
            props = self.calculate_properties(smiles, ['molecular_weight', 'logp', 'tpsa', 'qed'])
            if props:
                all_properties.append(props)
                
        if not all_properties:
            return {}
            
        # Convert to arrays
        prop_arrays = {}
        for prop in ['molecular_weight', 'logp', 'tpsa', 'qed']:
            values = [p.get(prop, np.nan) for p in all_properties]
            prop_arrays[prop] = np.array(values)
            
        # Calculate diversity metrics
        metrics = {}
        
        for prop, values in prop_arrays.items():
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                metrics[f'{prop}_mean'] = float(np.mean(valid_values))
                metrics[f'{prop}_std'] = float(np.std(valid_values))
                metrics[f'{prop}_range'] = float(np.max(valid_values) - np.min(valid_values))
                
        return metrics


# Convenience functions
def calculate_molecular_properties(smiles: Union[str, List[str]], 
                                 properties: Optional[List[str]] = None) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Convenience function to calculate molecular properties.
    
    Args:
        smiles: SMILES string or list of SMILES strings
        properties: List of properties to calculate
        
    Returns:
        Dictionary of properties or list of dictionaries
    """
    calculator = PropertyCalculator()
    return calculator.calculate_properties(smiles, properties)


def check_lipinski_compliance(smiles: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function to check Lipinski rule compliance.
    
    Args:
        smiles: SMILES string or list of SMILES strings
        
    Returns:
        Lipinski compliance results
    """
    calculator = PropertyCalculator()
    
    if isinstance(smiles, str):
        return calculator.check_lipinski_rule(smiles)
    else:
        return [calculator.check_lipinski_rule(smi) for smi in smiles]


def filter_drug_like(smiles_list: List[str], 
                    min_qed: float = 0.5,
                    max_violations: int = 1) -> List[str]:
    """
    Convenience function to filter drug-like molecules.
    
    Args:
        smiles_list: List of SMILES strings
        min_qed: Minimum QED score
        max_violations: Maximum Lipinski violations
        
    Returns:
        List of drug-like SMILES
    """
    calculator = PropertyCalculator()
    return calculator.filter_drug_like_molecules(smiles_list, min_qed, max_violations)


if __name__ == "__main__":
    # Example usage
    calculator = PropertyCalculator()
    
    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"  # Celecoxib
    ]
    
    for smiles in test_smiles:
        print(f"\nSMILES: {smiles}")
        
        # Calculate properties
        props = calculator.calculate_properties(smiles)
        print("Properties:", props)
        
        # Check drug-likeness
        drug_like = calculator.check_drug_likeness(smiles)
        print("Drug-likeness:", drug_like['drug_likeness_score'])
        print("Lipinski violations:", drug_like['lipinski']['num_violations'])