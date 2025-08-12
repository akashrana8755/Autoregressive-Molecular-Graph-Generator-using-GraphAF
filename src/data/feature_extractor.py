"""
Molecular feature extraction utilities for atoms and bonds.

This module provides the FeatureExtractor class for extracting comprehensive
features from molecular structures for use in graph neural networks.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem, Descriptors, Crippen
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError(
        "RDKit is required for feature extraction. "
        "Install with: pip install rdkit-pypi"
    )

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts atom and bond features from molecular structures.
    
    This class provides comprehensive feature extraction for atoms and bonds
    in molecular graphs, including chemical properties and structural information.
    """
    
    def __init__(self, 
                 use_chirality: bool = True,
                 use_partial_charge: bool = False,
                 max_atomic_num: int = 100):
        """
        Initialize the feature extractor.
        
        Args:
            use_chirality: Whether to include chirality information
            use_partial_charge: Whether to compute partial charges (slower)
            max_atomic_num: Maximum atomic number to consider
        """
        self.use_chirality = use_chirality
        self.use_partial_charge = use_partial_charge
        self.max_atomic_num = max_atomic_num
        
        # Define feature vocabularies
        self._setup_feature_vocabularies()
        
    def _setup_feature_vocabularies(self):
        """Setup vocabularies for categorical features."""
        # Atomic numbers for common elements in drug-like molecules
        self.atomic_nums = list(range(1, self.max_atomic_num + 1))
        
        # Degree (number of bonds)
        self.degrees = [0, 1, 2, 3, 4, 5, 6]
        
        # Formal charge
        self.formal_charges = [-3, -2, -1, 0, 1, 2, 3]
        
        # Hybridization
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]
        
        # Number of hydrogen atoms
        self.num_hs = [0, 1, 2, 3, 4]
        
        # Chirality
        self.chiral_types = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]
        
        # Bond types
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        
        # Bond stereo
        self.bond_stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]
        
    def get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """
        Extract comprehensive features for an atom.
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            List of atom features
        """
        features = []
        
        # Atomic number (one-hot encoded)
        atomic_num = atom.GetAtomicNum()
        features.extend(self._one_hot_encode(atomic_num, self.atomic_nums))
        
        # Degree (number of bonds)
        degree = atom.GetDegree()
        features.extend(self._one_hot_encode(degree, self.degrees))
        
        # Formal charge
        formal_charge = atom.GetFormalCharge()
        features.extend(self._one_hot_encode(formal_charge, self.formal_charges))
        
        # Hybridization
        hybridization = atom.GetHybridization()
        features.extend(self._one_hot_encode(hybridization, self.hybridizations))
        
        # Aromaticity
        features.append(float(atom.GetIsAromatic()))
        
        # Number of hydrogen atoms
        num_hs = atom.GetTotalNumHs()
        features.extend(self._one_hot_encode(num_hs, self.num_hs))
        
        # Chirality
        if self.use_chirality:
            chiral_type = atom.GetChiralTag()
            features.extend(self._one_hot_encode(chiral_type, self.chiral_types))
        
        # Additional properties
        features.append(float(atom.IsInRing()))
        features.append(float(atom.GetMass()))
        
        # Partial charge (if requested and available)
        if self.use_partial_charge:
            try:
                partial_charge = float(atom.GetProp('_GasteigerCharge'))
                features.append(partial_charge)
            except:
                features.append(0.0)
                
        return features
        
    def get_bond_features(self, bond: Chem.Bond) -> List[float]:
        """
        Extract comprehensive features for a bond.
        
        Args:
            bond: RDKit Bond object
            
        Returns:
            List of bond features
        """
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        features.extend(self._one_hot_encode(bond_type, self.bond_types))
        
        # Bond stereo configuration
        bond_stereo = bond.GetStereo()
        features.extend(self._one_hot_encode(bond_stereo, self.bond_stereos))
        
        # Conjugation
        features.append(float(bond.GetIsConjugated()))
        
        # Ring membership
        features.append(float(bond.IsInRing()))
        
        # Aromaticity
        features.append(float(bond.GetIsAromatic()))
        
        return features
        
    def get_graph_features(self, mol: Chem.Mol) -> Dict[str, torch.Tensor]:
        """
        Extract features for all atoms and bonds in a molecule.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary containing node and edge features as tensors
        """
        try:
            # Extract atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self.get_atom_features(atom)
                atom_features.append(features)
                
            # Extract bond features and edge indices
            bond_features = []
            edge_indices = []
            
            for bond in mol.GetBonds():
                # Get bond features
                features = self.get_bond_features(bond)
                
                # Get atom indices
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                bond_features.extend([features, features])
                
            # Convert to tensors
            node_features = torch.tensor(atom_features, dtype=torch.float)
            
            if bond_features:
                edge_features = torch.tensor(bond_features, dtype=torch.float)
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                # Handle molecules with no bonds (single atoms)
                edge_features = torch.empty((0, len(self.get_bond_features_dim())), dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            return {
                'x': node_features,
                'edge_attr': edge_features,
                'edge_index': edge_index
            }
            
        except Exception as e:
            logger.error(f"Error extracting graph features: {str(e)}")
            return None
            
    def get_atom_features_dim(self) -> int:
        """Get the dimensionality of atom features."""
        # Create a dummy atom to get feature dimensions
        mol = Chem.MolFromSmiles('C')
        if mol is None:
            return 0
        atom = mol.GetAtomByIdx(0)
        features = self.get_atom_features(atom)
        return len(features)
        
    def get_bond_features_dim(self) -> int:
        """Get the dimensionality of bond features."""
        # Create a dummy bond to get feature dimensions
        mol = Chem.MolFromSmiles('CC')
        if mol is None:
            return 0
        bond = mol.GetBondBetweenAtoms(0, 1)
        features = self.get_bond_features(bond)
        return len(features)
        
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get names of all features for interpretability.
        
        Returns:
            Dictionary with atom and bond feature names
        """
        atom_names = []
        bond_names = []
        
        # Atom feature names
        atom_names.extend([f'atomic_num_{i}' for i in self.atomic_nums])
        atom_names.extend([f'degree_{i}' for i in self.degrees])
        atom_names.extend([f'formal_charge_{i}' for i in self.formal_charges])
        atom_names.extend([f'hybridization_{i}' for i in range(len(self.hybridizations))])
        atom_names.append('is_aromatic')
        atom_names.extend([f'num_hs_{i}' for i in self.num_hs])
        
        if self.use_chirality:
            atom_names.extend([f'chiral_type_{i}' for i in range(len(self.chiral_types))])
            
        atom_names.extend(['is_in_ring', 'mass'])
        
        if self.use_partial_charge:
            atom_names.append('partial_charge')
            
        # Bond feature names
        bond_names.extend([f'bond_type_{i}' for i in range(len(self.bond_types))])
        bond_names.extend([f'bond_stereo_{i}' for i in range(len(self.bond_stereos))])
        bond_names.extend(['is_conjugated', 'is_in_ring', 'is_aromatic'])
        
        return {
            'atom_features': atom_names,
            'bond_features': bond_names
        }
        
    def _one_hot_encode(self, value: Any, vocabulary: List[Any]) -> List[float]:
        """
        One-hot encode a value given a vocabulary.
        
        Args:
            value: Value to encode
            vocabulary: List of possible values
            
        Returns:
            One-hot encoded vector
        """
        encoding = [0.0] * len(vocabulary)
        try:
            if value in vocabulary:
                idx = vocabulary.index(value)
                encoding[idx] = 1.0
        except (ValueError, TypeError):
            pass  # Keep all zeros for unknown values
            
        return encoding
        
    def compute_molecular_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute additional molecular descriptors.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary of molecular descriptors
        """
        try:
            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'fraction_csp3': Descriptors.FractionCsp3(mol),
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol)
            }
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error computing molecular descriptors: {str(e)}")
            return {}
            
    def extract_pharmacophore_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Extract pharmacophore-relevant features.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary of pharmacophore features
        """
        try:
            features = {
                'has_aromatic_ring': any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
                'has_basic_nitrogen': False,
                'has_acidic_group': False,
                'has_hydroxyl': False,
                'has_carbonyl': False
            }
            
            # Check for basic nitrogen
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 7:  # Nitrogen
                    # Simple heuristic for basic nitrogen
                    if atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() > 0:
                        features['has_basic_nitrogen'] = True
                        break
                        
            # Check for functional groups (simplified)
            smiles = Chem.MolToSmiles(mol)
            if 'O' in smiles:
                features['has_hydroxyl'] = 'OH' in smiles
                features['has_carbonyl'] = 'C=O' in smiles or 'C(=O)' in smiles
                
            if any(atom.GetAtomicNum() in [8, 16] and atom.GetFormalCharge() < 0 
                   for atom in mol.GetAtoms()):
                features['has_acidic_group'] = True
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pharmacophore features: {str(e)}")
            return {}