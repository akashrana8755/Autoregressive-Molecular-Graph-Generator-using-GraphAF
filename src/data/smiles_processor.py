"""
SMILES to molecular graph conversion utilities using RDKit.

This module provides the SMILESProcessor class for bidirectional conversion
between SMILES strings and molecular graph representations compatible with
PyTorch Geometric.
"""

import logging
from typing import Optional, Dict, Any, List
import warnings

import torch
from torch_geometric.data import Data
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, rdchem
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError(
        "RDKit is required for SMILES processing. "
        "Install with: pip install rdkit-pypi"
    )

logger = logging.getLogger(__name__)


class SMILESProcessor:
    """
    Converts SMILES strings to molecular graphs and vice versa.
    
    This class handles the conversion between SMILES string representations
    and PyTorch Geometric Data objects representing molecular graphs.
    """
    
    def __init__(self, 
                 add_self_loops: bool = False,
                 explicit_hydrogens: bool = False,
                 sanitize: bool = True):
        """
        Initialize the SMILES processor.
        
        Args:
            add_self_loops: Whether to add self-loops to atoms in the graph
            explicit_hydrogens: Whether to include explicit hydrogen atoms
            sanitize: Whether to sanitize molecules during processing
        """
        self.add_self_loops = add_self_loops
        self.explicit_hydrogens = explicit_hydrogens
        self.sanitize = sanitize
        
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert a SMILES string to a molecular graph.
        
        Args:
            smiles: SMILES string representation of the molecule
            
        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        try:
            # Parse SMILES string
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None
                
            # Sanitize molecule if requested
            if self.sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    logger.warning(f"Failed to sanitize molecule: {smiles}")
                    return None
                    
            # Add explicit hydrogens if requested
            if self.explicit_hydrogens:
                mol = Chem.AddHs(mol)
                
            # Extract atoms and bonds
            atoms = mol.GetAtoms()
            bonds = mol.GetBonds()
            
            if len(atoms) == 0:
                logger.warning(f"No atoms found in molecule: {smiles}")
                return None
                
            # Create node features (will be implemented by FeatureExtractor)
            # For now, use atomic numbers as placeholder
            node_features = []
            for atom in atoms:
                node_features.append([atom.GetAtomicNum()])
                
            # Create edge indices and edge features
            edge_indices = []
            edge_features = []
            
            for bond in bonds:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                
                # Bond type as placeholder feature (will be enhanced by FeatureExtractor)
                bond_type = bond.GetBondType()
                bond_feature = [self._bond_type_to_int(bond_type)]
                edge_features.extend([bond_feature, bond_feature])
                
            # Add self-loops if requested
            if self.add_self_loops:
                num_atoms = len(atoms)
                for i in range(num_atoms):
                    edge_indices.append([i, i])
                    edge_features.append([0])  # Self-loop feature
                    
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles,
                num_nodes=len(atoms)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error converting SMILES to graph: {smiles}, Error: {str(e)}")
            return None
            
    def graph_to_smiles(self, graph: Data) -> Optional[str]:
        """
        Convert a molecular graph back to SMILES string.
        
        Args:
            graph: PyTorch Geometric Data object representing molecular graph
            
        Returns:
            SMILES string or None if conversion fails
        """
        try:
            # If original SMILES is stored, return it
            if hasattr(graph, 'smiles') and graph.smiles:
                return graph.smiles
                
            # Otherwise, reconstruct from graph structure
            # This is a simplified implementation using RDKit's graph construction
            mol = self._graph_to_rdkit_mol(graph)
            if mol is not None:
                try:
                    if self.sanitize:
                        Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol, canonical=True)
                except:
                    logger.warning("Failed to sanitize reconstructed molecule")
                    return None
            else:
                logger.warning("Failed to reconstruct RDKit molecule from graph")
                return None
                
        except Exception as e:
            logger.error(f"Error converting graph to SMILES: {str(e)}")
            return None
    
    def _graph_to_rdkit_mol(self, graph: Data) -> Optional[Chem.Mol]:
        """
        Convert a molecular graph to RDKit molecule object.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            RDKit molecule object or None if conversion fails
        """
        try:
            if graph.x.size(0) == 0:
                return None
                
            # Create empty molecule
            mol = Chem.RWMol()
            
            # Add atoms based on node features
            # Assuming first feature is atomic number (simplified)
            for i in range(graph.x.size(0)):
                node_features = graph.x[i]
                # Extract atomic number (this is simplified - in practice you'd decode features)
                atomic_num = int(node_features[0].item()) if node_features[0] > 0 else 6  # Default to carbon
                atomic_num = max(1, min(atomic_num, 118))  # Clamp to valid range
                
                atom = Chem.Atom(atomic_num)
                mol.AddAtom(atom)
            
            # Add bonds based on edge information
            if graph.edge_index.size(1) > 0:
                # Process edges (avoid duplicates for undirected graphs)
                processed_edges = set()
                
                for i in range(graph.edge_index.size(1)):
                    src = int(graph.edge_index[0, i].item())
                    dst = int(graph.edge_index[1, i].item())
                    
                    # Skip self-loops and already processed edges
                    if src == dst:
                        continue
                        
                    edge_key = tuple(sorted([src, dst]))
                    if edge_key in processed_edges:
                        continue
                    processed_edges.add(edge_key)
                    
                    # Determine bond type from edge features
                    if graph.edge_attr.size(0) > i:
                        edge_features = graph.edge_attr[i]
                        bond_type = self._int_to_bond_type(int(edge_features[0].item()))
                    else:
                        bond_type = Chem.rdchem.BondType.SINGLE
                    
                    try:
                        mol.AddBond(src, dst, bond_type)
                    except:
                        # If bond addition fails, skip this bond
                        continue
            
            # Convert to regular molecule
            mol = mol.GetMol()
            return mol
            
        except Exception as e:
            logger.error(f"Error converting graph to RDKit molecule: {str(e)}")
            return None
    
    def _int_to_bond_type(self, bond_int: int):
        """Convert integer to RDKit bond type."""
        bond_type_map = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
        return bond_type_map.get(bond_int, Chem.rdchem.BondType.SINGLE)
            
    def validate_molecule(self, smiles: str) -> bool:
        """
        Validate if a SMILES string represents a valid molecule.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            if self.sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    return False
                    
            return True
            
        except Exception:
            return False
            
    def sanitize_smiles(self, smiles: str) -> Optional[str]:
        """
        Sanitize and canonicalize a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonicalized SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            if self.sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    return None
                    
            # Return canonical SMILES
            return Chem.MolToSmiles(mol, canonical=True)
            
        except Exception:
            return None
            
    def batch_process_smiles(self, smiles_list: List[str]) -> List[Optional[Data]]:
        """
        Process a batch of SMILES strings to molecular graphs.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of Data objects (None for failed conversions)
        """
        graphs = []
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            graphs.append(graph)
            
        return graphs
        
    def get_molecule_info(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a molecule from its SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with molecule information or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            info = {
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'canonical_smiles': Chem.MolToSmiles(mol, canonical=True)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting molecule info: {smiles}, Error: {str(e)}")
            return None
            
    def _bond_type_to_int(self, bond_type) -> int:
        """Convert RDKit bond type to integer."""
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
        }
        return bond_type_map.get(bond_type, 0)