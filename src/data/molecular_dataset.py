"""
PyTorch Geometric dataset classes for molecular data.

This module provides dataset classes for loading and processing molecular
datasets like ZINC15 and QM9 for use with PyTorch Geometric.
"""

import os
import logging
import pickle
from typing import List, Optional, Dict, Any, Callable, Union
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.data.dataset import Dataset as PyGDataset

from .smiles_processor import SMILESProcessor
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class MolecularDataset(Dataset):
    """
    Dataset class for molecular graphs extending torch.utils.data.Dataset.
    
    This class handles loading and processing of molecular data from SMILES
    strings and converts them to PyTorch Geometric Data objects.
    """
    
    def __init__(self,
                 smiles_list: List[str],
                 properties: Optional[Dict[str, List[float]]] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 smiles_processor: Optional[SMILESProcessor] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize the molecular dataset.
        
        Args:
            smiles_list: List of SMILES strings
            properties: Dictionary of property names to lists of values
            transform: Transform to apply to each data object
            pre_transform: Transform to apply during preprocessing
            smiles_processor: SMILES processor instance
            feature_extractor: Feature extractor instance
            cache_dir: Directory to cache processed data
            use_cache: Whether to use caching
        """
        super().__init__()
        
        self.smiles_list = smiles_list
        self.properties = properties or {}
        self.transform = transform
        self.pre_transform = pre_transform
        self.use_cache = use_cache
        
        # Initialize processors
        self.smiles_processor = smiles_processor or SMILESProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Setup caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "processed_data.pkl"
        else:
            self.cache_dir = None
            self.cache_file = None
            
        # Process data
        self.data_list = self._process_data()
        
    def _process_data(self) -> List[Optional[Data]]:
        """Process SMILES strings to molecular graphs."""
        # Try to load from cache
        if self.use_cache and self.cache_file and self.cache_file.exists():
            try:
                logger.info(f"Loading cached data from {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                
        logger.info(f"Processing {len(self.smiles_list)} molecules...")
        data_list = []
        
        for i, smiles in enumerate(self.smiles_list):
            try:
                # Convert SMILES to basic graph
                graph = self.smiles_processor.smiles_to_graph(smiles)
                
                if graph is None:
                    logger.warning(f"Failed to process SMILES at index {i}: {smiles}")
                    data_list.append(None)
                    continue
                    
                # Enhance with detailed features
                mol = self._smiles_to_mol(smiles)
                if mol is not None:
                    enhanced_features = self.feature_extractor.get_graph_features(mol)
                    if enhanced_features:
                        graph.x = enhanced_features['x']
                        graph.edge_attr = enhanced_features['edge_attr']
                        graph.edge_index = enhanced_features['edge_index']
                        
                # Add properties if available
                for prop_name, prop_values in self.properties.items():
                    if i < len(prop_values):
                        setattr(graph, prop_name, torch.tensor([prop_values[i]], dtype=torch.float))
                        
                # Add molecule index
                graph.mol_idx = i
                
                # Apply pre-transform if specified
                if self.pre_transform:
                    graph = self.pre_transform(graph)
                    
                data_list.append(graph)
                
            except Exception as e:
                logger.error(f"Error processing molecule {i}: {smiles}, Error: {e}")
                data_list.append(None)
                
        # Cache processed data
        if self.use_cache and self.cache_file:
            try:
                logger.info(f"Caching processed data to {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(data_list, f)
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
                
        valid_count = sum(1 for data in data_list if data is not None)
        logger.info(f"Successfully processed {valid_count}/{len(self.smiles_list)} molecules")
        
        return data_list
        
    def _smiles_to_mol(self, smiles: str):
        """Convert SMILES to RDKit molecule."""
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles)
        except:
            return None
            
    def __len__(self) -> int:
        """Return the number of molecules in the dataset."""
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Optional[Data]:
        """Get a molecular graph by index."""
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list)}")
            
        data = self.data_list[idx]
        
        if data is None:
            return None
            
        # Apply transform if specified
        if self.transform:
            data = self.transform(data)
            
        return data
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        valid_data = [data for data in self.data_list if data is not None]
        
        if not valid_data:
            return {}
            
        stats = {
            'total_molecules': len(self.data_list),
            'valid_molecules': len(valid_data),
            'invalid_molecules': len(self.data_list) - len(valid_data),
            'validity_rate': len(valid_data) / len(self.data_list),
        }
        
        # Node and edge statistics
        num_nodes = [data.num_nodes for data in valid_data]
        num_edges = [data.edge_index.size(1) for data in valid_data]
        
        stats.update({
            'avg_num_nodes': np.mean(num_nodes),
            'std_num_nodes': np.std(num_nodes),
            'min_num_nodes': np.min(num_nodes),
            'max_num_nodes': np.max(num_nodes),
            'avg_num_edges': np.mean(num_edges),
            'std_num_edges': np.std(num_edges),
            'min_num_edges': np.min(num_edges),
            'max_num_edges': np.max(num_edges),
        })
        
        # Property statistics
        for prop_name in self.properties:
            if hasattr(valid_data[0], prop_name):
                prop_values = [getattr(data, prop_name).item() for data in valid_data 
                              if hasattr(data, prop_name)]
                if prop_values:
                    stats[f'{prop_name}_mean'] = np.mean(prop_values)
                    stats[f'{prop_name}_std'] = np.std(prop_values)
                    stats[f'{prop_name}_min'] = np.min(prop_values)
                    stats[f'{prop_name}_max'] = np.max(prop_values)
                    
        return stats


class ZINC15Dataset(MolecularDataset):
    """
    Dataset class for ZINC15 molecular data.
    
    ZINC15 is a database of commercially available compounds for virtual screening.
    """
    
    def __init__(self,
                 data_path: str,
                 subset: str = "250k",
                 split: str = "train",
                 **kwargs):
        """
        Initialize ZINC15 dataset.
        
        Args:
            data_path: Path to ZINC15 data directory
            subset: Dataset subset ("250k", "1m", "full")
            split: Data split ("train", "val", "test")
            **kwargs: Additional arguments for MolecularDataset
        """
        self.data_path = Path(data_path)
        self.subset = subset
        self.split = split
        
        # Load SMILES data
        smiles_list = self._load_zinc15_smiles()
        
        super().__init__(smiles_list=smiles_list, **kwargs)
        
    def _load_zinc15_smiles(self) -> List[str]:
        """Load SMILES strings from ZINC15 dataset."""
        # Expected file format: {subset}_{split}.txt or {subset}_{split}.csv
        possible_files = [
            self.data_path / f"{self.subset}_{self.split}.txt",
            self.data_path / f"{self.subset}_{self.split}.csv",
            self.data_path / f"zinc15_{self.subset}_{self.split}.txt",
            self.data_path / f"zinc15_{self.subset}_{self.split}.csv"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                logger.info(f"Loading ZINC15 data from {file_path}")
                
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    # Assume SMILES column is named 'smiles' or 'SMILES'
                    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
                    return df[smiles_col].tolist()
                else:
                    # Assume plain text file with one SMILES per line
                    with open(file_path, 'r') as f:
                        return [line.strip() for line in f if line.strip()]
                        
        raise FileNotFoundError(
            f"Could not find ZINC15 data file for subset={self.subset}, split={self.split} "
            f"in directory {self.data_path}"
        )


class QM9Dataset(MolecularDataset):
    """
    Dataset class for QM9 molecular property data.
    
    QM9 contains quantum mechanical properties for 134k small organic molecules.
    """
    
    # QM9 property names and units
    QM9_PROPERTIES = {
        'mu': 'Dipole moment (D)',
        'alpha': 'Isotropic polarizability (Bohr^3)',
        'homo': 'HOMO energy (Hartree)',
        'lumo': 'LUMO energy (Hartree)',
        'gap': 'HOMO-LUMO gap (Hartree)',
        'r2': 'Electronic spatial extent (Bohr^2)',
        'zpve': 'Zero point vibrational energy (Hartree)',
        'u0': 'Internal energy at 0K (Hartree)',
        'u298': 'Internal energy at 298.15K (Hartree)',
        'h298': 'Enthalpy at 298.15K (Hartree)',
        'g298': 'Free energy at 298.15K (Hartree)',
        'cv': 'Heat capacity at 298.15K (cal/mol/K)',
    }
    
    def __init__(self,
                 data_path: str,
                 properties: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize QM9 dataset.
        
        Args:
            data_path: Path to QM9 data file (CSV format)
            properties: List of properties to include (default: all)
            **kwargs: Additional arguments for MolecularDataset
        """
        self.data_path = Path(data_path)
        self.property_names = properties or list(self.QM9_PROPERTIES.keys())
        
        # Load data
        smiles_list, property_dict = self._load_qm9_data()
        
        super().__init__(smiles_list=smiles_list, properties=property_dict, **kwargs)
        
    def _load_qm9_data(self) -> tuple[List[str], Dict[str, List[float]]]:
        """Load SMILES and properties from QM9 dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"QM9 data file not found: {self.data_path}")
            
        logger.info(f"Loading QM9 data from {self.data_path}")
        
        # Load CSV data
        df = pd.read_csv(self.data_path)
        
        # Extract SMILES
        smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
        if smiles_col not in df.columns:
            raise ValueError("No SMILES column found in QM9 data")
            
        smiles_list = df[smiles_col].tolist()
        
        # Extract properties
        property_dict = {}
        for prop in self.property_names:
            if prop in df.columns:
                property_dict[prop] = df[prop].tolist()
            else:
                logger.warning(f"Property {prop} not found in QM9 data")
                
        logger.info(f"Loaded {len(smiles_list)} molecules with {len(property_dict)} properties")
        
        return smiles_list, property_dict


def create_molecular_dataloader(dataset: MolecularDataset,
                               batch_size: int = 32,
                               shuffle: bool = True,
                               num_workers: int = 0,
                               collate_fn: Optional[Callable] = None,
                               **kwargs) -> DataLoader:
    """
    Create a DataLoader for molecular datasets.
    
    Args:
        dataset: MolecularDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_fn: Custom collate function
        **kwargs: Additional DataLoader arguments
        
    Returns:
        PyTorch DataLoader
    """
    # Filter out None values
    def molecular_collate_fn(batch):
        # Remove None values
        batch = [data for data in batch if data is not None]
        if not batch:
            return None
        
        # Use PyTorch Geometric's default collate function
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
    
    if collate_fn is None:
        collate_fn = molecular_collate_fn
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


def load_dataset(dataset_name: str,
                data_path: str,
                **kwargs) -> MolecularDataset:
    """
    Load a molecular dataset by name.
    
    Args:
        dataset_name: Name of the dataset ("zinc15", "qm9", or "custom")
        data_path: Path to dataset files
        **kwargs: Additional dataset arguments
        
    Returns:
        MolecularDataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "zinc15":
        return ZINC15Dataset(data_path, **kwargs)
    elif dataset_name == "qm9":
        return QM9Dataset(data_path, **kwargs)
    elif dataset_name == "custom":
        # For custom datasets, expect a list of SMILES strings
        if isinstance(data_path, (list, tuple)):
            smiles_list = data_path
        else:
            # Load from file
            with open(data_path, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        return MolecularDataset(smiles_list, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")