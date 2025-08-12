"""
PyTorch Geometric dataset integration for molecular datasets.

This module provides PyTorch Geometric compatible dataset classes with
built-in downloading, processing, and caching capabilities.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip

from .dataset_downloader import ZINC15Downloader, QM9Downloader, DatasetProcessor
from .smiles_processor import SMILESProcessor
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class ZINC15PyGDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for ZINC15 data with automatic downloading.
    
    This dataset automatically downloads and processes ZINC15 data,
    converting SMILES strings to molecular graphs.
    """
    
    def __init__(self,
                 root: str,
                 subset: str = "250k",
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_cache: bool = True):
        """
        Initialize ZINC15 PyTorch Geometric dataset.
        
        Args:
            root: Root directory to store dataset
            subset: Dataset subset ("250k", "1m")
            split: Data split ("train", "val", "test")
            transform: Transform to apply to each data object
            pre_transform: Transform to apply during preprocessing
            pre_filter: Filter to apply during preprocessing
            use_cache: Whether to use caching
        """
        self.subset = subset
        self.split = split
        self.use_cache = use_cache
        
        # Initialize processors
        self.smiles_processor = SMILESProcessor()
        self.feature_extractor = FeatureExtractor()
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return [f"zinc15_{self.subset}_{self.split}.txt"]
        
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"zinc15_{self.subset}_{self.split}_processed.pt"]
        
    def download(self):
        """Download raw ZINC15 data."""
        logger.info(f"Downloading ZINC15 {self.subset} {self.split} data...")
        
        downloader = ZINC15Downloader(data_dir=self.raw_dir)
        downloaded_files = downloader.download_subset(
            subset=self.subset,
            splits=[self.split]
        )
        
        if self.split not in downloaded_files:
            raise RuntimeError(f"Failed to download ZINC15 {self.subset} {self.split}")
            
        # Move file to expected location
        source_path = downloaded_files[self.split]
        target_path = Path(self.raw_dir) / self.raw_file_names[0]
        
        if source_path != target_path:
            source_path.rename(target_path)
            
        logger.info(f"Downloaded ZINC15 data to {target_path}")
        
    def process(self):
        """Process raw data into PyTorch Geometric format."""
        logger.info(f"Processing ZINC15 {self.subset} {self.split} data...")
        
        # Read SMILES strings
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        with open(raw_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Processing {len(smiles_list)} SMILES strings...")
        
        # Convert to molecular graphs
        data_list = []
        failed_count = 0
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Convert SMILES to basic graph
                graph = self.smiles_processor.smiles_to_graph(smiles)
                
                if graph is None:
                    failed_count += 1
                    continue
                    
                # Enhance with detailed features
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        enhanced_features = self.feature_extractor.get_graph_features(mol)
                        if enhanced_features:
                            graph.x = enhanced_features['x']
                            graph.edge_attr = enhanced_features['edge_attr']
                            graph.edge_index = enhanced_features['edge_index']
                except ImportError:
                    logger.warning("RDKit not available, using basic features")
                    
                # Add metadata
                graph.smiles = smiles
                graph.mol_idx = i
                
                # Apply pre-filter
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                    
                # Apply pre-transform
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                    
                data_list.append(graph)
                
            except Exception as e:
                logger.warning(f"Failed to process molecule {i}: {smiles}, Error: {e}")
                failed_count += 1
                
        logger.info(f"Successfully processed {len(data_list)}/{len(smiles_list)} molecules")
        logger.info(f"Failed to process {failed_count} molecules")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        logger.info(f"Saved processed data to {self.processed_paths[0]}")


class QM9PyGDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for QM9 data with automatic downloading.
    
    This dataset automatically downloads and processes QM9 data,
    including molecular properties.
    """
    
    # QM9 property names
    PROPERTY_NAMES = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
        'u0', 'u298', 'h298', 'g298', 'cv'
    ]
    
    def __init__(self,
                 root: str,
                 properties: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_cache: bool = True):
        """
        Initialize QM9 PyTorch Geometric dataset.
        
        Args:
            root: Root directory to store dataset
            properties: List of properties to include (default: all)
            transform: Transform to apply to each data object
            pre_transform: Transform to apply during preprocessing
            pre_filter: Filter to apply during preprocessing
            use_cache: Whether to use caching
        """
        self.properties = properties or self.PROPERTY_NAMES
        self.use_cache = use_cache
        
        # Initialize processors
        self.smiles_processor = SMILESProcessor()
        self.feature_extractor = FeatureExtractor()
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["qm9.csv"]
        
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return ["qm9_processed.pt"]
        
    def download(self):
        """Download raw QM9 data."""
        logger.info("Downloading QM9 data...")
        
        downloader = QM9Downloader(data_dir=self.raw_dir)
        qm9_file = downloader.download_qm9("processed")
        
        # Move file to expected location
        target_path = Path(self.raw_dir) / self.raw_file_names[0]
        
        if qm9_file != target_path:
            qm9_file.rename(target_path)
            
        logger.info(f"Downloaded QM9 data to {target_path}")
        
    def process(self):
        """Process raw data into PyTorch Geometric format."""
        logger.info("Processing QM9 data...")
        
        # Read CSV data
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        df = pd.read_csv(raw_path)
        
        # Extract SMILES and properties
        smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
        if smiles_col not in df.columns:
            raise ValueError("No SMILES column found in QM9 data")
            
        smiles_list = df[smiles_col].tolist()
        
        # Extract properties
        property_dict = {}
        for prop in self.properties:
            if prop in df.columns:
                property_dict[prop] = df[prop].tolist()
            else:
                logger.warning(f"Property {prop} not found in QM9 data")
                
        logger.info(f"Processing {len(smiles_list)} molecules with {len(property_dict)} properties...")
        
        # Convert to molecular graphs
        data_list = []
        failed_count = 0
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Convert SMILES to basic graph
                graph = self.smiles_processor.smiles_to_graph(smiles)
                
                if graph is None:
                    failed_count += 1
                    continue
                    
                # Enhance with detailed features
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        enhanced_features = self.feature_extractor.get_graph_features(mol)
                        if enhanced_features:
                            graph.x = enhanced_features['x']
                            graph.edge_attr = enhanced_features['edge_attr']
                            graph.edge_index = enhanced_features['edge_index']
                except ImportError:
                    logger.warning("RDKit not available, using basic features")
                    
                # Add properties
                for prop_name, prop_values in property_dict.items():
                    if i < len(prop_values) and not pd.isna(prop_values[i]):
                        setattr(graph, prop_name, torch.tensor([prop_values[i]], dtype=torch.float))
                        
                # Add metadata
                graph.smiles = smiles
                graph.mol_idx = i
                
                # Apply pre-filter
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                    
                # Apply pre-transform
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                    
                data_list.append(graph)
                
            except Exception as e:
                logger.warning(f"Failed to process molecule {i}: {smiles}, Error: {e}")
                failed_count += 1
                
        logger.info(f"Successfully processed {len(data_list)}/{len(smiles_list)} molecules")
        logger.info(f"Failed to process {failed_count} molecules")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        logger.info(f"Saved processed data to {self.processed_paths[0]}")


def create_dataset(dataset_name: str,
                  root: str,
                  **kwargs) -> InMemoryDataset:
    """
    Create a PyTorch Geometric dataset by name.
    
    Args:
        dataset_name: Name of the dataset ("zinc15", "qm9")
        root: Root directory for dataset
        **kwargs: Additional dataset arguments
        
    Returns:
        PyTorch Geometric dataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "zinc15":
        return ZINC15PyGDataset(root, **kwargs)
    elif dataset_name == "qm9":
        return QM9PyGDataset(root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "zinc15":
        return {
            "name": "ZINC15",
            "description": "Database of commercially available compounds for virtual screening",
            "subsets": ["250k", "1m"],
            "splits": ["train", "val", "test"],
            "properties": ["mol_weight", "logp"],
            "size": {
                "250k": {"train": 249455, "val": 24946, "test": 5000},
                "1m": {"train": 800000, "val": 100000, "test": 100000}
            }
        }
    elif dataset_name == "qm9":
        return {
            "name": "QM9",
            "description": "Quantum mechanical properties for 134k small organic molecules",
            "subsets": ["full"],
            "splits": ["full"],
            "properties": QM9PyGDataset.PROPERTY_NAMES,
            "size": {"full": 133885}
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")