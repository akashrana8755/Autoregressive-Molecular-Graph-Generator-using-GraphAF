"""
Dataset downloader utilities for molecular datasets.

This module provides utilities to download and process ZINC15 and QM9 datasets
with proper caching and preprocessing mechanisms.
"""

import os
import logging
import requests
import zipfile
import tarfile
import gzip
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Base class for dataset downloaders."""
    
    def __init__(self, data_dir: str = "data", cache_dir: Optional[str] = None):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Directory to store downloaded datasets
            cache_dir: Directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL with progress bar.
        
        Args:
            url: URL to download from
            filepath: Local path to save file
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {filepath.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            logger.info(f"Successfully downloaded {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
            
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """
        Extract archive file (zip, tar, tar.gz, gz).
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
            elif archive_path.suffix in ['.tar', '.gz'] or '.tar.' in archive_path.name:
                mode = 'r:gz' if archive_path.suffix == '.gz' or '.tar.gz' in archive_path.name else 'r'
                with tarfile.open(archive_path, mode) as tar_ref:
                    tar_ref.extractall(extract_dir)
                    
            elif archive_path.suffix == '.gz' and not '.tar.' in archive_path.name:
                # Single gzipped file
                output_path = extract_dir / archive_path.stem
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                        
            else:
                logger.error(f"Unsupported archive format: {archive_path}")
                return False
                
            logger.info(f"Successfully extracted {archive_path} to {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False


class ZINC15Downloader(DatasetDownloader):
    """Downloader for ZINC15 dataset."""
    
    # ZINC15 download URLs for different subsets
    ZINC15_URLS = {
        "250k": {
            "train": "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/data/zinc_train.txt",
            "val": "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/data/zinc_val.txt",
            "test": "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/data/zinc_test.txt"
        },
        "1m": {
            # Alternative URLs for larger subsets
            "train": "https://figshare.com/ndownloader/files/13612745",  # Example URL
            "val": "https://figshare.com/ndownloader/files/13612748",
            "test": "https://figshare.com/ndownloader/files/13612751"
        }
    }
    
    def __init__(self, **kwargs):
        """Initialize ZINC15 downloader."""
        super().__init__(**kwargs)
        self.zinc_dir = self.data_dir / "zinc15"
        self.zinc_dir.mkdir(parents=True, exist_ok=True)
        
    def download_subset(self, subset: str = "250k", splits: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Download ZINC15 dataset subset.
        
        Args:
            subset: Dataset subset ("250k", "1m")
            splits: List of splits to download (default: ["train", "val", "test"])
            
        Returns:
            Dictionary mapping split names to file paths
        """
        if subset not in self.ZINC15_URLS:
            raise ValueError(f"Unknown ZINC15 subset: {subset}. Available: {list(self.ZINC15_URLS.keys())}")
            
        splits = splits or ["train", "val", "test"]
        downloaded_files = {}
        
        for split in splits:
            if split not in self.ZINC15_URLS[subset]:
                logger.warning(f"Split {split} not available for subset {subset}")
                continue
                
            url = self.ZINC15_URLS[subset][split]
            filename = f"zinc15_{subset}_{split}.txt"
            filepath = self.zinc_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                logger.info(f"File already exists: {filepath}")
                downloaded_files[split] = filepath
                continue
                
            # Download file
            if self.download_file(url, filepath):
                downloaded_files[split] = filepath
            else:
                logger.error(f"Failed to download {split} split for ZINC15 {subset}")
                
        return downloaded_files
        
    def process_zinc15_file(self, filepath: Path, output_path: Optional[Path] = None) -> Path:
        """
        Process ZINC15 file to standardized format.
        
        Args:
            filepath: Path to raw ZINC15 file
            output_path: Path to save processed file
            
        Returns:
            Path to processed file
        """
        if output_path is None:
            output_path = filepath.parent / f"processed_{filepath.name}"
            
        logger.info(f"Processing ZINC15 file: {filepath}")
        
        # Read SMILES strings
        with open(filepath, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
            
        # Create DataFrame with SMILES
        df = pd.DataFrame({'smiles': smiles_list})
        
        # Add basic molecular properties if RDKit is available
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            valid_smiles = []
            mol_weights = []
            logps = []
            
            for smiles in tqdm(smiles_list, desc="Computing properties"):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
                    mol_weights.append(Descriptors.MolWt(mol))
                    logps.append(Descriptors.MolLogP(mol))
                    
            # Create processed DataFrame
            processed_df = pd.DataFrame({
                'smiles': valid_smiles,
                'mol_weight': mol_weights,
                'logp': logps
            })
            
            logger.info(f"Processed {len(processed_df)}/{len(smiles_list)} valid molecules")
            
        except ImportError:
            logger.warning("RDKit not available, skipping property computation")
            processed_df = df
            
        # Save processed data
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path


class QM9Downloader(DatasetDownloader):
    """Downloader for QM9 dataset."""
    
    # QM9 dataset URLs
    QM9_URLS = {
        "raw": "https://figshare.com/ndownloader/files/3195389",  # qm9.tar.bz2
        "processed": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    }
    
    def __init__(self, **kwargs):
        """Initialize QM9 downloader."""
        super().__init__(**kwargs)
        self.qm9_dir = self.data_dir / "qm9"
        self.qm9_dir.mkdir(parents=True, exist_ok=True)
        
    def download_qm9(self, version: str = "processed") -> Path:
        """
        Download QM9 dataset.
        
        Args:
            version: Dataset version ("raw", "processed")
            
        Returns:
            Path to downloaded file
        """
        if version not in self.QM9_URLS:
            raise ValueError(f"Unknown QM9 version: {version}. Available: {list(self.QM9_URLS.keys())}")
            
        url = self.QM9_URLS[version]
        
        if version == "raw":
            filename = "qm9.tar.bz2"
        else:
            filename = "qm9.csv"
            
        filepath = self.qm9_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath
            
        # Download file
        if self.download_file(url, filepath):
            # Extract if archive
            if version == "raw":
                extract_dir = self.qm9_dir / "raw"
                if self.extract_archive(filepath, extract_dir):
                    return extract_dir
                    
            return filepath
        else:
            raise RuntimeError(f"Failed to download QM9 {version}")
            
    def process_qm9_raw(self, raw_dir: Path) -> Path:
        """
        Process raw QM9 data to CSV format.
        
        Args:
            raw_dir: Directory containing raw QM9 files
            
        Returns:
            Path to processed CSV file
        """
        logger.info(f"Processing raw QM9 data from {raw_dir}")
        
        # This is a simplified version - actual QM9 processing is more complex
        # In practice, you would parse the SDF files and extract properties
        
        output_path = self.qm9_dir / "qm9_processed.csv"
        
        try:
            # Look for SDF files or other data files
            sdf_files = list(raw_dir.glob("*.sdf"))
            xyz_files = list(raw_dir.glob("*.xyz"))
            
            if not sdf_files and not xyz_files:
                raise FileNotFoundError("No SDF or XYZ files found in raw QM9 data")
                
            # For now, create a placeholder - actual implementation would parse these files
            logger.warning("Raw QM9 processing not fully implemented - using placeholder")
            
            # Create minimal DataFrame structure
            df = pd.DataFrame({
                'smiles': [],
                'mu': [],
                'alpha': [],
                'homo': [],
                'lumo': [],
                'gap': [],
                'r2': [],
                'zpve': [],
                'u0': [],
                'u298': [],
                'h298': [],
                'g298': [],
                'cv': []
            })
            
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed QM9 data to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to process raw QM9 data: {e}")
            raise


class DatasetProcessor:
    """Processor for molecular datasets with caching and validation."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize dataset processor.
        
        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, dataset_name: str, subset: str, split: str) -> Path:
        """Get cache file path for dataset."""
        return self.cache_dir / f"{dataset_name}_{subset}_{split}_processed.pkl"
        
    def is_cached(self, dataset_name: str, subset: str, split: str) -> bool:
        """Check if processed dataset is cached."""
        cache_path = self.get_cache_path(dataset_name, subset, split)
        return cache_path.exists()
        
    def process_and_cache(self, 
                         smiles_list: List[str],
                         dataset_name: str,
                         subset: str,
                         split: str,
                         properties: Optional[Dict[str, List[float]]] = None) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Process and cache molecular data.
        
        Args:
            smiles_list: List of SMILES strings
            dataset_name: Name of the dataset
            subset: Dataset subset
            split: Data split
            properties: Optional molecular properties
            
        Returns:
            Tuple of (processed_smiles, processed_properties)
        """
        cache_path = self.get_cache_path(dataset_name, subset, split)
        
        # Try to load from cache
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached data from {cache_path}")
                return cached_data['smiles'], cached_data.get('properties', {})
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                
        # Process data
        logger.info(f"Processing {len(smiles_list)} molecules for {dataset_name}_{subset}_{split}")
        
        processed_smiles = []
        processed_properties = {key: [] for key in (properties or {})}
        
        try:
            from rdkit import Chem
            
            for i, smiles in enumerate(tqdm(smiles_list, desc="Processing molecules")):
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Canonicalize SMILES
                    canonical_smiles = Chem.MolToSmiles(mol)
                    processed_smiles.append(canonical_smiles)
                    
                    # Add properties if available
                    for prop_name, prop_values in (properties or {}).items():
                        if i < len(prop_values):
                            processed_properties[prop_name].append(prop_values[i])
                            
        except ImportError:
            logger.warning("RDKit not available, using raw SMILES without validation")
            processed_smiles = smiles_list
            processed_properties = properties or {}
            
        # Cache processed data
        try:
            import pickle
            cache_data = {
                'smiles': processed_smiles,
                'properties': processed_properties,
                'metadata': {
                    'dataset': dataset_name,
                    'subset': subset,
                    'split': split,
                    'original_count': len(smiles_list),
                    'processed_count': len(processed_smiles)
                }
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached processed data to {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
            
        logger.info(f"Processed {len(processed_smiles)}/{len(smiles_list)} valid molecules")
        
        return processed_smiles, processed_properties


def download_and_prepare_dataset(dataset_name: str,
                                subset: str = "250k",
                                data_dir: str = "data",
                                cache_dir: Optional[str] = None,
                                force_download: bool = False) -> Dict[str, Path]:
    """
    Download and prepare a molecular dataset.
    
    Args:
        dataset_name: Name of dataset ("zinc15", "qm9")
        subset: Dataset subset
        data_dir: Directory to store data
        cache_dir: Directory for caching
        force_download: Whether to force re-download
        
    Returns:
        Dictionary mapping split names to processed file paths
    """
    if dataset_name.lower() == "zinc15":
        downloader = ZINC15Downloader(data_dir=data_dir, cache_dir=cache_dir)
        
        # Download raw data
        downloaded_files = downloader.download_subset(subset)
        
        # Process files
        processed_files = {}
        for split, filepath in downloaded_files.items():
            processed_path = downloader.process_zinc15_file(filepath)
            processed_files[split] = processed_path
            
        return processed_files
        
    elif dataset_name.lower() == "qm9":
        downloader = QM9Downloader(data_dir=data_dir, cache_dir=cache_dir)
        
        # Download processed version by default
        qm9_file = downloader.download_qm9("processed")
        
        return {"full": qm9_file}
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")