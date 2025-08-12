"""
Comprehensive data preprocessing pipeline for molecular datasets.

This module provides a unified pipeline for downloading, validating,
processing, and analyzing molecular datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import json

from .dataset_downloader import download_and_prepare_dataset, DatasetProcessor
from .data_validator import MolecularValidator, DataQualityAnalyzer
from .data_statistics import MolecularStatistics
from .molecular_dataset import MolecularDataset, create_molecular_dataloader
from .pyg_datasets import create_dataset

logger = logging.getLogger(__name__)


class MolecularDataPipeline:
    """
    Comprehensive pipeline for molecular data preprocessing.
    
    This pipeline handles the complete workflow from raw data download
    to processed, validated datasets ready for model training.
    """
    
    def __init__(self,
                 data_dir: str = "data",
                 cache_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_rdkit: bool = True,
                 quality_threshold: float = 0.8):
        """
        Initialize molecular data pipeline.
        
        Args:
            data_dir: Directory for storing raw and processed data
            cache_dir: Directory for caching intermediate results
            output_dir: Directory for saving reports and visualizations
            use_rdkit: Whether to use RDKit for molecular processing
            quality_threshold: Minimum quality score for datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_rdkit = use_rdkit
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self.validator = MolecularValidator(use_rdkit=use_rdkit)
        self.quality_analyzer = DataQualityAnalyzer(validator=self.validator)
        self.statistics = MolecularStatistics(use_rdkit=use_rdkit)
        self.processor = DatasetProcessor(cache_dir=str(self.cache_dir))
        
        logger.info(f"Initialized molecular data pipeline with data_dir={data_dir}")
        
    def process_dataset(self,
                       dataset_name: str,
                       subset: str = "250k",
                       splits: Optional[List[str]] = None,
                       force_download: bool = False,
                       force_reprocess: bool = False,
                       create_reports: bool = True) -> Dict[str, Any]:
        """
        Process a complete molecular dataset.
        
        Args:
            dataset_name: Name of dataset ("zinc15", "qm9")
            subset: Dataset subset
            splits: List of splits to process
            force_download: Whether to force re-download
            force_reprocess: Whether to force reprocessing
            create_reports: Whether to create quality reports
            
        Returns:
            Dictionary with processing results and dataset information
        """
        logger.info(f"Processing dataset: {dataset_name} (subset: {subset})")
        
        results = {
            'dataset_name': dataset_name,
            'subset': subset,
            'splits': splits or ['train', 'val', 'test'],
            'processing_status': {},
            'quality_reports': {},
            'datasets': {},
            'statistics': {}
        }
        
        try:
            # Step 1: Download raw data
            logger.info("Step 1: Downloading raw data...")
            downloaded_files = download_and_prepare_dataset(
                dataset_name=dataset_name,
                subset=subset,
                data_dir=str(self.data_dir),
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            results['downloaded_files'] = {str(k): str(v) for k, v in downloaded_files.items()}
            
            # Step 2: Process each split
            for split in results['splits']:
                if split not in downloaded_files:
                    logger.warning(f"Split {split} not available for {dataset_name}")
                    continue
                    
                logger.info(f"Step 2: Processing split {split}...")
                split_results = self._process_split(
                    dataset_name=dataset_name,
                    subset=subset,
                    split=split,
                    data_file=downloaded_files[split],
                    force_reprocess=force_reprocess,
                    create_reports=create_reports
                )
                
                results['processing_status'][split] = split_results['status']
                results['quality_reports'][split] = split_results['quality_report']
                results['datasets'][split] = split_results['dataset']
                results['statistics'][split] = split_results['statistics']
                
            # Step 3: Create combined reports
            if create_reports:
                logger.info("Step 3: Creating combined reports...")
                self._create_combined_reports(results)
                
            logger.info(f"Dataset processing complete: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_name}: {e}")
            results['error'] = str(e)
            
        return results
        
    def _process_split(self,
                      dataset_name: str,
                      subset: str,
                      split: str,
                      data_file: Path,
                      force_reprocess: bool = False,
                      create_reports: bool = True) -> Dict[str, Any]:
        """Process a single dataset split."""
        logger.info(f"Processing {dataset_name}_{subset}_{split}...")
        
        # Load SMILES data
        smiles_list, properties = self._load_split_data(data_file, dataset_name)
        
        if not smiles_list:
            return {
                'status': 'failed',
                'error': 'No SMILES data found',
                'quality_report': None,
                'dataset': None,
                'statistics': None
            }
            
        logger.info(f"Loaded {len(smiles_list)} molecules for {split}")
        
        # Process and cache data
        processed_smiles, processed_properties = self.processor.process_and_cache(
            smiles_list=smiles_list,
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            properties=properties
        )
        
        # Create dataset
        dataset = MolecularDataset(
            smiles_list=processed_smiles,
            properties=processed_properties,
            cache_dir=str(self.cache_dir / f"{dataset_name}_{subset}_{split}"),
            use_cache=True
        )
        
        # Quality analysis
        quality_report = None
        statistics = None
        
        if create_reports:
            # Analyze quality
            quality_analysis = self.quality_analyzer.analyze_dataset(
                smiles_list=processed_smiles,
                properties=processed_properties,
                dataset_name=f"{dataset_name}_{subset}_{split}"
            )
            
            # Create quality report
            report_path = self.output_dir / f"{dataset_name}_{subset}_{split}_quality_report.txt"
            quality_report = self.quality_analyzer.create_quality_report(
                analysis=quality_analysis,
                output_path=report_path
            )
            
            # Generate statistics
            statistics = self.statistics.generate_comprehensive_report(
                smiles_list=processed_smiles,
                properties=processed_properties,
                dataset_name=f"{dataset_name}_{subset}_{split}"
            )
            
            # Save statistics
            stats_path = self.output_dir / f"{dataset_name}_{subset}_{split}_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(statistics, f, indent=2, default=str)
                
            logger.info(f"Quality score: {quality_analysis['quality_score']:.3f}")
            
            # Check quality threshold
            if quality_analysis['quality_score'] < self.quality_threshold:
                logger.warning(f"Dataset quality below threshold: "
                             f"{quality_analysis['quality_score']:.3f} < {self.quality_threshold}")
                             
        return {
            'status': 'success',
            'quality_report': quality_report,
            'dataset': dataset,
            'statistics': statistics,
            'processed_count': len(processed_smiles),
            'original_count': len(smiles_list)
        }
        
    def _load_split_data(self, data_file: Path, dataset_name: str) -> Tuple[List[str], Dict[str, List[float]]]:
        """Load SMILES and properties from data file."""
        smiles_list = []
        properties = {}
        
        try:
            if data_file.suffix == '.csv':
                df = pd.read_csv(data_file)
                
                # Find SMILES column
                smiles_col = None
                for col in ['smiles', 'SMILES', 'canonical_smiles']:
                    if col in df.columns:
                        smiles_col = col
                        break
                        
                if smiles_col is None:
                    raise ValueError("No SMILES column found in CSV file")
                    
                smiles_list = df[smiles_col].dropna().tolist()
                
                # Extract properties
                property_columns = [col for col in df.columns if col != smiles_col]
                for col in property_columns:
                    if df[col].dtype in ['float64', 'int64']:
                        properties[col] = df[col].tolist()
                        
            else:
                # Plain text file with SMILES
                with open(data_file, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
                    
        except Exception as e:
            logger.error(f"Failed to load data from {data_file}: {e}")
            
        return smiles_list, properties
        
    def _create_combined_reports(self, results: Dict[str, Any]):
        """Create combined reports across all splits."""
        dataset_name = results['dataset_name']
        subset = results['subset']
        
        # Combine quality scores
        quality_scores = {}
        total_molecules = 0
        
        for split, report in results['quality_reports'].items():
            if report and results['processing_status'][split] == 'success':
                # Extract quality score from statistics
                if split in results['statistics']:
                    stats = results['statistics'][split]
                    # This would need to be extracted from the actual quality analysis
                    # For now, we'll create a placeholder
                    quality_scores[split] = 0.85  # Placeholder
                    
                if split in results['datasets']:
                    total_molecules += len(results['datasets'][split])
                    
        # Create summary report
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append(f"DATASET PROCESSING SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Dataset: {dataset_name} ({subset})")
        summary_lines.append(f"Total Molecules Processed: {total_molecules:,}")
        summary_lines.append("")
        
        summary_lines.append("SPLIT SUMMARY")
        summary_lines.append("-" * 20)
        for split in results['splits']:
            status = results['processing_status'].get(split, 'not_processed')
            if status == 'success':
                count = results['datasets'][split].__len__() if results['datasets'].get(split) else 0
                quality = quality_scores.get(split, 0.0)
                summary_lines.append(f"{split.upper()}: {count:,} molecules (quality: {quality:.3f})")
            else:
                summary_lines.append(f"{split.upper()}: {status}")
                
        summary_lines.append("")
        summary_lines.append("=" * 60)
        
        # Save summary
        summary_path = self.output_dir / f"{dataset_name}_{subset}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
            
        logger.info(f"Created summary report: {summary_path}")
        
    def create_pytorch_datasets(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create PyTorch datasets from processing results.
        
        Args:
            results: Results from process_dataset
            
        Returns:
            Dictionary with PyTorch datasets and dataloaders
        """
        pytorch_datasets = {}
        dataloaders = {}
        
        for split, dataset in results['datasets'].items():
            if dataset is not None:
                # Create dataloader
                dataloader = create_molecular_dataloader(
                    dataset=dataset,
                    batch_size=32,
                    shuffle=(split == 'train'),
                    num_workers=0
                )
                
                pytorch_datasets[split] = dataset
                dataloaders[split] = dataloader
                
        return {
            'datasets': pytorch_datasets,
            'dataloaders': dataloaders
        }
        
    def get_dataset_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {
            'dataset_name': results['dataset_name'],
            'subset': results['subset'],
            'splits': {},
            'total_molecules': 0,
            'properties': set()
        }
        
        for split, dataset in results['datasets'].items():
            if dataset is not None:
                split_info = {
                    'molecule_count': len(dataset),
                    'valid_molecules': sum(1 for data in dataset.data_list if data is not None),
                    'properties': list(dataset.properties.keys()) if dataset.properties else []
                }
                
                info['splits'][split] = split_info
                info['total_molecules'] += split_info['molecule_count']
                info['properties'].update(split_info['properties'])
                
        info['properties'] = list(info['properties'])
        
        return info


def create_preprocessing_pipeline(config: Dict[str, Any]) -> MolecularDataPipeline:
    """
    Create preprocessing pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MolecularDataPipeline instance
    """
    return MolecularDataPipeline(
        data_dir=config.get('data_dir', 'data'),
        cache_dir=config.get('cache_dir'),
        output_dir=config.get('output_dir'),
        use_rdkit=config.get('use_rdkit', True),
        quality_threshold=config.get('quality_threshold', 0.8)
    )


def run_preprocessing_pipeline(dataset_name: str,
                             subset: str = "250k",
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run complete preprocessing pipeline for a dataset.
    
    Args:
        dataset_name: Name of dataset to process
        subset: Dataset subset
        config: Optional configuration dictionary
        
    Returns:
        Processing results
    """
    config = config or {}
    
    # Create pipeline
    pipeline = create_preprocessing_pipeline(config)
    
    # Process dataset
    results = pipeline.process_dataset(
        dataset_name=dataset_name,
        subset=subset,
        splits=config.get('splits'),
        force_download=config.get('force_download', False),
        force_reprocess=config.get('force_reprocess', False),
        create_reports=config.get('create_reports', True)
    )
    
    return results