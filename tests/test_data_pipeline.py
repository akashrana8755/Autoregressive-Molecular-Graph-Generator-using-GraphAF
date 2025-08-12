"""
Tests for data loading and preprocessing pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.dataset_downloader import DatasetDownloader, ZINC15Downloader, QM9Downloader
from src.data.data_validator import MolecularValidator, DataQualityAnalyzer
from src.data.data_statistics import MolecularStatistics
from src.data.preprocessing_pipeline import MolecularDataPipeline


class TestDatasetDownloader:
    """Test dataset downloader functionality."""
    
    def test_dataset_downloader_init(self):
        """Test DatasetDownloader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = DatasetDownloader(data_dir=temp_dir)
            
            assert downloader.data_dir.exists()
            assert downloader.cache_dir.exists()
            
    def test_zinc15_downloader_init(self):
        """Test ZINC15Downloader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = ZINC15Downloader(data_dir=temp_dir)
            
            assert downloader.zinc_dir.exists()
            assert "250k" in downloader.ZINC15_URLS
            assert "train" in downloader.ZINC15_URLS["250k"]
            
    def test_qm9_downloader_init(self):
        """Test QM9Downloader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = QM9Downloader(data_dir=temp_dir)
            
            assert downloader.qm9_dir.exists()
            assert "processed" in downloader.QM9_URLS


class TestMolecularValidator:
    """Test molecular validator functionality."""
    
    def test_validator_init_without_rdkit(self):
        """Test validator initialization without RDKit."""
        validator = MolecularValidator(use_rdkit=False)
        assert not validator.use_rdkit
        
    def test_basic_smiles_validation(self):
        """Test basic SMILES validation."""
        validator = MolecularValidator(use_rdkit=False)
        
        # Valid SMILES
        assert validator._basic_smiles_validation("CCO")
        assert validator._basic_smiles_validation("c1ccccc1")
        assert validator._basic_smiles_validation("CC(=O)O")
        
        # Invalid SMILES
        assert not validator._basic_smiles_validation("CC(O")  # Unbalanced parentheses
        assert not validator._basic_smiles_validation("CC[O")  # Unbalanced brackets
        assert not validator._basic_smiles_validation("")      # Empty string
        
    def test_validate_smiles_basic(self):
        """Test SMILES validation without RDKit."""
        validator = MolecularValidator(use_rdkit=False)
        
        result = validator.validate_smiles("CCO")
        assert result['smiles'] == "CCO"
        assert result['is_valid']
        assert result['canonical_smiles'] == "CCO"
        assert len(result['errors']) == 0
        
        result = validator.validate_smiles("")
        assert not result['is_valid']
        assert len(result['errors']) > 0
        
    def test_validate_dataset_basic(self):
        """Test dataset validation without RDKit."""
        validator = MolecularValidator(use_rdkit=False)
        
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "invalid("]
        results = validator.validate_dataset(smiles_list, show_progress=False)
        
        assert results['total_molecules'] == 4
        assert results['valid_molecules'] == 3
        assert results['invalid_molecules'] == 1
        assert results['validity_rate'] == 0.75
        
    def test_get_duplicates(self):
        """Test duplicate detection."""
        validator = MolecularValidator(use_rdkit=False)
        
        smiles_list = ["CCO", "c1ccccc1", "CCO", "CC(=O)O"]
        results = validator.get_duplicates(smiles_list)
        
        assert results['total_molecules'] == 4
        assert results['unique_molecules'] == 3
        assert results['duplicate_molecules'] == 1
        assert results['uniqueness_rate'] == 0.75


class TestDataQualityAnalyzer:
    """Test data quality analyzer functionality."""
    
    def test_analyzer_init(self):
        """Test analyzer initialization."""
        analyzer = DataQualityAnalyzer()
        assert analyzer.validator is not None
        
    def test_analyze_dataset_basic(self):
        """Test basic dataset analysis."""
        analyzer = DataQualityAnalyzer()
        
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        properties = {"logp": [0.1, 2.1, -0.5]}
        
        analysis = analyzer.analyze_dataset(
            smiles_list=smiles_list,
            properties=properties,
            dataset_name="test_dataset"
        )
        
        assert analysis['dataset_name'] == "test_dataset"
        assert analysis['basic_stats']['total_molecules'] == 3
        assert 'validation_results' in analysis
        assert 'duplicate_analysis' in analysis
        assert 'property_analysis' in analysis
        assert 'quality_score' in analysis
        
    def test_create_quality_report(self):
        """Test quality report creation."""
        analyzer = DataQualityAnalyzer()
        
        # Mock analysis results
        analysis = {
            'dataset_name': 'test',
            'timestamp': '2023-01-01T00:00:00',
            'quality_score': 0.85,
            'basic_stats': {'total_molecules': 100},
            'validation_results': {
                'valid_molecules': 95,
                'total_molecules': 100,
                'validity_rate': 0.95,
                'invalid_molecules': 5,
                'error_counts': {'Parse error': 5},
                'warning_counts': {}
            },
            'duplicate_analysis': {
                'unique_molecules': 90,
                'total_molecules': 100,
                'uniqueness_rate': 0.90,
                'duplicate_groups': 5,
                'duplicate_molecules': 10
            },
            'property_analysis': {}
        }
        
        report = analyzer.create_quality_report(analysis)
        
        assert "MOLECULAR DATASET QUALITY REPORT" in report
        assert "test" in report
        assert "0.850" in report


class TestMolecularStatistics:
    """Test molecular statistics functionality."""
    
    def test_statistics_init_without_rdkit(self):
        """Test statistics initialization without RDKit."""
        stats = MolecularStatistics(use_rdkit=False)
        assert not stats.use_rdkit
        
    def test_compute_basic_statistics(self):
        """Test basic statistics computation."""
        stats = MolecularStatistics(use_rdkit=False)
        
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        results = stats.compute_basic_statistics(smiles_list)
        
        assert results['total_molecules'] == 3
        assert 'smiles_length' in results
        assert 'character_frequency' in results
        assert 'structural_patterns' in results
        
        # Check SMILES length statistics
        length_stats = results['smiles_length']
        assert 'mean' in length_stats
        assert 'std' in length_stats
        assert 'min' in length_stats
        assert 'max' in length_stats
        
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        stats = MolecularStatistics(use_rdkit=False)
        
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        properties = {"logp": [0.1, 2.1, -0.5]}
        
        report = stats.generate_comprehensive_report(
            smiles_list=smiles_list,
            properties=properties,
            dataset_name="test_dataset"
        )
        
        assert report['dataset_name'] == "test_dataset"
        assert 'basic_statistics' in report
        assert 'property_statistics' in report
        assert 'timestamp' in report


class TestMolecularDataPipeline:
    """Test molecular data pipeline functionality."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MolecularDataPipeline(
                data_dir=temp_dir,
                use_rdkit=False
            )
            
            assert pipeline.data_dir.exists()
            assert pipeline.cache_dir.exists()
            assert pipeline.output_dir.exists()
            assert not pipeline.use_rdkit
            
    def test_load_split_data_txt(self):
        """Test loading data from text file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MolecularDataPipeline(data_dir=temp_dir, use_rdkit=False)
            
            # Create test file
            test_file = Path(temp_dir) / "test.txt"
            with open(test_file, 'w') as f:
                f.write("CCO\n")
                f.write("c1ccccc1\n")
                f.write("CC(=O)O\n")
                
            smiles_list, properties = pipeline._load_split_data(test_file, "test")
            
            assert len(smiles_list) == 3
            assert "CCO" in smiles_list
            assert len(properties) == 0
            
    def test_load_split_data_csv(self):
        """Test loading data from CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MolecularDataPipeline(data_dir=temp_dir, use_rdkit=False)
            
            # Create test CSV file
            test_file = Path(temp_dir) / "test.csv"
            with open(test_file, 'w') as f:
                f.write("smiles,logp\n")
                f.write("CCO,0.1\n")
                f.write("c1ccccc1,2.1\n")
                f.write("CC(=O)O,-0.5\n")
                
            smiles_list, properties = pipeline._load_split_data(test_file, "test")
            
            assert len(smiles_list) == 3
            assert "CCO" in smiles_list
            assert "logp" in properties
            assert len(properties["logp"]) == 3


if __name__ == "__main__":
    pytest.main([__file__])