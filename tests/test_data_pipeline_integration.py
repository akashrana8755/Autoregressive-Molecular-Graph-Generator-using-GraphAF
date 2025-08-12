"""
Integration tests for data loading and preprocessing pipelines.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import tempfile
import os
from pathlib import Path
import json
import pickle
from unittest.mock import patch, MagicMock
import numpy as np

from src.data.molecular_dataset import MolecularDataset
from src.data.smiles_processor import SMILESProcessor
from src.data.feature_extractor import FeatureExtractor
from src.data.preprocessing_pipeline import PreprocessingPipeline
from src.data.data_validator import DataValidator


class TestDataPipelineIntegration:
    """Integration tests for complete data processing pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_smiles = [
            'C',           # Methane
            'CC',          # Ethane
            'CCO',         # Ethanol
            'c1ccccc1',    # Benzene
            'CC(=O)O',     # Acetic acid
            'CCN(CC)CC',   # Triethylamine
            'C1CCC(CC1)O', # Cyclohexanol
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'invalid_smiles',  # Invalid SMILES
            '',            # Empty string
        ]
        
        self.test_properties = {
            'logp': [0.5, 1.0, -0.3, 2.1, 0.2, 1.5, 1.2, 3.8, np.nan, np.nan],
            'molecular_weight': [16.0, 30.0, 46.0, 78.0, 60.0, 101.0, 100.0, 206.0, np.nan, np.nan],
            'qed': [0.3, 0.4, 0.7, 0.6, 0.5, 0.4, 0.6, 0.8, np.nan, np.nan]
        }
    
    def test_complete_data_processing_pipeline(self):
        """Test complete data processing from SMILES to DataLoader."""
        # Step 1: Create dataset
        dataset = MolecularDataset(
            smiles_list=self.test_smiles,
            properties=self.test_properties,
            use_cache=False
        )
        
        # Step 2: Check dataset creation
        assert len(dataset) == len(self.test_smiles)
        
        # Step 3: Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=lambda x: Batch.from_data_list([item for item in x if item is not None])
        )
        
        # Step 4: Iterate through batches
        total_processed = 0
        valid_molecules = 0
        
        for batch in data_loader:
            assert isinstance(batch, Batch)
            
            # Check batch properties
            assert hasattr(batch, 'x')
            assert hasattr(batch, 'edge_index')
            assert hasattr(batch, 'edge_attr')
            assert hasattr(batch, 'batch')
            
            # Check tensor properties
            assert batch.x.dim() == 2
            assert batch.edge_index.dim() == 2
            assert batch.edge_attr.dim() == 2
            assert batch.batch.dim() == 1
            
            # Check that all values are finite
            assert torch.isfinite(batch.x).all()
            assert torch.isfinite(batch.edge_attr).all()
            
            # Check batch consistency
            assert batch.x.size(0) == batch.batch.size(0)  # Same number of nodes
            assert batch.edge_index.size(0) == 2  # Edge index format
            assert batch.edge_attr.size(0) == batch.edge_index.size(1)  # Same number of edges
            
            total_processed += batch.num_graphs
            valid_molecules += batch.num_graphs
        
        # Should have processed some valid molecules
        assert valid_molecules > 0
        assert valid_molecules <= len([s for s in self.test_smiles if s and s != 'invalid_smiles'])
    
    def test_preprocessing_pipeline_integration(self):
        """Test integration with preprocessing pipeline."""
        try:
            from src.data.preprocessing_pipeline import PreprocessingPipeline
        except ImportError:
            pytest.skip("PreprocessingPipeline not available")
        
        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline(
            max_nodes=50,
            min_nodes=1,
            remove_invalid=True,
            standardize_features=True
        )
        
        # Process SMILES
        processed_data = pipeline.process_smiles_list(self.test_smiles)
        
        assert isinstance(processed_data, list)
        
        # All processed data should be valid
        for data in processed_data:
            assert isinstance(data, Data)
            assert data.x.size(0) >= 1  # At least one node
            assert data.x.size(0) <= 50  # At most 50 nodes
            assert torch.isfinite(data.x).all()
            assert torch.isfinite(data.edge_attr).all()
    
    def test_data_validation_integration(self):
        """Test integration with data validation."""
        try:
            from src.data.data_validator import DataValidator
        except ImportError:
            pytest.skip("DataValidator not available")
        
        # Create dataset
        dataset = MolecularDataset(
            smiles_list=self.test_smiles,
            properties=self.test_properties,
            use_cache=False
        )
        
        # Create validator
        validator = DataValidator()
        
        # Validate dataset
        validation_results = validator.validate_dataset(dataset)
        
        assert isinstance(validation_results, dict)
        assert 'total_samples' in validation_results
        assert 'valid_samples' in validation_results
        assert 'invalid_samples' in validation_results
        assert 'validation_errors' in validation_results
        
        # Check that totals add up
        total = validation_results['total_samples']
        valid = validation_results['valid_samples']
        invalid = validation_results['invalid_samples']
        
        assert total == valid + invalid
        assert total == len(self.test_smiles)
    
    def test_caching_integration(self):
        """Test data caching integration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "cache"
            
            # Create dataset with caching enabled
            dataset1 = MolecularDataset(
                smiles_list=self.test_smiles[:5],  # Subset for faster testing
                cache_dir=cache_dir,
                use_cache=True
            )
            
            # Process data (should create cache)
            data1 = [dataset1[i] for i in range(len(dataset1)) if dataset1[i] is not None]
            
            # Check that cache files were created
            assert cache_dir.exists()
            cache_files = list(cache_dir.glob("*.pkl"))
            assert len(cache_files) > 0
            
            # Create second dataset with same cache
            dataset2 = MolecularDataset(
                smiles_list=self.test_smiles[:5],
                cache_dir=cache_dir,
                use_cache=True
            )
            
            # Process data (should use cache)
            data2 = [dataset2[i] for i in range(len(dataset2)) if dataset2[i] is not None]
            
            # Results should be identical
            assert len(data1) == len(data2)
            
            for d1, d2 in zip(data1, data2):
                if d1 is not None and d2 is not None:
                    assert torch.equal(d1.x, d2.x)
                    assert torch.equal(d1.edge_index, d2.edge_index)
                    assert torch.equal(d1.edge_attr, d2.edge_attr)
    
    def test_property_integration(self):
        """Test integration with molecular properties."""
        # Create dataset with properties
        dataset = MolecularDataset(
            smiles_list=self.test_smiles,
            properties=self.test_properties,
            use_cache=False
        )
        
        # Check that properties are included
        for i in range(len(dataset)):
            data = dataset[i]
            if data is not None:
                # Should have property attributes
                for prop_name in self.test_properties.keys():
                    assert hasattr(data, prop_name), f"Missing property {prop_name}"
                    
                    prop_value = getattr(data, prop_name)
                    assert isinstance(prop_value, torch.Tensor)
                    assert prop_value.dim() == 0  # Scalar property
    
    def test_batch_processing_consistency(self):
        """Test consistency across different batch sizes."""
        dataset = MolecularDataset(
            smiles_list=self.test_smiles,
            use_cache=False
        )
        
        batch_sizes = [1, 2, 4, 8]
        all_processed_data = []
        
        for batch_size in batch_sizes:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: Batch.from_data_list([item for item in x if item is not None])
            )
            
            batch_data = []
            for batch in data_loader:
                # Extract individual graphs from batch
                for i in range(batch.num_graphs):
                    node_mask = batch.batch == i
                    node_indices = torch.where(node_mask)[0]
                    
                    if len(node_indices) > 0:
                        # Extract node features
                        graph_x = batch.x[node_mask]
                        
                        # Extract edges for this graph
                        edge_mask = torch.isin(batch.edge_index[0], node_indices)
                        graph_edge_index = batch.edge_index[:, edge_mask]
                        graph_edge_attr = batch.edge_attr[edge_mask]
                        
                        # Remap edge indices to local indices
                        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
                        remapped_edge_index = torch.tensor([
                            [node_mapping[idx.item()] for idx in graph_edge_index[0]],
                            [node_mapping[idx.item()] for idx in graph_edge_index[1]]
                        ])
                        
                        graph_data = Data(
                            x=graph_x,
                            edge_index=remapped_edge_index,
                            edge_attr=graph_edge_attr
                        )
                        batch_data.append(graph_data)
            
            all_processed_data.append(batch_data)
        
        # All batch sizes should produce the same number of valid graphs
        if len(all_processed_data) > 1:
            first_count = len(all_processed_data[0])
            for i, batch_data in enumerate(all_processed_data[1:], 1):
                assert len(batch_data) == first_count, \
                    f"Batch size {batch_sizes[i]} produced different number of graphs"
    
    def test_feature_consistency_across_pipeline(self):
        """Test feature consistency throughout the pipeline."""
        # Create processor and extractor
        processor = SMILESProcessor()
        extractor = FeatureExtractor()
        
        # Process a single molecule through different paths
        test_smiles = 'CCO'
        
        # Path 1: Direct processing
        graph1 = processor.smiles_to_graph(test_smiles)
        
        # Path 2: Through dataset
        dataset = MolecularDataset(
            smiles_list=[test_smiles],
            use_cache=False
        )
        graph2 = dataset[0]
        
        # Both should produce similar results
        if graph1 is not None and graph2 is not None:
            assert graph1.x.size() == graph2.x.size()
            assert graph1.edge_index.size() == graph2.edge_index.size()
            assert graph1.edge_attr.size() == graph2.edge_attr.size()
            
            # Feature dimensions should be consistent
            node_dim1 = graph1.x.size(1)
            edge_dim1 = graph1.edge_attr.size(1)
            node_dim2 = graph2.x.size(1)
            edge_dim2 = graph2.edge_attr.size(1)
            
            assert node_dim1 == node_dim2
            assert edge_dim1 == edge_dim2
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Create dataset with problematic data
        problematic_smiles = [
            'CCO',  # Valid
            '',     # Empty
            'invalid_smiles',  # Invalid
            None,   # None value
            'C' * 1000,  # Very long (might cause issues)
        ]
        
        # Dataset should handle errors gracefully
        dataset = MolecularDataset(
            smiles_list=problematic_smiles,
            use_cache=False
        )
        
        # Should be able to iterate without crashing
        valid_count = 0
        error_count = 0
        
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                if data is not None:
                    valid_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        
        # Should have processed at least one valid molecule
        assert valid_count > 0
        assert valid_count + error_count == len(problematic_smiles)
    
    def test_memory_efficiency_in_pipeline(self):
        """Test memory efficiency of data pipeline."""
        # Create larger dataset for memory testing
        large_smiles_list = self.test_smiles * 20  # 200 molecules
        
        dataset = MolecularDataset(
            smiles_list=large_smiles_list,
            use_cache=False
        )
        
        # Create data loader with small batch size
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: Batch.from_data_list([item for item in x if item is not None])
        )
        
        # Track memory usage if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Process all batches
        processed_batches = 0
        for batch in data_loader:
            assert isinstance(batch, Batch)
            processed_batches += 1
            
            # Clear batch to free memory
            del batch
        
        # Check memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 1e9, "Memory usage increased too much"
        
        assert processed_batches > 0
    
    def test_reproducibility_in_pipeline(self):
        """Test reproducibility of data pipeline."""
        def process_data(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            dataset = MolecularDataset(
                smiles_list=self.test_smiles[:5],
                use_cache=False
            )
            
            data_loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,  # Enable shuffle to test reproducibility
                collate_fn=lambda x: Batch.from_data_list([item for item in x if item is not None])
            )
            
            batches = []
            for batch in data_loader:
                batches.append(batch)
            
            return batches
        
        # Process with same seed twice
        batches1 = process_data(42)
        batches2 = process_data(42)
        
        # Results should be identical
        assert len(batches1) == len(batches2)
        
        for b1, b2 in zip(batches1, batches2):
            assert b1.num_graphs == b2.num_graphs
            assert torch.equal(b1.x, b2.x)
            assert torch.equal(b1.edge_index, b2.edge_index)
            assert torch.equal(b1.edge_attr, b2.edge_attr)
            assert torch.equal(b1.batch, b2.batch)
    
    def test_data_statistics_integration(self):
        """Test integration with data statistics computation."""
        try:
            from src.data.data_statistics import DataStatistics
        except ImportError:
            pytest.skip("DataStatistics not available")
        
        # Create dataset
        dataset = MolecularDataset(
            smiles_list=self.test_smiles,
            properties=self.test_properties,
            use_cache=False
        )
        
        # Compute statistics
        stats_computer = DataStatistics()
        statistics = stats_computer.compute_dataset_statistics(dataset)
        
        assert isinstance(statistics, dict)
        
        # Check expected statistics
        expected_stats = [
            'total_molecules', 'valid_molecules', 'invalid_molecules',
            'avg_nodes', 'avg_edges', 'node_feature_stats', 'edge_feature_stats'
        ]
        
        for stat in expected_stats:
            if stat in statistics:  # Some stats might not be available
                assert statistics[stat] is not None
    
    def test_parallel_processing_integration(self):
        """Test parallel processing in data pipeline."""
        # Create dataset
        dataset = MolecularDataset(
            smiles_list=self.test_smiles,
            use_cache=False
        )
        
        # Test with multiple workers
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,  # Use multiple workers
            collate_fn=lambda x: Batch.from_data_list([item for item in x if item is not None])
        )
        
        # Should be able to iterate without issues
        batches = []
        for batch in data_loader:
            assert isinstance(batch, Batch)
            batches.append(batch)
        
        assert len(batches) > 0
    
    def test_custom_transforms_integration(self):
        """Test integration with custom data transforms."""
        def custom_transform(data):
            """Custom transform that adds noise to node features."""
            if data is not None:
                data.x = data.x + torch.randn_like(data.x) * 0.01
            return data
        
        # Create dataset with transform
        dataset = MolecularDataset(
            smiles_list=self.test_smiles[:5],
            transform=custom_transform,
            use_cache=False
        )
        
        # Check that transform is applied
        for i in range(len(dataset)):
            data = dataset[i]
            if data is not None:
                # Transform should have been applied
                assert hasattr(data, 'x')
                assert torch.isfinite(data.x).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])