"""
Performance and memory usage tests for the molecular generation system.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import time
import gc
import psutil
import os
from unittest.mock import patch
import numpy as np

from src.models.graph_diffusion import GraphDiffusion
from src.models.graph_af import GraphAF
from src.data.molecular_dataset import MolecularDataset


class TestPerformanceMemory:
    """Performance and memory usage tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'node_dim': 32,
            'edge_dim': 16,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'max_nodes': 20,
            'num_timesteps': 100,
            'beta_schedule': 'linear',
            'num_flow_layers': 3,
            'num_node_types': 10
        }
        
        # Create test data
        self.test_graphs = self._create_test_graphs(100)  # 100 test graphs
    
    def _create_test_graphs(self, num_graphs):
        """Create test molecular graphs."""
        graphs = []
        for i in range(num_graphs):
            num_nodes = 5 + (i % 15)  # 5 to 19 nodes
            
            x = torch.randn(num_nodes, self.config['node_dim'])
            
            # Create random connectivity
            num_edges = min(num_nodes * 2, 40)  # Reasonable number of edges
            edge_indices = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, self.config['edge_dim'])
            
            graph = Data(
                x=x,
                edge_index=edge_indices,
                edge_attr=edge_attr
            )
            graphs.append(graph)
        
        return graphs
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _get_gpu_memory_usage(self):
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        return 0
    
    def test_model_forward_pass_performance(self):
        """Test forward pass performance."""
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Create test batch
        batch = Batch.from_data_list(self.test_graphs[:32])  # Batch of 32
        timesteps = torch.randint(0, 100, (batch.num_graphs,))
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model.forward(batch, timesteps)
        
        # Measure performance
        num_runs = 20
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model.forward(batch, timesteps)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Performance should be reasonable (less than 1 second per forward pass)
        assert avg_time < 1.0, f"Forward pass too slow: {avg_time:.3f}s"
        
        # Check throughput (molecules per second)
        throughput = batch.num_graphs / avg_time
        assert throughput > 10, f"Throughput too low: {throughput:.1f} molecules/s"
    
    def test_training_step_performance(self):
        """Test training step performance."""
        model = GraphDiffusion(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create data loader
        data_loader = DataLoader(
            self.test_graphs[:64],
            batch_size=8,
            shuffle=True
        )
        
        model.train()
        
        # Warm up
        for i, batch in enumerate(data_loader):
            if i >= 3:
                break
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
        
        # Measure training performance
        num_steps = 10
        start_time = time.time()
        
        step_count = 0
        for batch in data_loader:
            if step_count >= num_steps:
                break
                
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            
            step_count += 1
        
        end_time = time.time()
        avg_step_time = (end_time - start_time) / num_steps
        
        # Training step should be reasonably fast
        assert avg_step_time < 2.0, f"Training step too slow: {avg_step_time:.3f}s"
    
    def test_generation_performance(self):
        """Test molecule generation performance."""
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = model.sample(num_samples=5)
        
        # Measure generation performance
        num_samples = 20
        start_time = time.time()
        
        with torch.no_grad():
            generated = model.sample(num_samples=num_samples)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Check that generation completed
        assert len(generated) == num_samples
        
        # Generation should be reasonably fast
        time_per_molecule = generation_time / num_samples
        assert time_per_molecule < 5.0, f"Generation too slow: {time_per_molecule:.3f}s per molecule"
    
    def test_memory_usage_forward_pass(self):
        """Test memory usage during forward passes."""
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Measure initial memory
        gc.collect()
        initial_memory = self._get_memory_usage()
        initial_gpu_memory = self._get_gpu_memory_usage()
        
        # Process batches of different sizes
        batch_sizes = [1, 4, 8, 16, 32]
        memory_usage = []
        
        for batch_size in batch_sizes:
            batch = Batch.from_data_list(self.test_graphs[:batch_size])
            timesteps = torch.randint(0, 100, (batch.num_graphs,))
            
            with torch.no_grad():
                _ = model.forward(batch, timesteps)
            
            current_memory = self._get_memory_usage()
            memory_increase = current_memory - initial_memory
            memory_usage.append(memory_increase)
        
        # Memory usage should scale reasonably with batch size
        # Larger batches should use more memory, but not excessively
        assert memory_usage[-1] > memory_usage[0], "Memory should increase with batch size"
        assert memory_usage[-1] < 1000, f"Memory usage too high: {memory_usage[-1]:.1f}MB"
    
    def test_memory_usage_training(self):
        """Test memory usage during training."""
        model = GraphDiffusion(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Measure initial memory
        gc.collect()
        initial_memory = self._get_memory_usage()
        
        # Create data loader
        data_loader = DataLoader(
            self.test_graphs[:32],
            batch_size=4,
            shuffle=True
        )
        
        model.train()
        memory_measurements = []
        
        # Train for several steps
        for i, batch in enumerate(data_loader):
            if i >= 10:
                break
                
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            
            current_memory = self._get_memory_usage()
            memory_increase = current_memory - initial_memory
            memory_measurements.append(memory_increase)
        
        # Memory usage should stabilize (not grow indefinitely)
        if len(memory_measurements) >= 5:
            early_avg = np.mean(memory_measurements[:3])
            late_avg = np.mean(memory_measurements[-3:])
            memory_growth = late_avg - early_avg
            
            # Memory shouldn't grow too much during training
            assert memory_growth < 200, f"Memory growing too much during training: {memory_growth:.1f}MB"
    
    def test_memory_usage_generation(self):
        """Test memory usage during generation."""
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Measure initial memory
        gc.collect()
        initial_memory = self._get_memory_usage()
        
        # Generate molecules in batches
        total_generated = 0
        memory_measurements = []
        
        for _ in range(5):
            with torch.no_grad():
                generated = model.sample(num_samples=10)
            
            total_generated += len(generated)
            
            current_memory = self._get_memory_usage()
            memory_increase = current_memory - initial_memory
            memory_measurements.append(memory_increase)
            
            # Clear generated molecules
            del generated
            gc.collect()
        
        # Memory usage should not grow significantly
        max_memory_increase = max(memory_measurements)
        assert max_memory_increase < 500, f"Memory usage too high during generation: {max_memory_increase:.1f}MB"
        
        assert total_generated == 50, "Should have generated 50 molecules"
    
    def test_gpu_memory_efficiency(self):
        """Test GPU memory efficiency if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        model = GraphDiffusion(self.config).to(device)
        model.eval()
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated()
        
        # Move test data to GPU
        gpu_graphs = [graph.to(device) for graph in self.test_graphs[:16]]
        batch = Batch.from_data_list(gpu_graphs)
        timesteps = torch.randint(0, 100, (batch.num_graphs,), device=device)
        
        # Forward pass
        with torch.no_grad():
            output = model.forward(batch, timesteps)
        
        peak_gpu_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_gpu_memory - initial_gpu_memory) / 1024 / 1024  # MB
        
        # GPU memory usage should be reasonable
        assert memory_used < 2000, f"GPU memory usage too high: {memory_used:.1f}MB"
        
        # Clean up
        del batch, output, gpu_graphs
        torch.cuda.empty_cache()
    
    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        model = GraphDiffusion(self.config)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16]
        times = []
        
        for batch_size in batch_sizes:
            batch = Batch.from_data_list(self.test_graphs[:batch_size])
            timesteps = torch.randint(0, 100, (batch.num_graphs,))
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = model.forward(batch, timesteps)
            
            # Measure time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model.forward(batch, timesteps)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            times.append(avg_time)
        
        # Time should scale sub-linearly with batch size (batching efficiency)
        time_per_molecule = [t / bs for t, bs in zip(times, batch_sizes)]
        
        # Larger batches should be more efficient per molecule
        assert time_per_molecule[-1] < time_per_molecule[0], \
            "Batching should improve efficiency per molecule"
    
    def test_model_size_efficiency(self):
        """Test model parameter efficiency."""
        # Test different model sizes
        configs = [
            {'hidden_dim': 32, 'num_layers': 2},   # Small
            {'hidden_dim': 64, 'num_layers': 3},   # Medium
            {'hidden_dim': 128, 'num_layers': 4},  # Large
        ]
        
        for i, config_update in enumerate(configs):
            config = self.config.copy()
            config.update(config_update)
            
            model = GraphDiffusion(config)
            model_info = model.get_model_info()
            
            param_count = model_info['total_parameters']
            
            # Parameter count should be reasonable
            if i == 0:  # Small model
                assert param_count < 100_000, f"Small model too large: {param_count} parameters"
            elif i == 1:  # Medium model
                assert param_count < 500_000, f"Medium model too large: {param_count} parameters"
            else:  # Large model
                assert param_count < 2_000_000, f"Large model too large: {param_count} parameters"
    
    def test_data_loading_performance(self):
        """Test data loading performance."""
        # Create larger dataset
        large_smiles = ['C', 'CC', 'CCO', 'c1ccccc1'] * 250  # 1000 molecules
        
        dataset = MolecularDataset(
            smiles_list=large_smiles,
            use_cache=False
        )
        
        # Test different numbers of workers
        worker_counts = [0, 1, 2]
        loading_times = []
        
        for num_workers in worker_counts:
            data_loader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=lambda x: Batch.from_data_list([item for item in x if item is not None])
            )
            
            start_time = time.time()
            
            batch_count = 0
            for batch in data_loader:
                batch_count += 1
                if batch_count >= 20:  # Test first 20 batches
                    break
            
            end_time = time.time()
            loading_time = end_time - start_time
            loading_times.append(loading_time)
        
        # Data loading should complete in reasonable time
        for loading_time in loading_times:
            assert loading_time < 30, f"Data loading too slow: {loading_time:.1f}s"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Measure initial memory
        gc.collect()
        initial_memory = self._get_memory_usage()
        
        # Perform repeated operations
        for i in range(20):
            batch = Batch.from_data_list(self.test_graphs[:8])
            timesteps = torch.randint(0, 100, (batch.num_graphs,))
            
            with torch.no_grad():
                output = model.forward(batch, timesteps)
            
            # Clear references
            del batch, timesteps, output
            
            # Periodic garbage collection
            if i % 5 == 0:
                gc.collect()
        
        # Final memory check
        gc.collect()
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (no significant leaks)
        assert memory_increase < 100, f"Potential memory leak detected: {memory_increase:.1f}MB increase"
    
    def test_concurrent_processing(self):
        """Test concurrent processing performance."""
        import threading
        import queue
        
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Create work queue
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add work items
        for i in range(20):
            batch = Batch.from_data_list(self.test_graphs[i*2:(i+1)*2])
            timesteps = torch.randint(0, 100, (batch.num_graphs,))
            work_queue.put((batch, timesteps))
        
        def worker():
            while True:
                try:
                    batch, timesteps = work_queue.get(timeout=1)
                    with torch.no_grad():
                        output = model.forward(batch, timesteps)
                    result_queue.put(output)
                    work_queue.task_done()
                except queue.Empty:
                    break
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        # Wait for completion
        work_queue.join()
        
        # Wait for threads to finish
        for t in threads:
            t.join()
        
        # Check results
        results_count = result_queue.qsize()
        assert results_count == 20, f"Expected 20 results, got {results_count}"
    
    def test_large_molecule_handling(self):
        """Test performance with large molecules."""
        # Create large molecules
        large_graphs = []
        for i in range(10):
            num_nodes = 50 + i * 10  # 50 to 140 nodes
            
            x = torch.randn(num_nodes, self.config['node_dim'])
            
            # Create more edges for larger molecules
            num_edges = num_nodes * 3
            edge_indices = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, self.config['edge_dim'])
            
            graph = Data(
                x=x,
                edge_index=edge_indices,
                edge_attr=edge_attr
            )
            large_graphs.append(graph)
        
        model = GraphDiffusion(self.config)
        model.eval()
        
        # Test forward pass with large molecules
        start_time = time.time()
        
        for graph in large_graphs[:5]:  # Test first 5
            batch = Batch.from_data_list([graph])
            timesteps = torch.randint(0, 100, (1,))
            
            with torch.no_grad():
                output = model.forward(batch, timesteps)
            
            # Check that output is reasonable
            assert torch.isfinite(output['node_scores']).all()
            assert torch.isfinite(output['edge_scores']).all()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle large molecules in reasonable time
        assert total_time < 10, f"Large molecule processing too slow: {total_time:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])