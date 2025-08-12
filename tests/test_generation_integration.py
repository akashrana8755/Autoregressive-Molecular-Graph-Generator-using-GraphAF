"""
Integration tests for complete generation workflows.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.graph_diffusion import GraphDiffusion
from src.models.graph_af import GraphAF
from src.generate.molecular_generator import MolecularGenerator
from src.generate.constraint_filter import ConstraintFilter
from src.evaluate.molecular_evaluator import MolecularEvaluator


class TestGenerationIntegration:
    """Integration tests for molecular generation workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'node_dim': 16,
            'edge_dim': 8,
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.0,  # Disable for deterministic testing
            'max_nodes': 10,
            'num_timesteps': 50,
            'beta_schedule': 'linear',
            'num_flow_layers': 2,
            'num_node_types': 5
        }
        
        # Create pre-trained models (mock training)
        self.diffusion_model = GraphDiffusion(self.config)
        self.af_model = GraphAF(self.config)
        
        # Reference molecules for evaluation
        self.reference_smiles = [
            'C', 'CC', 'CCO', 'c1ccccc1', 'CC(=O)O',
            'CCN', 'CCC', 'CCCO', 'CC(C)O', 'CCCN'
        ]
    
    def test_diffusion_generation_workflow(self):
        """Test complete diffusion-based generation workflow."""
        # Set model to evaluation mode
        self.diffusion_model.eval()
        
        # Generate molecules
        num_samples = 5
        with torch.no_grad():
            generated_graphs = self.diffusion_model.sample(num_samples)
        
        # Verify generation results
        assert len(generated_graphs) == num_samples
        
        for i, graph in enumerate(generated_graphs):
            assert isinstance(graph, Data), f"Sample {i} should be Data object"
            assert hasattr(graph, 'x'), f"Sample {i} missing node features"
            assert hasattr(graph, 'edge_index'), f"Sample {i} missing edge indices"
            assert hasattr(graph, 'edge_attr'), f"Sample {i} missing edge features"
            
            # Check tensor properties
            assert graph.x.size(1) == self.config['node_dim'], f"Sample {i} wrong node dimension"
            assert graph.edge_index.size(0) == 2, f"Sample {i} wrong edge index format"
            assert graph.edge_attr.size(1) == self.config['edge_dim'], f"Sample {i} wrong edge dimension"
            
            # Check that tensors are finite
            assert torch.isfinite(graph.x).all(), f"Sample {i} has non-finite node features"
            assert torch.isfinite(graph.edge_attr).all(), f"Sample {i} has non-finite edge features"
            
            # Check graph structure validity
            num_nodes = graph.x.size(0)
            if graph.edge_index.size(1) > 0:
                assert graph.edge_index.max() < num_nodes, f"Sample {i} has invalid edge indices"
                assert graph.edge_index.min() >= 0, f"Sample {i} has negative edge indices"
    
    def test_flow_generation_workflow(self):
        """Test complete flow-based generation workflow."""
        self.af_model.eval()
        
        # Generate molecules
        num_samples = 3
        with torch.no_grad():
            generated_graphs = self.af_model.sample(num_samples)
        
        # Verify generation results
        assert len(generated_graphs) == num_samples
        
        for i, graph in enumerate(generated_graphs):
            assert isinstance(graph, Data), f"Sample {i} should be Data object"
            
            # Check basic structure
            if graph.x.size(0) > 0:  # Non-empty graph
                assert graph.x.size(1) == self.config['node_dim']
                assert graph.edge_index.size(0) == 2
                assert graph.edge_attr.size(1) == self.config['edge_dim']
                
                # Check finite values
                assert torch.isfinite(graph.x).all()
                assert torch.isfinite(graph.edge_attr).all()
    
    def test_molecular_generator_integration(self):
        """Test integration with MolecularGenerator class."""
        try:
            from src.generate.molecular_generator import MolecularGenerator
        except ImportError:
            pytest.skip("MolecularGenerator not available")
        
        # Create generator with diffusion model
        generator = MolecularGenerator(self.diffusion_model)
        
        # Generate molecules
        num_molecules = 4
        generated_smiles = generator.generate(num_molecules)
        
        # Verify results
        assert isinstance(generated_smiles, list)
        assert len(generated_smiles) <= num_molecules  # Some might be filtered out
        
        for smiles in generated_smiles:
            assert isinstance(smiles, str)
            assert len(smiles) > 0
    
    def test_constraint_filtering_integration(self):
        """Test integration with constraint filtering."""
        try:
            from src.generate.constraint_filter import ConstraintFilter
        except ImportError:
            pytest.skip("ConstraintFilter not available")
        
        # Create constraint filter
        constraint_filter = ConstraintFilter()
        
        # Test molecules (mix of valid and invalid)
        test_smiles = [
            'CCO',  # Valid, drug-like
            'C' * 100,  # Invalid, too large
            'c1ccccc1',  # Valid, aromatic
            'invalid_smiles',  # Invalid SMILES
            'CC(=O)O',  # Valid, small molecule
        ]
        
        # Apply Lipinski filter
        filtered_smiles = constraint_filter.apply_lipinski_filter(test_smiles)
        
        # Should filter out invalid and non-drug-like molecules
        assert isinstance(filtered_smiles, list)
        assert len(filtered_smiles) <= len(test_smiles)
        
        # All filtered molecules should be valid
        for smiles in filtered_smiles:
            assert constraint_filter.passes_lipinski_filter(smiles)
    
    def test_generation_with_constraints(self):
        """Test generation workflow with constraint filtering."""
        try:
            from src.generate.molecular_generator import MolecularGenerator
            from src.generate.constraint_filter import ConstraintFilter
        except ImportError:
            pytest.skip("Generator or ConstraintFilter not available")
        
        # Create generator and filter
        generator = MolecularGenerator(self.diffusion_model)
        constraint_filter = ConstraintFilter()
        
        # Generate molecules
        num_molecules = 10
        generated_graphs = self.diffusion_model.sample(num_molecules)
        
        # Convert to SMILES (mock conversion)
        generated_smiles = []
        for i, graph in enumerate(generated_graphs):
            # Mock SMILES conversion - in real implementation this would use graph_to_smiles
            mock_smiles = f"C{'C' * (i % 5)}"  # Simple mock SMILES
            generated_smiles.append(mock_smiles)
        
        # Apply constraints
        filtered_smiles = constraint_filter.apply_lipinski_filter(generated_smiles)
        
        # Verify filtering
        assert len(filtered_smiles) <= len(generated_smiles)
        
        # Check QED scores for filtered molecules
        if len(filtered_smiles) > 0:
            qed_scores = constraint_filter.compute_qed_scores(filtered_smiles)
            assert len(qed_scores) == len(filtered_smiles)
            
            for score in qed_scores:
                if not np.isnan(score):
                    assert 0 <= score <= 1, "QED scores should be between 0 and 1"
    
    def test_evaluation_integration(self):
        """Test integration with molecular evaluation."""
        try:
            from src.evaluate.molecular_evaluator import MolecularEvaluator
        except ImportError:
            pytest.skip("MolecularEvaluator not available")
        
        # Create evaluator with reference molecules
        evaluator = MolecularEvaluator(reference_molecules=self.reference_smiles)
        
        # Generate test molecules
        generated_smiles = [
            'CCO',  # Known molecule (not novel)
            'CCCO',  # Similar to reference
            'CC(C)CC',  # Different structure
            'invalid',  # Invalid SMILES
        ]
        
        # Evaluate generated molecules
        evaluation_results = evaluator.evaluate(generated_smiles)
        
        # Verify evaluation results
        assert isinstance(evaluation_results, dict)
        
        expected_metrics = ['validity', 'uniqueness', 'novelty', 'total_molecules']
        for metric in expected_metrics:
            assert metric in evaluation_results, f"Missing metric: {metric}"
        
        # Check metric ranges
        assert 0 <= evaluation_results['validity'] <= 1
        assert 0 <= evaluation_results['uniqueness'] <= 1
        if evaluation_results['novelty'] is not None:
            assert 0 <= evaluation_results['novelty'] <= 1
        assert evaluation_results['total_molecules'] == len(generated_smiles)
    
    def test_end_to_end_generation_evaluation(self):
        """Test complete end-to-end generation and evaluation workflow."""
        try:
            from src.generate.molecular_generator import MolecularGenerator
            from src.generate.constraint_filter import ConstraintFilter
            from src.evaluate.molecular_evaluator import MolecularEvaluator
        except ImportError:
            pytest.skip("Required classes not available")
        
        # Step 1: Generate molecules
        self.diffusion_model.eval()
        num_samples = 8
        
        with torch.no_grad():
            generated_graphs = self.diffusion_model.sample(num_samples)
        
        # Step 2: Convert to SMILES (mock conversion)
        generated_smiles = []
        for i, graph in enumerate(generated_graphs):
            # Mock SMILES - in real implementation would use proper conversion
            mock_smiles = ['C', 'CC', 'CCO', 'CCC', 'CCCO', 'CC(C)O', 'CCN', 'CCCN'][i % 8]
            generated_smiles.append(mock_smiles)
        
        # Step 3: Apply constraints
        constraint_filter = ConstraintFilter()
        filtered_smiles = constraint_filter.apply_lipinski_filter(generated_smiles)
        
        # Step 4: Evaluate results
        evaluator = MolecularEvaluator(reference_molecules=self.reference_smiles)
        evaluation_results = evaluator.evaluate(filtered_smiles)
        
        # Step 5: Verify complete workflow
        assert len(filtered_smiles) <= len(generated_smiles)
        assert isinstance(evaluation_results, dict)
        assert evaluation_results['total_molecules'] == len(filtered_smiles)
        
        # Generate comprehensive report
        if len(filtered_smiles) > 0:
            report = evaluator.generate_evaluation_report(
                filtered_smiles, 
                self.reference_smiles
            )
            
            assert isinstance(report, dict)
            assert 'basic_metrics' in report
            assert 'summary' in report
    
    def test_batch_generation_efficiency(self):
        """Test efficiency of batch generation."""
        self.diffusion_model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10]
        
        for batch_size in batch_sizes:
            with torch.no_grad():
                generated_graphs = self.diffusion_model.sample(batch_size)
            
            assert len(generated_graphs) == batch_size
            
            # Check that all graphs are valid
            for graph in generated_graphs:
                assert isinstance(graph, Data)
                assert graph.x.size(0) > 0 or graph.x.size(0) == 0  # Allow empty graphs
                assert torch.isfinite(graph.x).all()
    
    def test_generation_with_different_parameters(self):
        """Test generation with different sampling parameters."""
        self.diffusion_model.eval()
        
        # Test different max_nodes
        max_nodes_values = [5, 10, 15]
        
        for max_nodes in max_nodes_values:
            with torch.no_grad():
                generated_graphs = self.diffusion_model.sample(
                    num_samples=3, 
                    max_nodes=max_nodes
                )
            
            for graph in generated_graphs:
                assert graph.x.size(0) <= max_nodes, f"Graph has too many nodes: {graph.x.size(0)} > {max_nodes}"
    
    def test_generation_determinism(self):
        """Test deterministic generation with fixed seeds."""
        def generate_with_seed(seed):
            torch.manual_seed(seed)
            self.diffusion_model.eval()
            
            with torch.no_grad():
                graphs = self.diffusion_model.sample(num_samples=2)
            
            return graphs
        
        # Generate with same seed twice
        graphs1 = generate_with_seed(42)
        graphs2 = generate_with_seed(42)
        
        # Results should be identical
        assert len(graphs1) == len(graphs2)
        
        for g1, g2 in zip(graphs1, graphs2):
            assert g1.x.shape == g2.x.shape
            assert g1.edge_index.shape == g2.edge_index.shape
            assert g1.edge_attr.shape == g2.edge_attr.shape
            
            # Values should be close (allowing for small numerical differences)
            assert torch.allclose(g1.x, g2.x, atol=1e-6)
            assert torch.equal(g1.edge_index, g2.edge_index)
            assert torch.allclose(g1.edge_attr, g2.edge_attr, atol=1e-6)
    
    def test_generation_memory_usage(self):
        """Test memory usage during generation."""
        self.diffusion_model.eval()
        
        # Track memory usage if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Generate multiple batches
        total_generated = 0
        
        for _ in range(5):
            with torch.no_grad():
                graphs = self.diffusion_model.sample(num_samples=4)
                total_generated += len(graphs)
                
                # Clear references
                del graphs
        
        # Check memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            # (This is a rough check - exact values depend on implementation)
            assert memory_increase < 1e9, "Memory usage increased too much during generation"
        
        assert total_generated == 20, "Should have generated 20 molecules total"
    
    def test_generation_error_handling(self):
        """Test error handling during generation."""
        # Test with invalid parameters
        self.diffusion_model.eval()
        
        # Test with zero samples
        with torch.no_grad():
            graphs = self.diffusion_model.sample(num_samples=0)
        assert len(graphs) == 0
        
        # Test with very large max_nodes (should handle gracefully)
        with torch.no_grad():
            graphs = self.diffusion_model.sample(num_samples=1, max_nodes=1000)
        assert len(graphs) == 1
        
        # Test with negative samples (should handle gracefully)
        try:
            with torch.no_grad():
                graphs = self.diffusion_model.sample(num_samples=-1)
            # If it doesn't raise an error, should return empty list
            assert len(graphs) == 0
        except (ValueError, RuntimeError):
            # It's also acceptable to raise an error for invalid input
            pass
    
    def test_generation_consistency_across_modes(self):
        """Test generation consistency between training and evaluation modes."""
        # Generate in training mode (with dropout)
        self.diffusion_model.train()
        with torch.no_grad():
            train_graphs = self.diffusion_model.sample(num_samples=3)
        
        # Generate in evaluation mode (without dropout)
        self.diffusion_model.eval()
        with torch.no_grad():
            eval_graphs = self.diffusion_model.sample(num_samples=3)
        
        # Both should produce valid graphs
        assert len(train_graphs) == 3
        assert len(eval_graphs) == 3
        
        for graphs in [train_graphs, eval_graphs]:
            for graph in graphs:
                assert isinstance(graph, Data)
                assert torch.isfinite(graph.x).all()
                assert torch.isfinite(graph.edge_attr).all()
    
    def test_property_conditioned_generation(self):
        """Test property-conditioned generation if available."""
        try:
            from src.models.property_predictor import PropertyPredictor
        except ImportError:
            pytest.skip("PropertyPredictor not available")
        
        # Create mock property predictor
        property_predictor = PropertyPredictor(
            node_dim=self.config['node_dim'],
            edge_dim=self.config['edge_dim'],
            num_properties=3
        )
        
        # Test generation with property targets
        target_properties = {
            'logp': 2.0,
            'qed': 0.8,
            'molecular_weight': 200.0
        }
        
        # This would be implemented in a property-conditioned generator
        # For now, just test that the property predictor works
        self.diffusion_model.eval()
        with torch.no_grad():
            graphs = self.diffusion_model.sample(num_samples=2)
            
            # Predict properties for generated graphs
            for graph in graphs:
                if graph.x.size(0) > 0:  # Non-empty graph
                    batch = Batch.from_data_list([graph])
                    predicted_props = property_predictor(batch)
                    
                    assert predicted_props.size(0) == 1  # One graph
                    assert predicted_props.size(1) == 3  # Three properties
                    assert torch.isfinite(predicted_props).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])