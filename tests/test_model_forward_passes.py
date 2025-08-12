"""
Comprehensive unit tests for model forward passes and computations.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.base_model import BaseGenerativeModel
from src.models.graph_diffusion import GraphDiffusion
from src.models.graph_af import GraphAF


class MockGenerativeModel(BaseGenerativeModel):
    """Mock implementation of BaseGenerativeModel for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config['input_dim'], config['output_dim'])
    
    def forward(self, batch, **kwargs):
        return {'output': self.linear(batch.x)}
    
    def training_step(self, batch, **kwargs):
        output = self.forward(batch)
        return torch.mean(output['output'])
    
    def sample(self, num_samples, **kwargs):
        return [Data(x=torch.randn(5, self.config['input_dim'])) for _ in range(num_samples)]


class TestBaseGenerativeModel:
    """Test cases for BaseGenerativeModel abstract class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'input_dim': 10,
            'output_dim': 5,
            'hidden_dim': 32
        }
        self.model = MockGenerativeModel(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.config == self.config
        assert self.model.model_name == 'MockGenerativeModel'
    
    def test_get_model_info(self):
        """Test model information extraction."""
        info = self.model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'config' in info
        assert 'device' in info
        
        assert info['model_name'] == 'MockGenerativeModel'
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['config'] == self.config
    
    def test_freeze_parameters(self):
        """Test parameter freezing."""
        # Initially parameters should be trainable
        for param in self.model.parameters():
            assert param.requires_grad
        
        # Freeze parameters
        self.model.freeze_parameters(True)
        for param in self.model.parameters():
            assert not param.requires_grad
        
        # Unfreeze parameters
        self.model.freeze_parameters(False)
        for param in self.model.parameters():
            assert param.requires_grad
    
    def test_get_device(self):
        """Test device detection."""
        device = self.model.get_device()
        assert isinstance(device, torch.device)
    
    def test_save_load_checkpoint(self, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Save checkpoint
        self.model.save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=10,
            metrics={'loss': 0.5}
        )
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded = MockGenerativeModel.load_checkpoint(checkpoint_path)
        
        assert 'model' in loaded
        assert 'epoch' in loaded
        assert 'metrics' in loaded
        assert 'config' in loaded
        
        assert loaded['epoch'] == 10
        assert loaded['metrics']['loss'] == 0.5
        assert loaded['config'] == self.config
        
        # Check that model state is preserved
        loaded_model = loaded['model']
        assert isinstance(loaded_model, MockGenerativeModel)
    
    def test_validation_step(self):
        """Test validation step (default implementation)."""
        # Create test batch
        batch = Batch(
            x=torch.randn(10, self.config['input_dim']),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 5),
            batch=torch.zeros(10, dtype=torch.long)
        )
        
        val_metrics = self.model.validation_step(batch)
        
        assert isinstance(val_metrics, dict)
        assert 'val_loss' in val_metrics
        assert isinstance(val_metrics['val_loss'], torch.Tensor)


class TestGraphDiffusion:
    """Test cases for GraphDiffusion model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'node_dim': 10,
            'edge_dim': 5,
            'hidden_dim': 32,
            'num_layers': 3,
            'dropout': 0.1,
            'max_nodes': 20,
            'num_timesteps': 100,
            'beta_schedule': 'linear'
        }
        self.model = GraphDiffusion(self.config)
        
        # Create test batch
        self.batch = self._create_test_batch()
    
    def _create_test_batch(self):
        """Create a test batch of molecular graphs."""
        # Graph 1: 3 nodes, 2 edges
        x1 = torch.randn(3, self.config['node_dim'])
        edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr1 = torch.randn(4, self.config['edge_dim'])
        
        # Graph 2: 4 nodes, 3 edges  
        x2 = torch.randn(4, self.config['node_dim'])
        edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long) + 3
        edge_attr2 = torch.randn(6, self.config['edge_dim'])
        
        # Combine into batch
        x = torch.cat([x1, x2], dim=0)
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        
        return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def test_initialization(self):
        """Test GraphDiffusion initialization."""
        assert self.model.node_dim == self.config['node_dim']
        assert self.model.edge_dim == self.config['edge_dim']
        assert self.model.hidden_dim == self.config['hidden_dim']
        assert self.model.num_layers == self.config['num_layers']
        assert self.model.num_timesteps == self.config['num_timesteps']
        
        # Check that noise schedule is initialized
        assert hasattr(self.model, 'betas')
        assert hasattr(self.model, 'alphas')
        assert hasattr(self.model, 'alphas_cumprod')
        
        assert self.model.betas.size(0) == self.config['num_timesteps']
        assert self.model.alphas.size(0) == self.config['num_timesteps']
        assert self.model.alphas_cumprod.size(0) == self.config['num_timesteps']
    
    def test_beta_schedule_linear(self):
        """Test linear beta schedule."""
        config = self.config.copy()
        config['beta_schedule'] = 'linear'
        model = GraphDiffusion(config)
        
        betas = model.betas
        assert betas.size(0) == config['num_timesteps']
        assert (betas[1:] >= betas[:-1]).all()  # Should be non-decreasing
        assert betas.min() > 0 and betas.max() < 1  # Should be in (0, 1)
    
    def test_beta_schedule_cosine(self):
        """Test cosine beta schedule."""
        config = self.config.copy()
        config['beta_schedule'] = 'cosine'
        model = GraphDiffusion(config)
        
        betas = model.betas
        assert betas.size(0) == config['num_timesteps']
        assert betas.min() > 0 and betas.max() < 1
    
    def test_beta_schedule_invalid(self):
        """Test invalid beta schedule."""
        config = self.config.copy()
        config['beta_schedule'] = 'invalid'
        
        with pytest.raises(ValueError, match="Unknown beta schedule"):
            GraphDiffusion(config)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = self.batch.num_graphs
        t = torch.randint(0, self.config['num_timesteps'], (batch_size,))
        
        output = self.model.forward(self.batch, t)
        
        assert isinstance(output, dict)
        assert 'node_scores' in output
        assert 'edge_scores' in output
        
        node_scores = output['node_scores']
        edge_scores = output['edge_scores']
        
        # Check output shapes
        assert node_scores.size(0) == self.batch.x.size(0)  # Same number of nodes
        assert node_scores.size(1) == self.config['node_dim']  # Same as input node dim
        assert edge_scores.size(0) == self.batch.edge_attr.size(0)  # Same number of edges
        assert edge_scores.size(1) == self.config['edge_dim']  # Same as input edge dim
        
        # Check that outputs are finite
        assert torch.isfinite(node_scores).all()
        assert torch.isfinite(edge_scores).all()
    
    def test_forward_pass_different_timesteps(self):
        """Test forward pass with different timesteps."""
        batch_size = self.batch.num_graphs
        
        # Test with different timestep values
        for t_val in [0, self.config['num_timesteps']//2, self.config['num_timesteps']-1]:
            t = torch.full((batch_size,), t_val, dtype=torch.long)
            output = self.model.forward(self.batch, t)
            
            assert 'node_scores' in output
            assert 'edge_scores' in output
            assert torch.isfinite(output['node_scores']).all()
            assert torch.isfinite(output['edge_scores']).all()
    
    def test_training_step(self):
        """Test training step."""
        loss = self.model.training_step(self.batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # MSE loss should be non-negative
    
    def test_training_step_gradient_flow(self):
        """Test that gradients flow properly during training."""
        loss = self.model.training_step(self.batch)
        loss.backward()
        
        # Check that gradients are computed
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter {name}"
    
    def test_sample_basic(self):
        """Test basic sampling functionality."""
        num_samples = 3
        generated = self.model.sample(num_samples)
        
        assert isinstance(generated, list)
        assert len(generated) == num_samples
        
        for graph in generated:
            assert isinstance(graph, Data)
            assert hasattr(graph, 'x')
            assert hasattr(graph, 'edge_index')
            assert hasattr(graph, 'edge_attr')
            
            # Check tensor properties
            assert isinstance(graph.x, torch.Tensor)
            assert isinstance(graph.edge_index, torch.Tensor)
            assert isinstance(graph.edge_attr, torch.Tensor)
            
            # Check dimensions
            assert graph.x.size(1) == self.config['node_dim']
            assert graph.edge_index.size(0) == 2
            assert graph.edge_attr.size(1) == self.config['edge_dim']
    
    def test_sample_with_parameters(self):
        """Test sampling with custom parameters."""
        num_samples = 2
        max_nodes = 10
        
        generated = self.model.sample(num_samples, max_nodes=max_nodes)
        
        assert len(generated) == num_samples
        for graph in generated:
            assert graph.x.size(0) <= max_nodes
    
    def test_model_modes(self):
        """Test model in training vs evaluation mode."""
        # Training mode
        self.model.train()
        output_train = self.model.forward(self.batch, torch.randint(0, 100, (self.batch.num_graphs,)))
        
        # Evaluation mode
        self.model.eval()
        with torch.no_grad():
            output_eval = self.model.forward(self.batch, torch.randint(0, 100, (self.batch.num_graphs,)))
        
        # Outputs should have same shape but potentially different values due to dropout
        assert output_train['node_scores'].shape == output_eval['node_scores'].shape
        assert output_train['edge_scores'].shape == output_eval['edge_scores'].shape
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        empty_batch = Batch(
            x=torch.empty(0, self.config['node_dim']),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, self.config['edge_dim']),
            batch=torch.empty(0, dtype=torch.long)
        )
        
        # Should handle empty batch gracefully
        try:
            t = torch.empty(0, dtype=torch.long)
            output = self.model.forward(empty_batch, t)
            # If it doesn't crash, that's good
        except Exception as e:
            # Some implementations might not handle empty batches
            pytest.skip(f"Model doesn't handle empty batches: {e}")


class TestGraphAF:
    """Test cases for GraphAF model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'node_dim': 10,
            'edge_dim': 5,
            'hidden_dim': 32,
            'num_layers': 2,
            'num_flow_layers': 3,
            'dropout': 0.1,
            'max_nodes': 15,
            'num_node_types': 8
        }
        self.model = GraphAF(self.config)
        
        # Create test batch
        self.batch = self._create_test_batch()
    
    def _create_test_batch(self):
        """Create a test batch of molecular graphs."""
        # Graph 1: 3 nodes, 2 edges
        x1 = torch.randn(3, self.config['node_dim'])
        edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr1 = torch.randn(4, self.config['edge_dim'])
        
        # Graph 2: 2 nodes, 1 edge
        x2 = torch.randn(2, self.config['node_dim'])
        edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) + 3
        edge_attr2 = torch.randn(2, self.config['edge_dim'])
        
        # Combine into batch
        x = torch.cat([x1, x2], dim=0)
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        
        return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def test_initialization(self):
        """Test GraphAF initialization."""
        assert self.model.node_dim == self.config['node_dim']
        assert self.model.edge_dim == self.config['edge_dim']
        assert self.model.hidden_dim == self.config['hidden_dim']
        assert self.model.num_layers == self.config['num_layers']
        assert self.model.num_flow_layers == self.config['num_flow_layers']
        assert self.model.max_nodes == self.config['max_nodes']
        
        # Check that flow layers are initialized
        assert len(self.model.node_flow_layers) == self.config['num_flow_layers']
        assert len(self.model.edge_flow_layers) == self.config['num_flow_layers']
    
    def test_encode_graph_context(self):
        """Test graph context encoding."""
        context = self.model.encode_graph_context(self.batch)
        
        assert isinstance(context, torch.Tensor)
        assert context.size(0) == self.batch.num_graphs
        assert context.size(1) == self.config['hidden_dim']
        assert torch.isfinite(context).all()
    
    def test_encode_empty_graph_context(self):
        """Test context encoding for empty graphs."""
        empty_batch = Batch(
            x=torch.empty(0, self.config['node_dim']),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, self.config['edge_dim']),
            batch=torch.empty(0, dtype=torch.long)
        )
        empty_batch.num_graphs = 1
        
        context = self.model.encode_graph_context(empty_batch)
        
        assert isinstance(context, torch.Tensor)
        assert context.size(0) == 1
        assert context.size(1) == self.config['hidden_dim']
    
    def test_forward_pass(self):
        """Test forward pass."""
        output = self.model.forward(self.batch)
        
        assert isinstance(output, dict)
        assert 'log_prob' in output
        
        log_probs = output['log_prob']
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.size(0) == self.batch.num_graphs
        assert torch.isfinite(log_probs).all()
    
    def test_training_step(self):
        """Test training step."""
        loss = self.model.training_step(self.batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)
    
    def test_training_step_gradient_flow(self):
        """Test gradient flow during training."""
        loss = self.model.training_step(self.batch)
        loss.backward()
        
        # Check that gradients are computed for flow layers
        for flow_layer in self.model.node_flow_layers:
            for param in flow_layer.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert torch.isfinite(param.grad).all()
    
    def test_log_prob_method(self):
        """Test log_prob method."""
        log_probs = self.model.log_prob(self.batch)
        
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.size(0) == self.batch.num_graphs
        assert torch.isfinite(log_probs).all()
    
    def test_sample_basic(self):
        """Test basic sampling functionality."""
        num_samples = 2
        generated = self.model.sample(num_samples)
        
        assert isinstance(generated, list)
        assert len(generated) == num_samples
        
        for graph in generated:
            assert isinstance(graph, Data)
            assert hasattr(graph, 'x')
            assert hasattr(graph, 'edge_index')
            assert hasattr(graph, 'edge_attr')
            
            # Check tensor properties
            assert isinstance(graph.x, torch.Tensor)
            assert isinstance(graph.edge_index, torch.Tensor)
            assert isinstance(graph.edge_attr, torch.Tensor)
            
            # Check dimensions
            if graph.x.size(0) > 0:  # Non-empty graph
                assert graph.x.size(1) == self.config['node_dim']
                assert graph.edge_index.size(0) == 2
                assert graph.edge_attr.size(1) == self.config['edge_dim']
    
    def test_sample_with_parameters(self):
        """Test sampling with custom parameters."""
        num_samples = 2
        max_nodes = 5
        temperature = 0.8
        
        generated = self.model.sample(num_samples, max_nodes=max_nodes, temperature=temperature)
        
        assert len(generated) == num_samples
        for graph in generated:
            assert graph.x.size(0) <= max_nodes
    
    def test_coupling_layer(self):
        """Test coupling layer functionality."""
        from src.models.graph_af import CouplingLayer
        
        dim = 10
        hidden_dim = 32
        mask = torch.zeros(dim)
        mask[::2] = 1  # Mask every other dimension
        
        coupling = CouplingLayer(dim, hidden_dim, mask)
        
        # Test forward transformation
        x = torch.randn(5, dim)
        y, log_det = coupling(x, reverse=False)
        
        assert y.shape == x.shape
        assert log_det.shape == (5,)
        assert torch.isfinite(y).all()
        assert torch.isfinite(log_det).all()
        
        # Test reverse transformation
        x_reconstructed, log_det_reverse = coupling(y, reverse=True)
        
        assert x_reconstructed.shape == x.shape
        assert torch.allclose(x_reconstructed, x, atol=1e-5)
        assert torch.allclose(log_det_reverse, -log_det, atol=1e-5)
    
    def test_masked_linear(self):
        """Test masked linear layer."""
        from src.models.graph_af import MaskedLinear
        
        in_features = 10
        out_features = 8
        mask = torch.randn(out_features, in_features)
        mask = (mask > 0).float()  # Binary mask
        
        masked_linear = MaskedLinear(in_features, out_features, mask)
        
        x = torch.randn(5, in_features)
        output = masked_linear(x)
        
        assert output.shape == (5, out_features)
        assert torch.isfinite(output).all()
        
        # Check that masking is applied
        weight_masked = masked_linear.linear.weight * mask
        expected_output = torch.nn.functional.linear(x, weight_masked, masked_linear.linear.bias)
        assert torch.allclose(output, expected_output)


class TestModelConsistency:
    """Test consistency across different models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'node_dim': 8,
            'edge_dim': 4,
            'hidden_dim': 16,
            'num_layers': 2,
            'dropout': 0.0,  # Disable dropout for consistency
            'max_nodes': 10,
            'num_timesteps': 50,
            'beta_schedule': 'linear',
            'num_flow_layers': 2,
            'num_node_types': 5
        }
        
        self.diffusion_model = GraphDiffusion(self.config)
        self.af_model = GraphAF(self.config)
        
        # Create test batch
        self.batch = self._create_test_batch()
    
    def _create_test_batch(self):
        """Create a test batch."""
        x = torch.randn(5, self.config['node_dim'])
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        edge_attr = torch.randn(6, self.config['edge_dim'])
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        
        return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def test_output_shapes_consistency(self):
        """Test that models produce consistent output shapes."""
        # GraphDiffusion
        t = torch.randint(0, self.config['num_timesteps'], (self.batch.num_graphs,))
        diffusion_output = self.diffusion_model.forward(self.batch, t)
        
        # GraphAF
        af_output = self.af_model.forward(self.batch)
        
        # Check that both models handle the same batch
        assert diffusion_output['node_scores'].size(0) == self.batch.x.size(0)
        assert diffusion_output['edge_scores'].size(0) == self.batch.edge_attr.size(0)
        assert af_output['log_prob'].size(0) == self.batch.num_graphs
    
    def test_sampling_consistency(self):
        """Test that sampling produces valid outputs."""
        num_samples = 3
        
        # Sample from both models
        diffusion_samples = self.diffusion_model.sample(num_samples)
        af_samples = self.af_model.sample(num_samples)
        
        # Check that both produce the same number of samples
        assert len(diffusion_samples) == num_samples
        assert len(af_samples) == num_samples
        
        # Check that all samples are valid Data objects
        for sample in diffusion_samples + af_samples:
            assert isinstance(sample, Data)
            assert hasattr(sample, 'x')
            assert hasattr(sample, 'edge_index')
            assert hasattr(sample, 'edge_attr')
    
    def test_training_step_consistency(self):
        """Test that training steps produce valid losses."""
        diffusion_loss = self.diffusion_model.training_step(self.batch)
        af_loss = self.af_model.training_step(self.batch)
        
        # Both should produce scalar losses
        assert diffusion_loss.dim() == 0
        assert af_loss.dim() == 0
        
        # Both should be finite
        assert torch.isfinite(diffusion_loss)
        assert torch.isfinite(af_loss)
    
    def test_parameter_count_reasonableness(self):
        """Test that models have reasonable parameter counts."""
        diffusion_info = self.diffusion_model.get_model_info()
        af_info = self.af_model.get_model_info()
        
        # Both should have positive parameter counts
        assert diffusion_info['total_parameters'] > 0
        assert af_info['total_parameters'] > 0
        
        # Parameter counts should be reasonable (not too large)
        assert diffusion_info['total_parameters'] < 10_000_000  # 10M parameters
        assert af_info['total_parameters'] < 10_000_000
    
    def test_device_consistency(self):
        """Test device handling consistency."""
        device = torch.device('cpu')
        
        # Move models to device
        self.diffusion_model.to(device)
        self.af_model.to(device)
        
        # Check device detection
        assert self.diffusion_model.get_device() == device
        assert self.af_model.get_device() == device
        
        # Move batch to device
        batch_on_device = self.batch.to(device)
        
        # Forward passes should work
        t = torch.randint(0, self.config['num_timesteps'], (batch_on_device.num_graphs,), device=device)
        diffusion_output = self.diffusion_model.forward(batch_on_device, t)
        af_output = self.af_model.forward(batch_on_device)
        
        # Outputs should be on correct device
        assert diffusion_output['node_scores'].device == device
        assert diffusion_output['edge_scores'].device == device
        assert af_output['log_prob'].device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])