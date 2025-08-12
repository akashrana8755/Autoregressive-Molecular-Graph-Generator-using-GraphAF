"""
Comprehensive unit tests for loss function computations.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from unittest.mock import MagicMock, patch
import numpy as np

from src.training.losses import (
    DiffusionLoss, FlowLoss, PropertyConditionedLoss, 
    AdversarialLoss, MultiObjectiveLoss,
    compute_kl_divergence, compute_reconstruction_loss, focal_loss
)


class TestDiffusionLoss:
    """Test cases for DiffusionLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loss_fn = DiffusionLoss(
            loss_type='mse',
            node_weight=1.0,
            edge_weight=1.0,
            timestep_weighting='uniform'
        )
        
        # Create test data
        self.batch_size = 3
        self.num_nodes = 10
        self.num_edges = 15
        self.node_dim = 8
        self.edge_dim = 4
        
        self.predictions = {
            'node_scores': torch.randn(self.num_nodes, self.node_dim),
            'edge_scores': torch.randn(self.num_edges, self.edge_dim)
        }
        
        self.targets = {
            'node_noise': torch.randn(self.num_nodes, self.node_dim),
            'edge_noise': torch.randn(self.num_edges, self.edge_dim)
        }
        
        self.timesteps = torch.randint(0, 100, (self.batch_size,))
        self.alphas_cumprod = torch.linspace(0.99, 0.01, 100)
    
    def test_initialization(self):
        """Test DiffusionLoss initialization."""
        # Default initialization
        loss_fn = DiffusionLoss()
        assert loss_fn.loss_type == 'mse'
        assert loss_fn.node_weight == 1.0
        assert loss_fn.edge_weight == 1.0
        assert loss_fn.timestep_weighting == 'uniform'
        
        # Custom initialization
        loss_fn = DiffusionLoss(
            loss_type='l1',
            node_weight=0.5,
            edge_weight=2.0,
            timestep_weighting='snr',
            snr_gamma=10.0
        )
        assert loss_fn.loss_type == 'l1'
        assert loss_fn.node_weight == 0.5
        assert loss_fn.edge_weight == 2.0
        assert loss_fn.timestep_weighting == 'snr'
        assert loss_fn.snr_gamma == 10.0
    
    def test_invalid_loss_type(self):
        """Test initialization with invalid loss type."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            DiffusionLoss(loss_type='invalid')
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_dict = self.loss_fn(self.predictions, self.targets, self.timesteps)
        
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'node_loss' in loss_dict
        assert 'edge_loss' in loss_dict
        
        # Check that losses are scalars
        assert loss_dict['total_loss'].dim() == 0
        assert loss_dict['node_loss'].dim() == 0
        assert loss_dict['edge_loss'].dim() == 0
        
        # Check that losses are non-negative (MSE)
        assert loss_dict['total_loss'].item() >= 0
        assert loss_dict['node_loss'].item() >= 0
        assert loss_dict['edge_loss'].item() >= 0
        
        # Check that total loss is combination of node and edge losses
        expected_total = self.loss_fn.node_weight * loss_dict['node_loss'] + \
                        self.loss_fn.edge_weight * loss_dict['edge_loss']
        assert torch.allclose(loss_dict['total_loss'], expected_total)
    
    def test_different_loss_types(self):
        """Test different loss types."""
        loss_types = ['mse', 'l1', 'huber']
        
        for loss_type in loss_types:
            loss_fn = DiffusionLoss(loss_type=loss_type)
            loss_dict = loss_fn(self.predictions, self.targets, self.timesteps)
            
            assert 'total_loss' in loss_dict
            assert torch.isfinite(loss_dict['total_loss'])
            assert loss_dict['total_loss'].item() >= 0
    
    def test_weighted_losses(self):
        """Test weighted node and edge losses."""
        node_weight = 0.3
        edge_weight = 0.7
        
        loss_fn = DiffusionLoss(node_weight=node_weight, edge_weight=edge_weight)
        loss_dict = loss_fn(self.predictions, self.targets, self.timesteps)
        
        # Total loss should be weighted combination
        expected_total = node_weight * loss_dict['node_loss'] + edge_weight * loss_dict['edge_loss']
        assert torch.allclose(loss_dict['total_loss'], expected_total)
    
    def test_snr_weighting(self):
        """Test SNR-based timestep weighting."""
        loss_fn = DiffusionLoss(timestep_weighting='snr', snr_gamma=5.0)
        
        # Create batch for SNR weighting
        batch = Batch(
            x=torch.randn(self.num_nodes, self.node_dim),
            edge_index=torch.randint(0, self.num_nodes, (2, self.num_edges)),
            edge_attr=torch.randn(self.num_edges, self.edge_dim),
            batch=torch.repeat_interleave(torch.arange(self.batch_size), 
                                        torch.tensor([3, 4, 3]))  # 3+4+3=10 nodes
        )
        
        loss_dict = loss_fn(
            self.predictions, self.targets, self.timesteps, 
            self.alphas_cumprod, batch
        )
        
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss'])
    
    def test_truncated_snr_weighting(self):
        """Test truncated SNR weighting."""
        loss_fn = DiffusionLoss(timestep_weighting='truncated_snr', snr_gamma=3.0)
        
        batch = Batch(
            x=torch.randn(self.num_nodes, self.node_dim),
            edge_index=torch.randint(0, self.num_nodes, (2, self.num_edges)),
            edge_attr=torch.randn(self.num_edges, self.edge_dim),
            batch=torch.repeat_interleave(torch.arange(self.batch_size), 
                                        torch.tensor([3, 4, 3]))
        )
        
        loss_dict = loss_fn(
            self.predictions, self.targets, self.timesteps,
            self.alphas_cumprod, batch
        )
        
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss'])
    
    def test_zero_predictions_targets(self):
        """Test with zero predictions and targets."""
        zero_predictions = {
            'node_scores': torch.zeros(self.num_nodes, self.node_dim),
            'edge_scores': torch.zeros(self.num_edges, self.edge_dim)
        }
        
        zero_targets = {
            'node_noise': torch.zeros(self.num_nodes, self.node_dim),
            'edge_noise': torch.zeros(self.num_edges, self.edge_dim)
        }
        
        loss_dict = self.loss_fn(zero_predictions, zero_targets, self.timesteps)
        
        # Loss should be zero when predictions match targets exactly
        assert loss_dict['total_loss'].item() == 0.0
        assert loss_dict['node_loss'].item() == 0.0
        assert loss_dict['edge_loss'].item() == 0.0
    
    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        # Create parameters that require gradients
        node_scores = torch.randn(self.num_nodes, self.node_dim, requires_grad=True)
        edge_scores = torch.randn(self.num_edges, self.edge_dim, requires_grad=True)
        
        predictions = {'node_scores': node_scores, 'edge_scores': edge_scores}
        
        loss_dict = self.loss_fn(predictions, self.targets, self.timesteps)
        loss_dict['total_loss'].backward()
        
        # Check that gradients are computed
        assert node_scores.grad is not None
        assert edge_scores.grad is not None
        assert torch.isfinite(node_scores.grad).all()
        assert torch.isfinite(edge_scores.grad).all()


class TestFlowLoss:
    """Test cases for FlowLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loss_fn = FlowLoss(
            regularization_weight=0.01,
            entropy_regularization=0.1,
            gradient_penalty_weight=1.0
        )
        
        self.batch_size = 4
        self.log_probs = torch.randn(self.batch_size)
    
    def test_initialization(self):
        """Test FlowLoss initialization."""
        loss_fn = FlowLoss()
        assert loss_fn.regularization_weight == 0.0
        assert loss_fn.entropy_regularization == 0.0
        assert loss_fn.gradient_penalty_weight == 0.0
        
        loss_fn = FlowLoss(
            regularization_weight=0.05,
            entropy_regularization=0.2,
            gradient_penalty_weight=2.0
        )
        assert loss_fn.regularization_weight == 0.05
        assert loss_fn.entropy_regularization == 0.2
        assert loss_fn.gradient_penalty_weight == 2.0
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_dict = self.loss_fn(self.log_probs)
        
        assert isinstance(loss_dict, dict)
        assert 'nll_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        # NLL loss should be negative mean of log probs
        expected_nll = -self.log_probs.mean()
        assert torch.allclose(loss_dict['nll_loss'], expected_nll)
        assert torch.allclose(loss_dict['total_loss'], expected_nll)
    
    def test_l2_regularization(self):
        """Test L2 regularization."""
        # Create mock model
        model = nn.Linear(10, 5)
        
        loss_dict = self.loss_fn(self.log_probs, model=model)
        
        assert 'l2_loss' in loss_dict
        assert loss_dict['l2_loss'].item() > 0  # Should have positive L2 loss
        
        # Total loss should include L2 regularization
        expected_total = loss_dict['nll_loss'] + loss_dict['l2_loss']
        assert torch.allclose(loss_dict['total_loss'], expected_total)
    
    def test_entropy_regularization(self):
        """Test entropy regularization."""
        loss_fn = FlowLoss(entropy_regularization=0.5)
        loss_dict = loss_fn(self.log_probs)
        
        assert 'entropy_loss' in loss_dict
        
        # Entropy loss should be negative variance of log probs
        expected_entropy = -0.5 * self.log_probs.var()
        assert torch.allclose(loss_dict['entropy_loss'], expected_entropy)
    
    def test_gradient_penalty(self):
        """Test gradient penalty computation."""
        # Create mock model and batch
        model = MagicMock()
        model.return_value = {'log_prob': torch.randn(2)}
        
        batch = Batch(
            x=torch.randn(5, 8, requires_grad=True),
            edge_index=torch.randint(0, 5, (2, 6)),
            edge_attr=torch.randn(6, 4),
            batch=torch.tensor([0, 0, 1, 1, 1]),
            num_graphs=2
        )
        
        loss_dict = self.loss_fn(self.log_probs[:2], model=model, batch=batch)
        
        # Should compute gradient penalty
        assert 'gradient_penalty' in loss_dict
        assert torch.isfinite(loss_dict['gradient_penalty'])
    
    def test_combined_regularization(self):
        """Test combination of all regularization terms."""
        model = nn.Linear(10, 5)
        batch = Batch(
            x=torch.randn(5, 8, requires_grad=True),
            edge_index=torch.randint(0, 5, (2, 6)),
            edge_attr=torch.randn(6, 4),
            batch=torch.tensor([0, 0, 1, 1, 1]),
            num_graphs=2
        )
        
        # Mock model forward pass
        with patch.object(model, 'forward', return_value={'log_prob': torch.randn(2)}):
            loss_dict = self.loss_fn(self.log_probs[:2], model=model, batch=batch)
        
        # Should have all loss components
        expected_keys = ['nll_loss', 'l2_loss', 'entropy_loss', 'gradient_penalty', 'total_loss']
        for key in expected_keys:
            assert key in loss_dict
        
        # Total loss should be sum of all components
        expected_total = (loss_dict['nll_loss'] + loss_dict['l2_loss'] + 
                         loss_dict['entropy_loss'] + loss_dict['gradient_penalty'])
        assert torch.allclose(loss_dict['total_loss'], expected_total)


class TestPropertyConditionedLoss:
    """Test cases for PropertyConditionedLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generation_loss = DiffusionLoss()
        self.loss_fn = PropertyConditionedLoss(
            generation_loss=self.generation_loss,
            property_loss_weight=0.1,
            property_loss_type='mse'
        )
        
        # Test data
        self.batch_size = 3
        self.num_properties = 4
        
        self.generation_outputs = {
            'node_scores': torch.randn(10, 8),
            'edge_scores': torch.randn(15, 4)
        }
        
        self.generation_targets = {
            'node_noise': torch.randn(10, 8),
            'edge_noise': torch.randn(15, 4)
        }
        
        self.property_predictions = torch.randn(self.batch_size, self.num_properties)
        self.property_targets = torch.randn(self.batch_size, self.num_properties)
        self.timesteps = torch.randint(0, 100, (self.batch_size,))
    
    def test_initialization(self):
        """Test PropertyConditionedLoss initialization."""
        assert self.loss_fn.generation_loss == self.generation_loss
        assert self.loss_fn.property_loss_weight == 0.1
        assert isinstance(self.loss_fn.property_loss_fn, nn.MSELoss)
        
        # Test with different property loss types
        loss_fn = PropertyConditionedLoss(
            generation_loss=self.generation_loss,
            property_loss_type='l1'
        )
        assert isinstance(loss_fn.property_loss_fn, nn.L1Loss)
    
    def test_invalid_property_loss_type(self):
        """Test initialization with invalid property loss type."""
        with pytest.raises(ValueError, match="Unknown property loss type"):
            PropertyConditionedLoss(
                generation_loss=self.generation_loss,
                property_loss_type='invalid'
            )
    
    def test_forward_with_properties(self):
        """Test forward pass with property predictions."""
        loss_dict = self.loss_fn(
            self.generation_outputs,
            self.generation_targets,
            self.property_predictions,
            self.property_targets,
            timesteps=self.timesteps
        )
        
        # Should have both generation and property losses
        assert 'gen_total_loss' in loss_dict
        assert 'gen_node_loss' in loss_dict
        assert 'gen_edge_loss' in loss_dict
        assert 'property_loss' in loss_dict
        assert 'weighted_property_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        # Total loss should include both components
        expected_total = loss_dict['gen_total_loss'] + loss_dict['weighted_property_loss']
        assert torch.allclose(loss_dict['total_loss'], expected_total)
    
    def test_forward_without_properties(self):
        """Test forward pass without property predictions."""
        loss_dict = self.loss_fn(
            self.generation_outputs,
            self.generation_targets,
            timesteps=self.timesteps
        )
        
        # Should only have generation losses
        assert 'gen_total_loss' in loss_dict
        assert 'total_loss' in loss_dict
        assert 'property_loss' not in loss_dict
        
        # Total loss should equal generation loss
        assert torch.allclose(loss_dict['total_loss'], loss_dict['gen_total_loss'])
    
    def test_missing_property_values(self):
        """Test handling of missing property values (NaN)."""
        # Create targets with some NaN values
        property_targets_with_nan = self.property_targets.clone()
        property_targets_with_nan[0, :2] = float('nan')  # First sample, first 2 properties
        property_targets_with_nan[2, 1] = float('nan')   # Third sample, second property
        
        loss_dict = self.loss_fn(
            self.generation_outputs,
            self.generation_targets,
            self.property_predictions,
            property_targets_with_nan,
            timesteps=self.timesteps
        )
        
        # Should still compute property loss for valid values
        assert 'property_loss' in loss_dict
        assert torch.isfinite(loss_dict['property_loss'])
    
    def test_all_nan_properties(self):
        """Test handling when all property values are NaN."""
        property_targets_all_nan = torch.full_like(self.property_targets, float('nan'))
        
        loss_dict = self.loss_fn(
            self.generation_outputs,
            self.generation_targets,
            self.property_predictions,
            property_targets_all_nan,
            timesteps=self.timesteps
        )
        
        # Should not have property loss when all values are NaN
        assert 'property_loss' not in loss_dict
        assert torch.allclose(loss_dict['total_loss'], loss_dict['gen_total_loss'])


class TestAdversarialLoss:
    """Test cases for AdversarialLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock discriminator
        self.discriminator = MagicMock()
        self.discriminator.return_value = torch.randn(3)  # 3 samples
        
        self.loss_fn = AdversarialLoss(
            discriminator=self.discriminator,
            adversarial_weight=0.1,
            gradient_penalty_weight=10.0
        )
        
        # Test batches
        self.real_batch = Batch(
            x=torch.randn(8, 10),
            edge_index=torch.randint(0, 8, (2, 12)),
            edge_attr=torch.randn(12, 5),
            batch=torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),
            num_graphs=3
        )
        
        self.fake_batch = Batch(
            x=torch.randn(8, 10),
            edge_index=torch.randint(0, 8, (2, 12)),
            edge_attr=torch.randn(12, 5),
            batch=torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),
            num_graphs=3
        )
    
    def test_initialization(self):
        """Test AdversarialLoss initialization."""
        assert self.loss_fn.discriminator == self.discriminator
        assert self.loss_fn.adversarial_weight == 0.1
        assert self.loss_fn.gradient_penalty_weight == 10.0
    
    def test_generator_loss(self):
        """Test generator adversarial loss."""
        gen_loss = self.loss_fn.generator_loss(self.fake_batch)
        
        assert isinstance(gen_loss, torch.Tensor)
        assert gen_loss.dim() == 0  # Scalar
        assert torch.isfinite(gen_loss)
        
        # Should call discriminator
        self.discriminator.assert_called_once_with(self.fake_batch)
    
    def test_discriminator_loss(self):
        """Test discriminator loss with gradient penalty."""
        # Reset mock
        self.discriminator.reset_mock()
        self.discriminator.side_effect = [torch.randn(3), torch.randn(3), torch.randn(3)]
        
        loss_dict = self.loss_fn.discriminator_loss(self.real_batch, self.fake_batch)
        
        assert isinstance(loss_dict, dict)
        assert 'discriminator_loss' in loss_dict
        assert 'wasserstein_distance' in loss_dict
        assert 'gradient_penalty' in loss_dict
        
        # Check that all losses are finite
        for loss_value in loss_dict.values():
            assert torch.isfinite(loss_value)
        
        # Should call discriminator multiple times (real, fake, interpolated)
        assert self.discriminator.call_count >= 2


class TestMultiObjectiveLoss:
    """Test cases for MultiObjectiveLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create component losses
        self.loss1 = DiffusionLoss()
        self.loss2 = FlowLoss()
        
        self.losses = {
            'diffusion': self.loss1,
            'flow': self.loss2
        }
        
        self.weights = {
            'diffusion': 0.7,
            'flow': 0.3
        }
        
        self.loss_fn = MultiObjectiveLoss(
            losses=self.losses,
            weights=self.weights,
            adaptive_weighting=False
        )
        
        # Test data
        self.predictions = {
            'node_scores': torch.randn(10, 8),
            'edge_scores': torch.randn(15, 4)
        }
        
        self.targets = {
            'node_noise': torch.randn(10, 8),
            'edge_noise': torch.randn(15, 4)
        }
        
        self.timesteps = torch.randint(0, 100, (3,))
        self.log_probs = torch.randn(3)
    
    def test_initialization_fixed_weights(self):
        """Test initialization with fixed weights."""
        assert not self.loss_fn.adaptive_weighting
        assert len(self.loss_fn.loss_names) == 2
        assert 'diffusion' in self.loss_fn.loss_names
        assert 'flow' in self.loss_fn.loss_names
    
    def test_initialization_adaptive_weights(self):
        """Test initialization with adaptive weights."""
        loss_fn = MultiObjectiveLoss(
            losses=self.losses,
            adaptive_weighting=True,
            temperature=2.0
        )
        
        assert loss_fn.adaptive_weighting
        assert loss_fn.temperature == 2.0
        assert hasattr(loss_fn, 'log_weights')
        assert loss_fn.log_weights.requires_grad
    
    def test_forward_fixed_weights(self):
        """Test forward pass with fixed weights."""
        # Mock the individual loss functions to return simple dictionaries
        with patch.object(self.loss1, 'forward', return_value={'total_loss': torch.tensor(1.0)}):
            with patch.object(self.loss2, 'forward', return_value={'total_loss': torch.tensor(2.0)}):
                loss_dict = self.loss_fn(
                    self.predictions, self.targets, 
                    timesteps=self.timesteps, log_probs=self.log_probs
                )
        
        assert 'total_loss' in loss_dict
        assert 'weight_diffusion' in loss_dict
        assert 'weight_flow' in loss_dict
        
        # Check that weights are as expected
        assert loss_dict['weight_diffusion'] == 0.7
        assert loss_dict['weight_flow'] == 0.3
        
        # Check total loss calculation
        expected_total = 0.7 * 1.0 + 0.3 * 2.0
        assert torch.allclose(loss_dict['total_loss'], torch.tensor(expected_total))
    
    def test_forward_adaptive_weights(self):
        """Test forward pass with adaptive weights."""
        loss_fn = MultiObjectiveLoss(
            losses=self.losses,
            adaptive_weighting=True,
            temperature=1.0
        )
        
        with patch.object(self.loss1, 'forward', return_value={'total_loss': torch.tensor(1.0)}):
            with patch.object(self.loss2, 'forward', return_value={'total_loss': torch.tensor(2.0)}):
                loss_dict = loss_fn(
                    self.predictions, self.targets,
                    timesteps=self.timesteps, log_probs=self.log_probs
                )
        
        assert 'total_loss' in loss_dict
        assert 'weight_diffusion' in loss_dict
        assert 'weight_flow' in loss_dict
        
        # Adaptive weights should sum to 1
        total_weight = loss_dict['weight_diffusion'] + loss_dict['weight_flow']
        assert torch.allclose(torch.tensor(total_weight), torch.tensor(1.0), atol=1e-6)
    
    def test_gradient_flow_adaptive_weights(self):
        """Test gradient flow through adaptive weights."""
        loss_fn = MultiObjectiveLoss(
            losses=self.losses,
            adaptive_weighting=True
        )
        
        with patch.object(self.loss1, 'forward', return_value={'total_loss': torch.tensor(1.0)}):
            with patch.object(self.loss2, 'forward', return_value={'total_loss': torch.tensor(2.0)}):
                loss_dict = loss_fn(
                    self.predictions, self.targets,
                    timesteps=self.timesteps, log_probs=self.log_probs
                )
        
        loss_dict['total_loss'].backward()
        
        # Check that adaptive weights have gradients
        assert loss_fn.log_weights.grad is not None
        assert torch.isfinite(loss_fn.log_weights.grad).all()


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        mu = torch.randn(5, 10)
        logvar = torch.randn(5, 10)
        
        # Test different reductions
        kl_mean = compute_kl_divergence(mu, logvar, reduction='mean')
        kl_sum = compute_kl_divergence(mu, logvar, reduction='sum')
        kl_none = compute_kl_divergence(mu, logvar, reduction='none')
        
        assert kl_mean.dim() == 0  # Scalar
        assert kl_sum.dim() == 0   # Scalar
        assert kl_none.shape == mu.shape  # Same shape as input
        
        # Check that mean and sum are related correctly
        assert torch.allclose(kl_mean, kl_none.mean())
        assert torch.allclose(kl_sum, kl_none.sum())
        
        # KL divergence should be non-negative
        assert kl_mean.item() >= 0
        assert kl_sum.item() >= 0
        assert (kl_none >= 0).all()
    
    def test_compute_reconstruction_loss(self):
        """Test reconstruction loss computation."""
        predictions = torch.randn(5, 10)
        targets = torch.randn(5, 10)
        
        # Test different loss types
        mse_loss = compute_reconstruction_loss(predictions, targets, 'mse')
        l1_loss = compute_reconstruction_loss(predictions, targets, 'l1')
        
        assert mse_loss.dim() == 0
        assert l1_loss.dim() == 0
        assert mse_loss.item() >= 0
        assert l1_loss.item() >= 0
        
        # Test BCE loss with sigmoid inputs
        predictions_sigmoid = torch.randn(5, 10)
        targets_binary = torch.randint(0, 2, (5, 10)).float()
        
        bce_loss = compute_reconstruction_loss(predictions_sigmoid, targets_binary, 'bce')
        assert bce_loss.dim() == 0
        assert bce_loss.item() >= 0
    
    def test_compute_reconstruction_loss_invalid(self):
        """Test reconstruction loss with invalid loss type."""
        predictions = torch.randn(5, 10)
        targets = torch.randn(5, 10)
        
        with pytest.raises(ValueError, match="Unknown loss type"):
            compute_reconstruction_loss(predictions, targets, 'invalid')
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        batch_size = 8
        num_classes = 5
        
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Test different parameters
        focal = focal_loss(predictions, targets, alpha=1.0, gamma=2.0)
        focal_high_gamma = focal_loss(predictions, targets, alpha=1.0, gamma=5.0)
        
        assert focal.dim() == 0
        assert focal_high_gamma.dim() == 0
        assert focal.item() >= 0
        assert focal_high_gamma.item() >= 0
        
        # Test different reductions
        focal_sum = focal_loss(predictions, targets, reduction='sum')
        focal_none = focal_loss(predictions, targets, reduction='none')
        
        assert focal_sum.dim() == 0
        assert focal_none.shape == (batch_size,)
        assert torch.allclose(focal_sum, focal_none.sum())
    
    def test_perfect_predictions(self):
        """Test loss functions with perfect predictions."""
        # Perfect reconstruction
        targets = torch.randn(5, 10)
        predictions = targets.clone()
        
        mse_loss = compute_reconstruction_loss(predictions, targets, 'mse')
        l1_loss = compute_reconstruction_loss(predictions, targets, 'l1')
        
        assert mse_loss.item() < 1e-6  # Should be very close to zero
        assert l1_loss.item() < 1e-6
        
        # Perfect KL divergence (standard normal)
        mu = torch.zeros(5, 10)
        logvar = torch.zeros(5, 10)
        
        kl = compute_kl_divergence(mu, logvar)
        assert kl.item() < 1e-6  # Should be very close to zero


if __name__ == "__main__":
    pytest.main([__file__, "-v"])