"""
Loss functions for molecular generation models.

This module implements specialized loss functions for GraphDiffusion and GraphAF
models, including diffusion losses, flow losses, and auxiliary losses for
improved training stability and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, Optional, Tuple, Any
import numpy as np
import math


class DiffusionLoss(nn.Module):
    """
    Loss function for GraphDiffusion models.
    
    Implements the denoising score matching loss for training diffusion models
    on molecular graphs, with support for different noise schedules and
    weighting strategies.
    """
    
    def __init__(self, 
                 loss_type: str = 'mse',
                 node_weight: float = 1.0,
                 edge_weight: float = 1.0,
                 timestep_weighting: str = 'uniform',
                 snr_gamma: float = 5.0):
        """
        Initialize the diffusion loss.
        
        Args:
            loss_type: Type of loss ('mse', 'l1', 'huber')
            node_weight: Weight for node prediction loss
            edge_weight: Weight for edge prediction loss
            timestep_weighting: Timestep weighting strategy ('uniform', 'snr', 'truncated_snr')
            snr_gamma: Gamma parameter for SNR weighting
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.timestep_weighting = timestep_weighting
        self.snr_gamma = snr_gamma
        
        # Base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                timesteps: torch.Tensor,
                alphas_cumprod: Optional[torch.Tensor] = None,
                batch: Optional[Batch] = None) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion loss.
        
        Args:
            predictions: Dictionary with 'node_scores' and 'edge_scores'
            targets: Dictionary with 'node_noise' and 'edge_noise'
            timesteps: Timestep tensor [batch_size]
            alphas_cumprod: Cumulative alpha values for SNR weighting
            batch: Graph batch for additional context
            
        Returns:
            Dictionary containing loss components and total loss
        """
        # Node loss
        node_loss = self.base_loss(predictions['node_scores'], targets['node_noise'])
        node_loss = node_loss.mean()
        
        # Edge loss
        edge_loss = self.base_loss(predictions['edge_scores'], targets['edge_noise'])
        edge_loss = edge_loss.mean()
        
        # Apply timestep weighting
        if self.timestep_weighting != 'uniform' and alphas_cumprod is not None:
            weights = self._compute_timestep_weights(timesteps, alphas_cumprod)
            
            # Apply weights to losses
            if batch is not None:
                # Expand weights to match node/edge dimensions
                num_nodes_per_graph = torch.bincount(batch.batch)
                node_weights = torch.repeat_interleave(weights, num_nodes_per_graph)
                
                edge_batch = batch.batch[batch.edge_index[0]]
                edge_weights = weights[edge_batch]
                
                # Recompute weighted losses
                node_loss_per_element = self.base_loss(predictions['node_scores'], targets['node_noise'])
                edge_loss_per_element = self.base_loss(predictions['edge_scores'], targets['edge_noise'])
                
                node_loss = (node_loss_per_element * node_weights.unsqueeze(-1)).mean()
                edge_loss = (edge_loss_per_element * edge_weights.unsqueeze(-1)).mean()
        
        # Combine losses
        total_loss = self.node_weight * node_loss + self.edge_weight * edge_loss
        
        return {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss
        }
        
    def _compute_timestep_weights(self, 
                                 timesteps: torch.Tensor, 
                                 alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """Compute timestep-dependent loss weights."""
        if self.timestep_weighting == 'snr':
            # Signal-to-noise ratio weighting
            snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            weights = torch.minimum(snr, torch.tensor(self.snr_gamma, device=snr.device))
            
        elif self.timestep_weighting == 'truncated_snr':
            # Truncated SNR weighting (clip very high SNR values)
            snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            weights = torch.clamp(snr, max=self.snr_gamma)
            
        else:
            # Uniform weighting
            weights = torch.ones_like(timesteps, dtype=torch.float)
            
        return weights


class FlowLoss(nn.Module):
    """
    Loss function for GraphAF (autoregressive flow) models.
    
    Implements the negative log-likelihood loss for training normalizing flow
    models on molecular graphs, with support for regularization and
    stability improvements.
    """
    
    def __init__(self,
                 regularization_weight: float = 0.0,
                 entropy_regularization: float = 0.0,
                 gradient_penalty_weight: float = 0.0):
        """
        Initialize the flow loss.
        
        Args:
            regularization_weight: Weight for L2 regularization
            entropy_regularization: Weight for entropy regularization
            gradient_penalty_weight: Weight for gradient penalty
        """
        super().__init__()
        
        self.regularization_weight = regularization_weight
        self.entropy_regularization = entropy_regularization
        self.gradient_penalty_weight = gradient_penalty_weight
        
    def forward(self, 
                log_probs: torch.Tensor,
                model: Optional[nn.Module] = None,
                batch: Optional[Batch] = None) -> Dict[str, torch.Tensor]:
        """
        Compute flow loss (negative log-likelihood).
        
        Args:
            log_probs: Log probabilities from the flow model [batch_size]
            model: Model for regularization (optional)
            batch: Graph batch for additional context (optional)
            
        Returns:
            Dictionary containing loss components and total loss
        """
        # Negative log-likelihood
        nll_loss = -log_probs.mean()
        
        total_loss = nll_loss
        loss_dict = {
            'nll_loss': nll_loss,
            'total_loss': total_loss
        }
        
        # L2 regularization
        if self.regularization_weight > 0.0 and model is not None:
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            l2_loss = self.regularization_weight * l2_reg
            total_loss += l2_loss
            loss_dict['l2_loss'] = l2_loss
            
        # Entropy regularization (encourage diversity)
        if self.entropy_regularization > 0.0:
            # Approximate entropy using log probability variance
            entropy_loss = -self.entropy_regularization * log_probs.var()
            total_loss += entropy_loss
            loss_dict['entropy_loss'] = entropy_loss
            
        # Gradient penalty for stability
        if self.gradient_penalty_weight > 0.0 and model is not None and batch is not None:
            grad_penalty = self._compute_gradient_penalty(model, batch)
            total_loss += self.gradient_penalty_weight * grad_penalty
            loss_dict['gradient_penalty'] = grad_penalty
            
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
        
    def _compute_gradient_penalty(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Compute gradient penalty for training stability."""
        # Create interpolated samples
        alpha = torch.rand(batch.num_graphs, 1, device=batch.x.device)
        
        # Expand alpha to match node dimensions
        num_nodes_per_graph = torch.bincount(batch.batch)
        alpha_nodes = torch.repeat_interleave(alpha, num_nodes_per_graph, dim=0)
        
        # Create noise
        noise_x = torch.randn_like(batch.x)
        
        # Interpolate
        interpolated_x = alpha_nodes * batch.x + (1 - alpha_nodes) * noise_x
        interpolated_x.requires_grad_(True)
        
        # Create interpolated batch
        interpolated_batch = Batch(
            x=interpolated_x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Forward pass
        outputs = model(interpolated_batch)
        log_probs = outputs.get('log_prob', torch.zeros(batch.num_graphs, device=batch.x.device))
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=log_probs.sum(),
            inputs=interpolated_x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty


class PropertyConditionedLoss(nn.Module):
    """
    Loss function for property-conditioned molecular generation.
    
    Combines generation loss with property prediction loss for models
    that generate molecules with specific target properties.
    """
    
    def __init__(self,
                 generation_loss: nn.Module,
                 property_loss_weight: float = 0.1,
                 property_loss_type: str = 'mse'):
        """
        Initialize property-conditioned loss.
        
        Args:
            generation_loss: Base generation loss (DiffusionLoss or FlowLoss)
            property_loss_weight: Weight for property prediction loss
            property_loss_type: Type of property loss ('mse', 'l1', 'huber')
        """
        super().__init__()
        
        self.generation_loss = generation_loss
        self.property_loss_weight = property_loss_weight
        
        if property_loss_type == 'mse':
            self.property_loss_fn = nn.MSELoss()
        elif property_loss_type == 'l1':
            self.property_loss_fn = nn.L1Loss()
        elif property_loss_type == 'huber':
            self.property_loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown property loss type: {property_loss_type}")
            
    def forward(self,
                generation_outputs: Dict[str, torch.Tensor],
                generation_targets: Dict[str, torch.Tensor],
                property_predictions: Optional[torch.Tensor] = None,
                property_targets: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute combined generation and property loss.
        
        Args:
            generation_outputs: Outputs from generation model
            generation_targets: Target values for generation
            property_predictions: Predicted properties [batch_size, num_properties]
            property_targets: Target properties [batch_size, num_properties]
            **kwargs: Additional arguments for generation loss
            
        Returns:
            Dictionary containing all loss components
        """
        # Compute generation loss
        gen_loss_dict = self.generation_loss(generation_outputs, generation_targets, **kwargs)
        
        total_loss = gen_loss_dict['total_loss']
        loss_dict = {f'gen_{k}': v for k, v in gen_loss_dict.items()}
        
        # Compute property loss if available
        if property_predictions is not None and property_targets is not None:
            # Handle missing property values (NaN)
            valid_mask = ~torch.isnan(property_targets)
            
            if valid_mask.any():
                valid_predictions = property_predictions[valid_mask]
                valid_targets = property_targets[valid_mask]
                
                property_loss = self.property_loss_fn(valid_predictions, valid_targets)
                weighted_property_loss = self.property_loss_weight * property_loss
                
                total_loss += weighted_property_loss
                
                loss_dict.update({
                    'property_loss': property_loss,
                    'weighted_property_loss': weighted_property_loss
                })
                
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for improved molecular generation quality.
    
    Implements adversarial training between generator and discriminator
    to improve the quality and diversity of generated molecules.
    """
    
    def __init__(self,
                 discriminator: nn.Module,
                 adversarial_weight: float = 0.1,
                 gradient_penalty_weight: float = 10.0):
        """
        Initialize adversarial loss.
        
        Args:
            discriminator: Discriminator network
            adversarial_weight: Weight for adversarial loss
            gradient_penalty_weight: Weight for gradient penalty (WGAN-GP)
        """
        super().__init__()
        
        self.discriminator = discriminator
        self.adversarial_weight = adversarial_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        
    def generator_loss(self, generated_batch: Batch) -> torch.Tensor:
        """Compute generator adversarial loss."""
        fake_scores = self.discriminator(generated_batch)
        # Generator wants discriminator to think generated samples are real
        gen_loss = -fake_scores.mean()
        return self.adversarial_weight * gen_loss
        
    def discriminator_loss(self, 
                          real_batch: Batch, 
                          fake_batch: Batch) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss with gradient penalty."""
        # Real samples
        real_scores = self.discriminator(real_batch)
        
        # Fake samples
        fake_scores = self.discriminator(fake_batch)
        
        # Wasserstein loss
        d_loss = fake_scores.mean() - real_scores.mean()
        
        # Gradient penalty
        gp = self._gradient_penalty(real_batch, fake_batch)
        
        total_d_loss = d_loss + self.gradient_penalty_weight * gp
        
        return {
            'discriminator_loss': total_d_loss,
            'wasserstein_distance': -d_loss,
            'gradient_penalty': gp
        }
        
    def _gradient_penalty(self, real_batch: Batch, fake_batch: Batch) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_batch.num_graphs
        device = real_batch.x.device
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1, device=device)
        
        # Expand to node level
        num_nodes_per_graph_real = torch.bincount(real_batch.batch)
        num_nodes_per_graph_fake = torch.bincount(fake_batch.batch)
        
        # Ensure same number of nodes (pad or truncate if necessary)
        min_nodes = torch.minimum(num_nodes_per_graph_real, num_nodes_per_graph_fake)
        
        alpha_nodes = torch.repeat_interleave(alpha, min_nodes, dim=0)
        
        # Interpolate node features
        real_x_truncated = real_batch.x[:alpha_nodes.size(0)]
        fake_x_truncated = fake_batch.x[:alpha_nodes.size(0)]
        
        interpolated_x = alpha_nodes * real_x_truncated + (1 - alpha_nodes) * fake_x_truncated
        interpolated_x.requires_grad_(True)
        
        # Create interpolated batch (simplified - use real batch structure)
        interpolated_batch = Batch(
            x=interpolated_x,
            edge_index=real_batch.edge_index,
            edge_attr=real_batch.edge_attr,
            batch=real_batch.batch[:alpha_nodes.size(0)]
        )
        
        # Discriminator output
        d_interpolated = self.discriminator(interpolated_batch)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated.sum(),
            inputs=interpolated_x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss for balancing multiple training objectives.
    
    Combines multiple loss functions with adaptive or fixed weighting
    to optimize for multiple objectives simultaneously.
    """
    
    def __init__(self,
                 losses: Dict[str, nn.Module],
                 weights: Optional[Dict[str, float]] = None,
                 adaptive_weighting: bool = False,
                 temperature: float = 1.0):
        """
        Initialize multi-objective loss.
        
        Args:
            losses: Dictionary of loss functions
            weights: Fixed weights for each loss (if not adaptive)
            adaptive_weighting: Whether to use adaptive loss weighting
            temperature: Temperature for adaptive weighting
        """
        super().__init__()
        
        self.losses = nn.ModuleDict(losses)
        self.adaptive_weighting = adaptive_weighting
        self.temperature = temperature
        
        if weights is None:
            weights = {name: 1.0 for name in losses.keys()}
        self.register_buffer('weights', torch.tensor(list(weights.values())))
        self.loss_names = list(losses.keys())
        
        if adaptive_weighting:
            # Learnable weights for adaptive balancing
            self.log_weights = nn.Parameter(torch.zeros(len(losses)))
            
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.
        
        Args:
            *args, **kwargs: Arguments passed to individual loss functions
            
        Returns:
            Dictionary containing all loss components
        """
        individual_losses = {}
        loss_values = []
        
        # Compute individual losses
        for i, (name, loss_fn) in enumerate(self.losses.items()):
            loss_output = loss_fn(*args, **kwargs)
            
            if isinstance(loss_output, dict):
                individual_losses.update({f'{name}_{k}': v for k, v in loss_output.items()})
                loss_value = loss_output.get('total_loss', loss_output.get('loss', 0.0))
            else:
                individual_losses[name] = loss_output
                loss_value = loss_output
                
            loss_values.append(loss_value)
            
        loss_tensor = torch.stack(loss_values)
        
        # Compute weights
        if self.adaptive_weighting:
            # Softmax weighting with temperature
            weights = F.softmax(self.log_weights / self.temperature, dim=0)
        else:
            weights = self.weights
            
        # Weighted combination
        total_loss = (weights * loss_tensor).sum()
        
        # Add weight information
        weight_dict = {f'weight_{name}': weights[i].item() 
                      for i, name in enumerate(self.loss_names)}
        
        return {
            **individual_losses,
            **weight_dict,
            'total_loss': total_loss
        }


# Utility functions for loss computation

def compute_kl_divergence(mu: torch.Tensor, 
                         logvar: torch.Tensor, 
                         reduction: str = 'mean') -> torch.Tensor:
    """
    Compute KL divergence for VAE-style models.
    
    Args:
        mu: Mean of the latent distribution
        logvar: Log variance of the latent distribution
        reduction: Reduction method ('mean', 'sum', 'none')
        
    Returns:
        KL divergence loss
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


def compute_reconstruction_loss(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               loss_type: str = 'mse',
                               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute reconstruction loss for autoencoder-style models.
    
    Args:
        predictions: Predicted values
        targets: Target values
        loss_type: Type of loss ('mse', 'bce', 'l1')
        reduction: Reduction method
        
    Returns:
        Reconstruction loss
    """
    if loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction=reduction)
    elif loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
        
    return loss_fn(predictions, targets)


def focal_loss(predictions: torch.Tensor,
               targets: torch.Tensor,
               alpha: float = 1.0,
               gamma: float = 2.0,
               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute focal loss for handling class imbalance.
    
    Args:
        predictions: Predicted logits
        targets: Target labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: Reduction method
        
    Returns:
        Focal loss
    """
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss