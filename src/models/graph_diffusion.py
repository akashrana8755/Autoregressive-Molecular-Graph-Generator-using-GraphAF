"""
GraphDiffusion model implementation for molecular generation.

This module implements a graph diffusion model based on stochastic differential
equations (SDEs) for generating molecular graphs. The model learns to reverse
a diffusion process that gradually adds noise to molecular graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import math

from .base_model import BaseGenerativeModel


class GraphDiffusion(BaseGenerativeModel):
    """
    Graph diffusion model for molecular generation using SDE-based diffusion.
    
    This model implements a continuous-time diffusion process on molecular graphs,
    learning to reverse the forward diffusion process to generate new molecules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphDiffusion model.
        
        Args:
            config: Configuration dictionary containing:
                - node_dim: Dimension of node features
                - edge_dim: Dimension of edge features  
                - hidden_dim: Hidden dimension for GNN layers
                - num_layers: Number of GNN layers
                - dropout: Dropout rate
                - max_nodes: Maximum number of nodes in generated graphs
                - beta_schedule: Noise schedule type ('linear', 'cosine')
                - num_timesteps: Number of diffusion timesteps
        """
        super().__init__(config)
        
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config.get('dropout', 0.1)
        self.max_nodes = config.get('max_nodes', 50)
        self.num_timesteps = config.get('num_timesteps', 1000)
        
        # Initialize noise schedule
        self.beta_schedule = config.get('beta_schedule', 'linear')
        self.register_buffer('betas', self._get_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Time embedding
        self.time_embed_dim = self.hidden_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Node feature processing
        self.node_encoder = nn.Linear(self.node_dim, self.hidden_dim)
        self.node_decoder = nn.Linear(self.hidden_dim, self.node_dim)
        
        # Edge feature processing  
        self.edge_encoder = nn.Linear(self.edge_dim, self.hidden_dim)
        self.edge_decoder = nn.Linear(self.hidden_dim, self.edge_dim)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=4,
                    dropout=self.dropout,
                    concat=False,
                    edge_dim=self.hidden_dim
                )
            )
        
        # Output layers for score prediction
        self.node_score_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.node_dim)
        )
        
        self.edge_score_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.time_embed_dim, self.hidden_dim),
            nn.SiLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.edge_dim)
        )
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """
        Get the noise schedule for diffusion.
        
        Returns:
            Beta values for each timestep
        """
        if self.beta_schedule == 'linear':
            return torch.linspace(1e-4, 0.02, self.num_timesteps)
        elif self.beta_schedule == 'cosine':
            timesteps = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((timesteps / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def forward(self, batch: Batch, t: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict noise/score at time t.
        
        Args:
            batch: Batch of molecular graphs
            t: Timestep tensor [batch_size]
            
        Returns:
            Dictionary containing predicted node and edge scores
        """
        # Encode time
        t_embed = self.time_embedding(t.float().unsqueeze(-1))  # [batch_size, time_embed_dim]
        
        # Encode node and edge features
        node_h = self.node_encoder(batch.x)  # [num_nodes, hidden_dim]
        edge_h = self.edge_encoder(batch.edge_attr)  # [num_edges, hidden_dim]
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_h_new = gnn_layer(node_h, batch.edge_index, edge_h)
            node_h = node_h + node_h_new  # Residual connection
            node_h = F.dropout(node_h, p=self.dropout, training=self.training)
        
        # Expand time embedding to match node dimensions
        batch_size = t.size(0)
        num_nodes_per_graph = torch.bincount(batch.batch)
        t_embed_expanded = torch.repeat_interleave(t_embed, num_nodes_per_graph, dim=0)
        
        # Predict node scores
        node_input = torch.cat([node_h, t_embed_expanded], dim=-1)
        node_scores = self.node_score_net(node_input)
        
        # Predict edge scores
        edge_start_nodes = node_h[batch.edge_index[0]]  # [num_edges, hidden_dim]
        edge_end_nodes = node_h[batch.edge_index[1]]    # [num_edges, hidden_dim]
        
        # Get time embedding for edges
        edge_batch = batch.batch[batch.edge_index[0]]  # Batch assignment for each edge
        t_embed_edges = t_embed[edge_batch]  # [num_edges, time_embed_dim]
        
        edge_input = torch.cat([edge_start_nodes, edge_end_nodes, t_embed_edges], dim=-1)
        edge_scores = self.edge_score_net(edge_input)
        
        return {
            'node_scores': node_scores,
            'edge_scores': edge_scores
        }
    
    def training_step(self, batch: Batch, **kwargs) -> torch.Tensor:
        """
        Compute diffusion loss for training.
        
        Args:
            batch: Batch of molecular graphs
            
        Returns:
            Diffusion loss tensor
        """
        batch_size = batch.num_graphs
        device = batch.x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        node_noise = torch.randn_like(batch.x)
        edge_noise = torch.randn_like(batch.edge_attr)
        
        # Add noise to the data (forward diffusion)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Expand to match node/edge dimensions
        num_nodes_per_graph = torch.bincount(batch.batch)
        sqrt_alphas_cumprod_nodes = torch.repeat_interleave(
            sqrt_alphas_cumprod_t, num_nodes_per_graph, dim=0
        ).unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_nodes = torch.repeat_interleave(
            sqrt_one_minus_alphas_cumprod_t, num_nodes_per_graph, dim=0
        ).unsqueeze(-1)
        
        # For edges
        edge_batch = batch.batch[batch.edge_index[0]]
        sqrt_alphas_cumprod_edges = sqrt_alphas_cumprod_t[edge_batch].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_edges = sqrt_one_minus_alphas_cumprod_t[edge_batch].unsqueeze(-1)
        
        # Forward diffusion: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        noisy_nodes = sqrt_alphas_cumprod_nodes * batch.x + sqrt_one_minus_alphas_cumprod_nodes * node_noise
        noisy_edges = sqrt_alphas_cumprod_edges * batch.edge_attr + sqrt_one_minus_alphas_cumprod_edges * edge_noise
        
        # Create noisy batch
        noisy_batch = batch.clone()
        noisy_batch.x = noisy_nodes
        noisy_batch.edge_attr = noisy_edges
        
        # Predict the noise
        predicted = self.forward(noisy_batch, t)
        
        # Compute loss (predict the noise)
        node_loss = F.mse_loss(predicted['node_scores'], node_noise)
        edge_loss = F.mse_loss(predicted['edge_scores'], edge_noise)
        
        total_loss = node_loss + edge_loss
        
        return total_loss
    
    def sample(self, num_samples: int, **kwargs) -> List[Data]:
        """
        Generate molecular graphs using DDPM sampling.
        
        Args:
            num_samples: Number of molecules to generate
            **kwargs: Additional sampling parameters
                - max_nodes: Maximum nodes per molecule
                - guidance_scale: Classifier-free guidance scale
                
        Returns:
            List of generated molecular graphs
        """
        device = self.get_device()
        max_nodes = kwargs.get('max_nodes', self.max_nodes)
        
        # Start from pure noise
        # For simplicity, generate graphs with fixed number of nodes
        # In practice, you might want to sample the number of nodes
        nodes_per_sample = max_nodes
        total_nodes = num_samples * nodes_per_sample
        
        # Initialize with noise
        x_t = torch.randn(total_nodes, self.node_dim, device=device)
        
        # Create edge indices for fully connected graphs (will be pruned later)
        edge_indices = []
        edge_attrs = []
        batch_indices = []
        
        for i in range(num_samples):
            start_idx = i * nodes_per_sample
            end_idx = (i + 1) * nodes_per_sample
            
            # Create fully connected graph
            nodes = torch.arange(start_idx, end_idx, device=device)
            edge_index = torch.combinations(nodes, r=2, with_replacement=False).t()
            # Add reverse edges
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            edge_indices.append(edge_index)
            edge_attrs.append(torch.randn(edge_index.size(1), self.edge_dim, device=device))
            batch_indices.extend([i] * nodes_per_sample)
        
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        batch = torch.tensor(batch_indices, device=device)
        
        # Reverse diffusion process
        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                
                # Create batch
                current_batch = Batch(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch
                )
                
                # Predict noise
                predicted = self.forward(current_batch, t_tensor)
                predicted_noise_nodes = predicted['node_scores']
                predicted_noise_edges = predicted['edge_scores']
                
                # Compute denoising step
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                # Denoising formula
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
                
                x_t = coeff1 * (x_t - coeff2 * predicted_noise_nodes)
                edge_attr = coeff1 * (edge_attr - coeff2 * predicted_noise_edges)
                
                # Add noise (except for the last step)
                if t > 0:
                    noise = torch.randn_like(x_t)
                    edge_noise = torch.randn_like(edge_attr)
                    sigma_t = torch.sqrt(beta_t)
                    x_t = x_t + sigma_t * noise
                    edge_attr = edge_attr + sigma_t * edge_noise
        
        # Convert to list of Data objects
        generated_graphs = []
        for i in range(num_samples):
            start_idx = i * nodes_per_sample
            end_idx = (i + 1) * nodes_per_sample
            
            # Extract nodes for this graph
            node_features = x_t[start_idx:end_idx]
            
            # Extract edges for this graph
            graph_edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            graph_edge_index = edge_index[:, graph_edge_mask] - start_idx
            graph_edge_attr = edge_attr[graph_edge_mask]
            
            # Create Data object
            graph = Data(
                x=node_features,
                edge_index=graph_edge_index,
                edge_attr=graph_edge_attr
            )
            generated_graphs.append(graph)
        
        return generated_graphs