"""
Conditional molecular generation with property conditioning.

This module provides enhanced versions of generative models that can
condition generation on target molecular properties, enabling
property-guided molecular design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from .graph_diffusion import GraphDiffusion
from .graph_af import GraphAF
from .property_predictor import PropertyPredictor, MultiTaskPropertyPredictor


class ConditionalGraphDiffusion(GraphDiffusion):
    """
    Property-conditioned graph diffusion model.
    
    This model extends GraphDiffusion to accept property conditioning,
    allowing generation of molecules with target properties.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize conditional graph diffusion model.
        
        Args:
            config: Configuration dictionary with additional keys:
                - property_dim: Dimension of property conditioning vector
                - property_names: List of property names for conditioning
                - conditioning_type: Type of conditioning ('concat', 'cross_attention', 'film')
                - property_dropout: Dropout rate for property conditioning during training
        """
        super().__init__(config)
        
        self.property_dim = config.get('property_dim', 0)
        self.property_names = config.get('property_names', [])
        self.conditioning_type = config.get('conditioning_type', 'concat')
        self.property_dropout = config.get('property_dropout', 0.1)
        
        if self.property_dim > 0:
            self._setup_property_conditioning()
            
    def _setup_property_conditioning(self):
        """Setup property conditioning components."""
        # Property embedding
        self.property_embedding = nn.Sequential(
            nn.Linear(self.property_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        if self.conditioning_type == 'concat':
            # Modify score networks to accept property conditioning
            self.node_score_net = nn.Sequential(
                nn.Linear(self.hidden_dim + self.time_embed_dim + self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.node_dim)
            )
            
            self.edge_score_net = nn.Sequential(
                nn.Linear(self.hidden_dim * 2 + self.time_embed_dim + self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.edge_dim)
            )
            
        elif self.conditioning_type == 'cross_attention':
            # Cross-attention layers for property conditioning
            self.property_cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True
            )
            
        elif self.conditioning_type == 'film':
            # FiLM (Feature-wise Linear Modulation) conditioning
            self.film_scale = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.film_shift = nn.Linear(self.hidden_dim, self.hidden_dim)
            
    def forward(self, 
                batch: Batch, 
                t: torch.Tensor, 
                properties: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with property conditioning.
        
        Args:
            batch: Batch of molecular graphs
            t: Timestep tensor
            properties: Property conditioning tensor [batch_size, property_dim]
            
        Returns:
            Dictionary containing predicted scores
        """
        # Encode time
        t_embed = self.time_embedding(t.float().unsqueeze(-1))
        
        # Encode property conditioning
        if properties is not None and self.property_dim > 0:
            # Apply property dropout during training for classifier-free guidance
            if self.training and self.property_dropout > 0:
                mask = torch.rand(properties.size(0), device=properties.device) > self.property_dropout
                properties = properties * mask.unsqueeze(-1)
                
            prop_embed = self.property_embedding(properties)
        else:
            prop_embed = torch.zeros(t.size(0), self.hidden_dim, device=self.get_device())
            
        # Encode node and edge features
        node_h = self.node_encoder(batch.x)
        edge_h = self.edge_encoder(batch.edge_attr)
        
        # Apply GNN layers with property conditioning
        for gnn_layer in self.gnn_layers:
            node_h_new = gnn_layer(node_h, batch.edge_index, edge_h)
            
            # Apply property conditioning
            if self.conditioning_type == 'film' and self.property_dim > 0:
                # Expand property embedding to node level
                batch_size = t.size(0)
                num_nodes_per_graph = torch.bincount(batch.batch)
                prop_embed_expanded = torch.repeat_interleave(prop_embed, num_nodes_per_graph, dim=0)
                
                # FiLM conditioning
                scale = self.film_scale(prop_embed_expanded)
                shift = self.film_shift(prop_embed_expanded)
                node_h_new = node_h_new * (1 + scale) + shift
                
            node_h = node_h + node_h_new  # Residual connection
            node_h = F.dropout(node_h, p=self.dropout, training=self.training)
            
        # Expand embeddings to match node/edge dimensions
        batch_size = t.size(0)
        num_nodes_per_graph = torch.bincount(batch.batch)
        t_embed_expanded = torch.repeat_interleave(t_embed, num_nodes_per_graph, dim=0)
        prop_embed_expanded = torch.repeat_interleave(prop_embed, num_nodes_per_graph, dim=0)
        
        # Predict node scores
        if self.conditioning_type == 'concat':
            node_input = torch.cat([node_h, t_embed_expanded, prop_embed_expanded], dim=-1)
        else:
            node_input = torch.cat([node_h, t_embed_expanded], dim=-1)
            
        node_scores = self.node_score_net(node_input)
        
        # Predict edge scores
        edge_start_nodes = node_h[batch.edge_index[0]]
        edge_end_nodes = node_h[batch.edge_index[1]]
        
        # Get embeddings for edges
        edge_batch = batch.batch[batch.edge_index[0]]
        t_embed_edges = t_embed[edge_batch]
        prop_embed_edges = prop_embed[edge_batch]
        
        if self.conditioning_type == 'concat':
            edge_input = torch.cat([edge_start_nodes, edge_end_nodes, t_embed_edges, prop_embed_edges], dim=-1)
        else:
            edge_input = torch.cat([edge_start_nodes, edge_end_nodes, t_embed_edges], dim=-1)
            
        edge_scores = self.edge_score_net(edge_input)
        
        return {
            'node_scores': node_scores,
            'edge_scores': edge_scores
        }
        
    def sample(self, 
               num_samples: int, 
               properties: Optional[torch.Tensor] = None,
               guidance_scale: float = 1.0,
               **kwargs) -> List[Data]:
        """
        Generate molecules with property conditioning.
        
        Args:
            num_samples: Number of molecules to generate
            properties: Target properties [num_samples, property_dim]
            guidance_scale: Classifier-free guidance scale
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated molecular graphs
        """
        device = self.get_device()
        max_nodes = kwargs.get('max_nodes', self.max_nodes)
        
        # Prepare property conditioning
        if properties is None and self.property_dim > 0:
            # Sample random properties if not provided
            properties = torch.randn(num_samples, self.property_dim, device=device)
        elif properties is not None:
            properties = properties.to(device)
            
        # Start from pure noise
        nodes_per_sample = max_nodes
        total_nodes = num_samples * nodes_per_sample
        
        x_t = torch.randn(total_nodes, self.node_dim, device=device)
        
        # Create edge structure (simplified - fully connected)
        edge_indices = []
        edge_attrs = []
        batch_indices = []
        
        for i in range(num_samples):
            start_idx = i * nodes_per_sample
            end_idx = (i + 1) * nodes_per_sample
            
            nodes = torch.arange(start_idx, end_idx, device=device)
            edge_index = torch.combinations(nodes, r=2, with_replacement=False).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            edge_indices.append(edge_index)
            edge_attrs.append(torch.randn(edge_index.size(1), self.edge_dim, device=device))
            batch_indices.extend([i] * nodes_per_sample)
            
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        batch = torch.tensor(batch_indices, device=device)
        
        # Reverse diffusion with property conditioning
        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                
                current_batch = Batch(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch
                )
                
                if guidance_scale > 1.0 and properties is not None:
                    # Classifier-free guidance
                    # Conditional prediction
                    predicted_cond = self.forward(current_batch, t_tensor, properties)
                    
                    # Unconditional prediction (with null properties)
                    null_properties = torch.zeros_like(properties)
                    predicted_uncond = self.forward(current_batch, t_tensor, null_properties)
                    
                    # Guided prediction
                    predicted_noise_nodes = (
                        predicted_uncond['node_scores'] + 
                        guidance_scale * (predicted_cond['node_scores'] - predicted_uncond['node_scores'])
                    )
                    predicted_noise_edges = (
                        predicted_uncond['edge_scores'] + 
                        guidance_scale * (predicted_cond['edge_scores'] - predicted_uncond['edge_scores'])
                    )
                else:
                    # Standard conditional prediction
                    predicted = self.forward(current_batch, t_tensor, properties)
                    predicted_noise_nodes = predicted['node_scores']
                    predicted_noise_edges = predicted['edge_scores']
                
                # Denoising step
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
                
                x_t = coeff1 * (x_t - coeff2 * predicted_noise_nodes)
                edge_attr = coeff1 * (edge_attr - coeff2 * predicted_noise_edges)
                
                # Add noise (except for last step)
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
            
            node_features = x_t[start_idx:end_idx]
            
            graph_edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            graph_edge_index = edge_index[:, graph_edge_mask] - start_idx
            graph_edge_attr = edge_attr[graph_edge_mask]
            
            graph = Data(
                x=node_features,
                edge_index=graph_edge_index,
                edge_attr=graph_edge_attr
            )
            generated_graphs.append(graph)
            
        return generated_graphs


class ConditionalGraphAF(GraphAF):
    """
    Property-conditioned graph autoregressive flow model.
    
    This model extends GraphAF to accept property conditioning for
    targeted molecular generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize conditional GraphAF model.
        
        Args:
            config: Configuration dictionary with property conditioning parameters
        """
        super().__init__(config)
        
        self.property_dim = config.get('property_dim', 0)
        self.property_names = config.get('property_names', [])
        self.property_dropout = config.get('property_dropout', 0.1)
        
        if self.property_dim > 0:
            self._setup_property_conditioning()
            
    def _setup_property_conditioning(self):
        """Setup property conditioning components."""
        # Property embedding
        self.property_embedding = nn.Sequential(
            nn.Linear(self.property_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Modify context aggregator to include properties
        self.context_aggregator = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),  # graph + property
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Update predictors to use property-conditioned context
        self.node_type_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_node_types)
        )
        
        self.stop_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
    def encode_graph_context(self, 
                           batch: Batch, 
                           properties: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode graph context with property conditioning.
        
        Args:
            batch: Current graph batch
            properties: Property conditioning tensor
            
        Returns:
            Property-conditioned context vector
        """
        # Get base graph context
        if batch.x.size(0) == 0:
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            graph_context = torch.zeros(batch_size, self.hidden_dim, device=self.get_device())
        else:
            node_h = self.node_encoder(batch.x)
            
            if batch.edge_attr.size(0) > 0:
                edge_h = self.edge_encoder(batch.edge_attr)
                
                for gnn_layer in self.gnn_layers:
                    node_h_new = gnn_layer(node_h, batch.edge_index, edge_h)
                    node_h = node_h + node_h_new
                    node_h = F.dropout(node_h, p=self.dropout, training=self.training)
                    
            from torch_geometric.nn import global_mean_pool
            graph_context = global_mean_pool(node_h, batch.batch)
            
        # Add property conditioning
        if properties is not None and self.property_dim > 0:
            # Apply property dropout during training
            if self.training and self.property_dropout > 0:
                mask = torch.rand(properties.size(0), device=properties.device) > self.property_dropout
                properties = properties * mask.unsqueeze(-1)
                
            prop_embed = self.property_embedding(properties)
            
            # Combine graph and property context
            combined_context = torch.cat([graph_context, prop_embed], dim=-1)
            context = self.context_aggregator(combined_context)
        else:
            # Use zero property embedding
            prop_embed = torch.zeros(graph_context.size(0), self.hidden_dim, device=self.get_device())
            combined_context = torch.cat([graph_context, prop_embed], dim=-1)
            context = self.context_aggregator(combined_context)
            
        return context
        
    def sample(self, 
               num_samples: int, 
               properties: Optional[torch.Tensor] = None,
               **kwargs) -> List[Data]:
        """
        Generate molecules with property conditioning.
        
        Args:
            num_samples: Number of molecules to generate
            properties: Target properties [num_samples, property_dim]
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated molecular graphs
        """
        device = self.get_device()
        max_nodes = kwargs.get('max_nodes', self.max_nodes)
        temperature = kwargs.get('temperature', 1.0)
        
        # Prepare property conditioning
        if properties is None and self.property_dim > 0:
            properties = torch.randn(num_samples, self.property_dim, device=device)
        elif properties is not None:
            properties = properties.to(device)
            
        generated_graphs = []
        
        self.eval()
        with torch.no_grad():
            for i in range(num_samples):
                # Get property for this sample
                sample_properties = properties[i:i+1] if properties is not None else None
                
                graph = self._sample_single_graph_conditional(
                    max_nodes, temperature, sample_properties
                )
                generated_graphs.append(graph)
                
        return generated_graphs
        
    def _sample_single_graph_conditional(self, 
                                       max_nodes: int, 
                                       temperature: float,
                                       properties: Optional[torch.Tensor]) -> Data:
        """Sample a single graph with property conditioning."""
        device = self.get_device()
        
        nodes = []
        edges = []
        edge_attrs = []
        
        for step in range(max_nodes):
            # Create current graph context
            if len(nodes) > 0:
                current_x = torch.stack(nodes)
                if len(edges) > 0:
                    current_edge_index = torch.stack(edges).t()
                    current_edge_attr = torch.stack(edge_attrs)
                else:
                    current_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
                    current_edge_attr = torch.empty(0, self.edge_dim, device=device)
                    
                current_batch = Batch(
                    x=current_x,
                    edge_index=current_edge_index,
                    edge_attr=current_edge_attr,
                    batch=torch.zeros(len(nodes), dtype=torch.long, device=device)
                )
                context = self.encode_graph_context(current_batch, properties)
            else:
                # Empty graph context with properties
                empty_batch = Batch(
                    x=torch.empty(0, self.node_dim, device=device),
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
                    edge_attr=torch.empty(0, self.edge_dim, device=device),
                    batch=torch.empty(0, dtype=torch.long, device=device)
                )
                empty_batch.num_graphs = 1
                context = self.encode_graph_context(empty_batch, properties)
                
            # Check stopping condition
            if step > 0:
                stop_logits = self.stop_predictor(context)
                stop_prob = torch.sigmoid(stop_logits / temperature)
                if torch.rand(1, device=device) < stop_prob:
                    break
                    
            # Sample new node
            z = torch.randn(1, self.node_dim, device=device) * temperature
            
            new_node = z
            for flow_layer in reversed(self.node_flow_layers):
                new_node, _ = flow_layer(new_node, reverse=False)
                
            nodes.append(new_node.squeeze(0))
            
            # Add edges to existing nodes
            for existing_idx in range(len(nodes) - 1):
                node1_h = self.node_encoder(nodes[-1].unsqueeze(0))
                node2_h = self.node_encoder(nodes[existing_idx].unsqueeze(0))
                edge_input = torch.cat([node1_h, node2_h, context], dim=-1)
                
                edge_logits = self.edge_existence_predictor(edge_input)
                edge_prob = torch.sigmoid(edge_logits / temperature)
                
                if torch.rand(1, device=device) < edge_prob:
                    # Sample edge attributes
                    z_edge = torch.randn(1, self.edge_dim, device=device) * temperature
                    
                    new_edge_attr = z_edge
                    for flow_layer in reversed(self.edge_flow_layers):
                        new_edge_attr, _ = flow_layer(new_edge_attr, reverse=False)
                        
                    # Add bidirectional edges
                    edges.extend([
                        torch.tensor([len(nodes) - 1, existing_idx], device=device),
                        torch.tensor([existing_idx, len(nodes) - 1], device=device)
                    ])
                    edge_attrs.extend([
                        new_edge_attr.squeeze(0),
                        new_edge_attr.squeeze(0)
                    ])
                    
        # Create final graph
        if len(nodes) == 0:
            return Data(
                x=torch.empty(0, self.node_dim, device=device),
                edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
                edge_attr=torch.empty(0, self.edge_dim, device=device)
            )
            
        final_x = torch.stack(nodes)
        if len(edges) > 0:
            final_edge_index = torch.stack(edges).t()
            final_edge_attr = torch.stack(edge_attrs)
        else:
            final_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            final_edge_attr = torch.empty(0, self.edge_dim, device=device)
            
        return Data(
            x=final_x,
            edge_index=final_edge_index,
            edge_attr=final_edge_attr
        )


class MultiObjectiveGenerator:
    """
    Multi-objective molecular generator with property optimization.
    
    This class provides utilities for generating molecules that satisfy
    multiple property constraints simultaneously.
    """
    
    def __init__(self,
                 generator: Union[ConditionalGraphDiffusion, ConditionalGraphAF],
                 property_predictor: Optional[Union[PropertyPredictor, MultiTaskPropertyPredictor]] = None):
        """
        Initialize multi-objective generator.
        
        Args:
            generator: Conditional generative model
            property_predictor: Property prediction model for optimization
        """
        self.generator = generator
        self.property_predictor = property_predictor
        
    def generate_with_constraints(self,
                                num_samples: int,
                                property_targets: Dict[str, Tuple[float, float]],
                                max_iterations: int = 10,
                                **kwargs) -> List[Data]:
        """
        Generate molecules satisfying property constraints.
        
        Args:
            num_samples: Number of molecules to generate
            property_targets: Dictionary of property name to (min, max) ranges
            max_iterations: Maximum optimization iterations
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated molecules satisfying constraints
        """
        device = self.generator.get_device()
        
        # Convert targets to conditioning vector
        target_properties = self._targets_to_vector(property_targets, num_samples, device)
        
        # Generate initial molecules
        molecules = self.generator.sample(
            num_samples=num_samples,
            properties=target_properties,
            **kwargs
        )
        
        # Iterative refinement if property predictor is available
        if self.property_predictor is not None:
            for iteration in range(max_iterations):
                # Predict properties of current molecules
                predicted_props = self.property_predictor.predict_properties(molecules)
                
                # Check which molecules need refinement
                needs_refinement = self._check_constraints(predicted_props, property_targets)
                
                if not any(needs_refinement):
                    break  # All molecules satisfy constraints
                    
                # Refine molecules that don't satisfy constraints
                refined_molecules = self._refine_molecules(
                    molecules, needs_refinement, property_targets, **kwargs
                )
                
                # Update molecules
                for i, needs_refine in enumerate(needs_refinement):
                    if needs_refine and i < len(refined_molecules):
                        molecules[i] = refined_molecules[i]
                        
        return molecules
        
    def _targets_to_vector(self, 
                          targets: Dict[str, Tuple[float, float]], 
                          num_samples: int,
                          device: torch.device) -> torch.Tensor:
        """Convert property targets to conditioning vectors."""
        # For simplicity, use the midpoint of each range
        target_values = []
        for prop_name in self.generator.property_names:
            if prop_name in targets:
                min_val, max_val = targets[prop_name]
                target_val = (min_val + max_val) / 2.0
            else:
                target_val = 0.0  # Default value
            target_values.append(target_val)
            
        # Create tensor
        target_tensor = torch.tensor(target_values, device=device).unsqueeze(0)
        target_tensor = target_tensor.repeat(num_samples, 1)
        
        return target_tensor
        
    def _check_constraints(self, 
                          predictions: Dict[str, List[float]], 
                          targets: Dict[str, Tuple[float, float]]) -> List[bool]:
        """Check which molecules satisfy property constraints."""
        num_molecules = len(next(iter(predictions.values())))
        needs_refinement = [False] * num_molecules
        
        for prop_name, (min_val, max_val) in targets.items():
            if prop_name in predictions:
                prop_values = predictions[prop_name]
                for i, value in enumerate(prop_values):
                    if not (min_val <= value <= max_val):
                        needs_refinement[i] = True
                        
        return needs_refinement
        
    def _refine_molecules(self, 
                         molecules: List[Data],
                         needs_refinement: List[bool],
                         targets: Dict[str, Tuple[float, float]],
                         **kwargs) -> List[Data]:
        """Refine molecules that don't satisfy constraints."""
        # Simple approach: regenerate molecules that need refinement
        num_to_refine = sum(needs_refinement)
        if num_to_refine == 0:
            return []
            
        device = self.generator.get_device()
        target_properties = self._targets_to_vector(targets, num_to_refine, device)
        
        refined = self.generator.sample(
            num_samples=num_to_refine,
            properties=target_properties,
            **kwargs
        )
        
        return refined