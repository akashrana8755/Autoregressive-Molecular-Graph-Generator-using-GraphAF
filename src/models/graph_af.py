"""
GraphAF (Graph Autoregressive Flow) model implementation for molecular generation.

This module implements an autoregressive flow model that generates molecular graphs
sequentially by adding nodes and edges one at a time, using normalizing flows
to model the conditional distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_model import BaseGenerativeModel


class MaskedLinear(nn.Module):
    """Masked linear layer for autoregressive flows."""
    
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flows."""
    
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer('mask', mask)
        
        self.scale_net = nn.Sequential(
            MaskedLinear(dim, hidden_dim, mask),
            nn.ReLU(),
            MaskedLinear(hidden_dim, hidden_dim, mask),
            nn.ReLU(),
            MaskedLinear(hidden_dim, dim, mask)
        )
        
        self.translate_net = nn.Sequential(
            MaskedLinear(dim, hidden_dim, mask),
            nn.ReLU(),
            MaskedLinear(hidden_dim, hidden_dim, mask),
            nn.ReLU(),
            MaskedLinear(hidden_dim, dim, mask)
        )
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or reverse pass through coupling layer.
        
        Args:
            x: Input tensor
            reverse: Whether to perform reverse transformation
            
        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        x_masked = x * self.mask
        
        scale = self.scale_net(x_masked)
        translate = self.translate_net(x_masked)
        
        # Ensure scale is positive
        scale = torch.tanh(scale) * 2.0  # Scale between -2 and 2
        
        if not reverse:
            # Forward: y = x * exp(scale) + translate
            y = x * torch.exp(scale) + translate
            log_det = scale.sum(dim=-1)
        else:
            # Reverse: x = (y - translate) * exp(-scale)
            y = (x - translate) * torch.exp(-scale)
            log_det = -scale.sum(dim=-1)
        
        return y, log_det


class GraphAF(BaseGenerativeModel):
    """
    Graph Autoregressive Flow model for molecular generation.
    
    This model generates molecular graphs sequentially using autoregressive flows,
    adding nodes and edges one at a time while modeling the conditional distributions
    with normalizing flows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphAF model.
        
        Args:
            config: Configuration dictionary containing:
                - node_dim: Dimension of node features
                - edge_dim: Dimension of edge features
                - hidden_dim: Hidden dimension for GNN layers
                - num_layers: Number of GNN layers
                - num_flow_layers: Number of flow layers
                - dropout: Dropout rate
                - max_nodes: Maximum number of nodes in generated graphs
        """
        super().__init__(config)
        
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_flow_layers = config.get('num_flow_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        self.max_nodes = config.get('max_nodes', 50)
        
        # Node type vocabulary (for discrete node types)
        self.num_node_types = config.get('num_node_types', 10)  # Common atom types
        
        # Graph neural network for context encoding
        self.node_encoder = nn.Linear(self.node_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(self.edge_dim, self.hidden_dim)
        
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
        
        # Context aggregation
        self.context_aggregator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Node addition components
        self.node_type_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_node_types)
        )
        
        # Node feature flow layers
        self.node_flow_layers = nn.ModuleList()
        for i in range(self.num_flow_layers):
            # Create alternating masks for coupling layers
            mask = torch.zeros(self.node_dim)
            mask[i % 2::2] = 1
            self.node_flow_layers.append(
                CouplingLayer(self.node_dim, self.hidden_dim, mask)
            )
        
        # Edge addition components
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # node1 + node2 + context
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)  # Binary: edge exists or not
        )
        
        # Edge feature flow layers
        self.edge_flow_layers = nn.ModuleList()
        for i in range(self.num_flow_layers):
            mask = torch.zeros(self.edge_dim)
            mask[i % 2::2] = 1
            self.edge_flow_layers.append(
                CouplingLayer(self.edge_dim, self.hidden_dim, mask)
            )
        
        # Stop token predictor (when to stop adding nodes)
        self.stop_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def encode_graph_context(self, batch: Batch) -> torch.Tensor:
        """
        Encode the current graph state into a context vector.
        
        Args:
            batch: Current graph batch
            
        Returns:
            Context vector for each graph in the batch
        """
        if batch.x.size(0) == 0:  # Empty graph
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            return torch.zeros(batch_size, self.hidden_dim, device=self.get_device())
        
        # Encode node and edge features
        node_h = self.node_encoder(batch.x)
        
        if batch.edge_attr.size(0) > 0:
            edge_h = self.edge_encoder(batch.edge_attr)
            
            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                node_h_new = gnn_layer(node_h, batch.edge_index, edge_h)
                node_h = node_h + node_h_new  # Residual connection
                node_h = F.dropout(node_h, p=self.dropout, training=self.training)
        
        # Aggregate to graph-level representation
        graph_context = global_mean_pool(node_h, batch.batch)
        graph_context = self.context_aggregator(graph_context)
        
        return graph_context
    
    def forward(self, batch: Batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (compute log probabilities).
        
        Args:
            batch: Batch of complete molecular graphs
            
        Returns:
            Dictionary containing log probabilities and intermediate results
        """
        device = self.get_device()
        batch_size = batch.num_graphs
        
        total_log_prob = torch.zeros(batch_size, device=device)
        
        # Process each graph in the batch sequentially
        for graph_idx in range(batch_size):
            # Extract single graph
            mask = batch.batch == graph_idx
            node_indices = torch.where(mask)[0]
            
            if len(node_indices) == 0:
                continue
                
            graph_x = batch.x[mask]
            
            # Find edges for this graph
            edge_mask = torch.isin(batch.edge_index[0], node_indices)
            graph_edge_index = batch.edge_index[:, edge_mask]
            graph_edge_attr = batch.edge_attr[edge_mask]
            
            # Remap edge indices to local indices
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
            graph_edge_index = torch.tensor([
                [node_mapping[idx.item()] for idx in graph_edge_index[0]],
                [node_mapping[idx.item()] for idx in graph_edge_index[1]]
            ], device=device)
            
            # Compute log probability for this graph
            graph_log_prob = self._compute_graph_log_prob(
                graph_x, graph_edge_index, graph_edge_attr
            )
            total_log_prob[graph_idx] = graph_log_prob
        
        return {'log_prob': total_log_prob}
    
    def _compute_graph_log_prob(self, 
                               nodes: torch.Tensor, 
                               edge_index: torch.Tensor, 
                               edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of a single graph using autoregressive decomposition.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Log probability of the graph
        """
        device = nodes.device
        num_nodes = nodes.size(0)
        log_prob = torch.tensor(0.0, device=device)
        
        # Build graph incrementally and compute conditional probabilities
        current_nodes = torch.empty(0, self.node_dim, device=device)
        current_edges = torch.empty(2, 0, dtype=torch.long, device=device)
        current_edge_attr = torch.empty(0, self.edge_dim, device=device)
        
        for step in range(num_nodes):
            # Create current graph batch
            if current_nodes.size(0) > 0:
                current_batch = Batch(
                    x=current_nodes,
                    edge_index=current_edges,
                    edge_attr=current_edge_attr,
                    batch=torch.zeros(current_nodes.size(0), dtype=torch.long, device=device)
                )
                context = self.encode_graph_context(current_batch)
            else:
                context = torch.zeros(1, self.hidden_dim, device=device)
            
            # Add new node
            new_node = nodes[step:step+1]  # [1, node_dim]
            
            # Compute node probability using flows
            node_log_prob = self._compute_node_log_prob(new_node, context)
            log_prob += node_log_prob
            
            # Add node to current graph
            current_nodes = torch.cat([current_nodes, new_node], dim=0)
            
            # Add edges from new node to existing nodes
            for existing_node_idx in range(step):
                # Check if edge exists in target graph
                edge_exists = self._check_edge_exists(
                    edge_index, step, existing_node_idx
                )
                
                if edge_exists:
                    # Find the edge attributes
                    edge_attr_idx = self._find_edge_attr_index(
                        edge_index, step, existing_node_idx
                    )
                    if edge_attr_idx is not None:
                        edge_features = edge_attr[edge_attr_idx:edge_attr_idx+1]
                        
                        # Compute edge probability
                        edge_log_prob = self._compute_edge_log_prob(
                            edge_features, current_nodes[step:step+1], 
                            current_nodes[existing_node_idx:existing_node_idx+1], context
                        )
                        log_prob += edge_log_prob
                        
                        # Add edge to current graph (both directions)
                        new_edges = torch.tensor([
                            [step, existing_node_idx],
                            [existing_node_idx, step]
                        ], device=device).t()
                        
                        current_edges = torch.cat([current_edges, new_edges], dim=1)
                        current_edge_attr = torch.cat([
                            current_edge_attr, edge_features, edge_features
                        ], dim=0)
        
        return log_prob
    
    def _compute_node_log_prob(self, node: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute log probability of a node using normalizing flows."""
        log_prob = torch.tensor(0.0, device=node.device)
        
        # Transform through flow layers
        z = node.clone()
        for flow_layer in self.node_flow_layers:
            z, layer_log_det = flow_layer(z, reverse=True)
            log_prob += layer_log_det.sum()
        
        # Base distribution log probability (standard Gaussian)
        base_log_prob = -0.5 * (z ** 2).sum() - 0.5 * z.size(-1) * np.log(2 * np.pi)
        log_prob += base_log_prob
        
        return log_prob
    
    def _compute_edge_log_prob(self, 
                              edge_attr: torch.Tensor,
                              node1: torch.Tensor, 
                              node2: torch.Tensor, 
                              context: torch.Tensor) -> torch.Tensor:
        """Compute log probability of an edge using normalizing flows."""
        log_prob = torch.tensor(0.0, device=edge_attr.device)
        
        # Transform through flow layers
        z = edge_attr.clone()
        for flow_layer in self.edge_flow_layers:
            z, layer_log_det = flow_layer(z, reverse=True)
            log_prob += layer_log_det.sum()
        
        # Base distribution log probability
        base_log_prob = -0.5 * (z ** 2).sum() - 0.5 * z.size(-1) * np.log(2 * np.pi)
        log_prob += base_log_prob
        
        return log_prob
    
    def _check_edge_exists(self, edge_index: torch.Tensor, node1: int, node2: int) -> bool:
        """Check if an edge exists between two nodes."""
        edge_exists = ((edge_index[0] == node1) & (edge_index[1] == node2)).any()
        return edge_exists.item()
    
    def _find_edge_attr_index(self, edge_index: torch.Tensor, node1: int, node2: int) -> Optional[int]:
        """Find the index of edge attributes for a specific edge."""
        edge_mask = (edge_index[0] == node1) & (edge_index[1] == node2)
        indices = torch.where(edge_mask)[0]
        return indices[0].item() if len(indices) > 0 else None
    
    def training_step(self, batch: Batch, **kwargs) -> torch.Tensor:
        """
        Compute negative log likelihood loss for training.
        
        Args:
            batch: Batch of molecular graphs
            
        Returns:
            Negative log likelihood loss
        """
        outputs = self.forward(batch)
        log_probs = outputs['log_prob']
        
        # Negative log likelihood
        loss = -log_probs.mean()
        
        return loss
    
    def sample(self, num_samples: int, **kwargs) -> List[Data]:
        """
        Generate molecular graphs using autoregressive sampling.
        
        Args:
            num_samples: Number of molecules to generate
            **kwargs: Additional sampling parameters
                - max_nodes: Maximum nodes per molecule
                - temperature: Sampling temperature
                
        Returns:
            List of generated molecular graphs
        """
        device = self.get_device()
        max_nodes = kwargs.get('max_nodes', self.max_nodes)
        temperature = kwargs.get('temperature', 1.0)
        
        generated_graphs = []
        
        self.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                graph = self._sample_single_graph(max_nodes, temperature)
                generated_graphs.append(graph)
        
        return generated_graphs
    
    def _sample_single_graph(self, max_nodes: int, temperature: float) -> Data:
        """Sample a single molecular graph autoregressively."""
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
                context = self.encode_graph_context(current_batch)
            else:
                context = torch.zeros(1, self.hidden_dim, device=device)
            
            # Check if we should stop adding nodes
            if step > 0:  # Don't stop on first node
                stop_logits = self.stop_predictor(context)
                stop_prob = torch.sigmoid(stop_logits / temperature)
                if torch.rand(1, device=device) < stop_prob:
                    break
            
            # Sample new node from base distribution and transform through flows
            z = torch.randn(1, self.node_dim, device=device) * temperature
            
            # Transform through flows (forward direction)
            new_node = z
            for flow_layer in reversed(self.node_flow_layers):
                new_node, _ = flow_layer(new_node, reverse=False)
            
            nodes.append(new_node.squeeze(0))
            
            # Add edges to existing nodes
            for existing_idx in range(len(nodes) - 1):
                # Predict edge existence
                node1_h = self.node_encoder(nodes[-1].unsqueeze(0))
                node2_h = self.node_encoder(nodes[existing_idx].unsqueeze(0))
                edge_input = torch.cat([node1_h, node2_h, context], dim=-1)
                
                edge_logits = self.edge_existence_predictor(edge_input)
                edge_prob = torch.sigmoid(edge_logits / temperature)
                
                if torch.rand(1, device=device) < edge_prob:
                    # Sample edge attributes
                    z_edge = torch.randn(1, self.edge_dim, device=device) * temperature
                    
                    # Transform through edge flows
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
            # Return empty graph
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
    
    def log_prob(self, batch: Batch) -> torch.Tensor:
        """
        Compute log probability of a batch of graphs.
        
        Args:
            batch: Batch of molecular graphs
            
        Returns:
            Log probabilities for each graph in the batch
        """
        outputs = self.forward(batch)
        return outputs['log_prob']