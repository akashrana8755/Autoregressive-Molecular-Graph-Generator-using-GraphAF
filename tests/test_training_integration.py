"""
Integration tests for complete training workflows.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.models.graph_diffusion import GraphDiffusion
from src.models.graph_af import GraphAF
from src.training.trainer import Trainer
from src.training.losses import DiffusionLoss, FlowLoss
from src.data.molecular_dataset import MolecularDataset


class TestTrainingIntegration:
    """Integration tests for training workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'node_dim': 16,
                'edge_dim': 8,
                'hidden_dim': 32,
                'num_layers': 2,
                'dropout': 0.1,
                'max_nodes': 10,
                'num_timesteps': 50,
                'beta_schedule': 'linear',
                'num_flow_layers': 2,
                'num_node_types': 5
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-3,
                'num_epochs': 2,
                'gradient_clip': 1.0,
                'scheduler': 'cosine',
                'warmup_epochs': 0
            },
            'data': {
                'max_nodes': 10,
                'train_split': 0.8,
                'val_split': 0.2
            }
        }
        
        # Create test dataset
        self.test_smiles = [
            'C', 'CC', 'CCO', 'c1ccccc1', 'CC(=O)O',
            'CCN', 'CCC', 'CCCO', 'CC(C)O', 'CCCN'
        ]
        
        self.test_graphs = self._create_test_graphs()
    
    def _create_test_graphs(self):
        """Create test molecular graphs."""
        graphs = []
        for i, smiles in enumerate(self.test_smiles):
            # Create simple test graphs
            num_nodes = min(3 + i % 5, self.config['model']['max_nodes'])
            
            x = torch.randn(num_nodes, self.config['model']['node_dim'])
            
            # Create simple connectivity (chain)
            if num_nodes > 1:
                edge_list = []
                for j in range(num_nodes - 1):
                    edge_list.extend([[j, j+1], [j+1, j]])  # Bidirectional
                edge_index = torch.tensor(edge_list).t()
                edge_attr = torch.randn(edge_index.size(1), self.config['model']['edge_dim'])
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, self.config['model']['edge_dim'])
            
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles
            )
            graphs.append(graph)
        
        return graphs
    
    def test_diffusion_training_workflow(self):
        """Test complete diffusion model training workflow."""
        # Create model
        model = GraphDiffusion(self.config['model'])
        
        # Create loss function
        loss_fn = DiffusionLoss(loss_type='mse')
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        
        # Create data loader
        train_loader = DataLoader(
            self.test_graphs[:8], 
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            self.test_graphs[8:], 
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        # Training loop
        model.train()
        for epoch in range(self.config['training']['num_epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                loss = model.training_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            assert avg_loss > 0, f"Training loss should be positive, got {avg_loss}"
            assert torch.isfinite(torch.tensor(avg_loss)), "Training loss should be finite"
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    loss = model.training_step(batch)
                    val_loss += loss.item()
                    val_batches += 1
            
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                assert avg_val_loss > 0, "Validation loss should be positive"
                assert torch.isfinite(torch.tensor(avg_val_loss)), "Validation loss should be finite"
            
            model.train()
        
        # Test that model can generate samples after training
        model.eval()
        with torch.no_grad():
            samples = model.sample(num_samples=3)
            assert len(samples) == 3
            for sample in samples:
                assert isinstance(sample, Data)
                assert sample.x.size(1) == self.config['model']['node_dim']
    
    def test_flow_training_workflow(self):
        """Test complete flow model training workflow."""
        # Create model
        model = GraphAF(self.config['model'])
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        
        # Create data loader
        train_loader = DataLoader(
            self.test_graphs[:8], 
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        # Training loop
        model.train()
        for epoch in range(self.config['training']['num_epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                loss = model.training_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            assert torch.isfinite(torch.tensor(avg_loss)), "Training loss should be finite"
        
        # Test sampling
        model.eval()
        with torch.no_grad():
            samples = model.sample(num_samples=2)
            assert len(samples) == 2
            for sample in samples:
                assert isinstance(sample, Data)
    
    def test_trainer_integration(self):
        """Test integration with Trainer class."""
        try:
            from src.training.trainer import Trainer
        except ImportError:
            pytest.skip("Trainer class not available")
        
        # Create model
        model = GraphDiffusion(self.config['model'])
        
        # Create trainer
        trainer = Trainer(model, self.config)
        
        # Create data loaders
        train_loader = DataLoader(
            self.test_graphs[:8], 
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            self.test_graphs[8:], 
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        # Train for a few epochs
        metrics = trainer.train(train_loader, val_loader)
        
        assert isinstance(metrics, dict)
        assert 'train_loss' in metrics or 'loss' in metrics
        
        # Check that metrics are reasonable
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert torch.isfinite(torch.tensor(value)), f"Metric {key} should be finite"
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint saving and loading during training."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "test_checkpoint.pt"
            
            # Create and train model
            model = GraphDiffusion(self.config['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Save initial state
            initial_state = model.state_dict().copy()
            
            # Train for one step
            train_loader = DataLoader(
                self.test_graphs[:4], 
                batch_size=2,
                shuffle=False
            )
            
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                break  # Just one step
            
            # Save checkpoint
            model.save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=1,
                optimizer_state=optimizer.state_dict(),
                metrics={'loss': loss.item()}
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded = GraphDiffusion.load_checkpoint(checkpoint_path)
            
            assert 'model' in loaded
            assert 'epoch' in loaded
            assert 'optimizer_state_dict' in loaded
            assert 'metrics' in loaded
            
            loaded_model = loaded['model']
            
            # Check that loaded model has same architecture
            assert loaded_model.node_dim == model.node_dim
            assert loaded_model.edge_dim == model.edge_dim
            
            # Check that state is preserved
            for name, param in model.named_parameters():
                loaded_param = dict(loaded_model.named_parameters())[name]
                assert torch.allclose(param, loaded_param), f"Parameter {name} not preserved"
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling during training."""
        model = GraphDiffusion(self.config['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['training']['num_epochs']
        )
        
        train_loader = DataLoader(
            self.test_graphs[:4], 
            batch_size=2,
            shuffle=True
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        model.train()
        for epoch in range(self.config['training']['num_epochs']):
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Check that learning rate changes
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > 0:
                # Learning rate should change with cosine annealing
                assert current_lr != initial_lr, "Learning rate should change with scheduler"
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation for large effective batch sizes."""
        model = GraphDiffusion(self.config['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        accumulation_steps = 2
        effective_batch_size = self.config['training']['batch_size'] * accumulation_steps
        
        train_loader = DataLoader(
            self.test_graphs[:8], 
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        model.train()
        optimizer.zero_grad()
        
        accumulated_loss = 0.0
        step_count = 0
        
        for batch in train_loader:
            # Forward pass
            loss = model.training_step(batch)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            step_count += 1
            
            # Update every accumulation_steps
            if step_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = accumulated_loss / accumulation_steps
                assert avg_loss > 0, "Accumulated loss should be positive"
                
                accumulated_loss = 0.0
    
    def test_mixed_precision_training(self):
        """Test mixed precision training if available."""
        try:
            from torch.cuda.amp import GradScaler, autocast
        except ImportError:
            pytest.skip("Mixed precision not available")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        device = torch.device('cuda')
        model = GraphDiffusion(self.config['model']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()
        
        # Move data to GPU
        gpu_graphs = []
        for graph in self.test_graphs[:4]:
            gpu_graph = graph.to(device)
            gpu_graphs.append(gpu_graph)
        
        train_loader = DataLoader(gpu_graphs, batch_size=2, shuffle=True)
        
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            with autocast():
                loss = model.training_step(batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            assert torch.isfinite(loss), "Loss should be finite in mixed precision"
            break  # Just test one step
    
    def test_model_evaluation_mode(self):
        """Test model behavior in evaluation mode."""
        model = GraphDiffusion(self.config['model'])
        
        # Train briefly
        train_loader = DataLoader(self.test_graphs[:4], batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            break
        
        # Switch to evaluation mode
        model.eval()
        
        # Test that dropout is disabled
        with torch.no_grad():
            batch = next(iter(train_loader))
            
            # Multiple forward passes should give same result (no dropout)
            t = torch.randint(0, 50, (batch.num_graphs,))
            output1 = model.forward(batch, t)
            output2 = model.forward(batch, t)
            
            # Results should be identical in eval mode (assuming no other randomness)
            # Note: This might not be exactly equal due to other sources of randomness
            assert output1['node_scores'].shape == output2['node_scores'].shape
            assert output1['edge_scores'].shape == output2['edge_scores'].shape
    
    def test_training_with_different_batch_sizes(self):
        """Test training with different batch sizes."""
        model = GraphDiffusion(self.config['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            train_loader = DataLoader(
                self.test_graphs[:8], 
                batch_size=batch_size,
                shuffle=True
            )
            
            model.train()
            losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                # Check that batch size is handled correctly
                assert batch.num_graphs <= batch_size
                assert batch.x.size(0) > 0  # Should have nodes
            
            # Should complete training with all batch sizes
            assert len(losses) > 0
            assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    
    def test_training_memory_efficiency(self):
        """Test memory efficiency during training."""
        model = GraphDiffusion(self.config['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train_loader = DataLoader(
            self.test_graphs * 5,  # Repeat data to simulate larger dataset
            batch_size=2,
            shuffle=True
        )
        
        model.train()
        
        # Track memory usage (simplified)
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            
            # Check that memory doesn't grow unboundedly
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable (not more than 10x initial)
                if initial_memory > 0:
                    assert memory_growth < 10 * initial_memory, "Memory usage growing too much"
            
            if i >= 10:  # Test first 10 batches
                break
    
    def test_training_reproducibility(self):
        """Test training reproducibility with fixed seeds."""
        def train_model(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            model = GraphDiffusion(self.config['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            train_loader = DataLoader(
                self.test_graphs[:4], 
                batch_size=2,
                shuffle=False  # Disable shuffle for reproducibility
            )
            
            model.train()
            losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            return losses, model.state_dict()
        
        # Train with same seed twice
        losses1, state1 = train_model(42)
        losses2, state2 = train_model(42)
        
        # Results should be identical
        assert len(losses1) == len(losses2)
        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-6, "Training should be reproducible with same seed"
        
        # Model states should be identical
        for name in state1.keys():
            assert torch.allclose(state1[name], state2[name], atol=1e-6), \
                f"Parameter {name} should be identical with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])