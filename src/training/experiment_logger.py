"""
Experiment logging and tracking for molecular generation models.

This module provides comprehensive logging capabilities including metrics tracking,
visualization, checkpoint management, and integration with experiment tracking
platforms like Weights & Biases and TensorBoard.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings

# Optional imports for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import torch
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and manages training metrics with history and statistics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.rolling_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.best_metrics = {}
        self.step_count = 0
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
            
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
                
            self.metrics[name].append((step, value))
            self.rolling_metrics[name].append(value)
            
            # Update best metrics
            if name not in self.best_metrics:
                self.best_metrics[name] = {'value': value, 'step': step}
            else:
                # Assume lower is better for losses, higher for accuracies
                if 'loss' in name.lower() or 'error' in name.lower():
                    if value < self.best_metrics[name]['value']:
                        self.best_metrics[name] = {'value': value, 'step': step}
                else:
                    if value > self.best_metrics[name]['value']:
                        self.best_metrics[name] = {'value': value, 'step': step}
                        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics."""
        current = {}
        for name, values in self.metrics.items():
            if values:
                current[name] = values[-1][1]
        return current
        
    def get_rolling_stats(self, metric_name: str) -> Dict[str, float]:
        """Get rolling statistics for a metric."""
        if metric_name not in self.rolling_metrics:
            return {}
            
        values = list(self.rolling_metrics[metric_name])
        if not values:
            return {}
            
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
    def get_metric_history(self, metric_name: str) -> List[tuple]:
        """Get full history for a metric."""
        return self.metrics.get(metric_name, [])
        
    def save_metrics(self, filepath: Union[str, Path]):
        """Save metrics to file."""
        filepath = Path(filepath)
        
        # Convert to serializable format
        serializable_metrics = {}
        for name, values in self.metrics.items():
            serializable_metrics[name] = {
                'history': values,
                'best': self.best_metrics.get(name, {}),
                'rolling_stats': self.get_rolling_stats(name)
            }
            
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
    def load_metrics(self, filepath: Union[str, Path]):
        """Load metrics from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for name, metric_data in data.items():
            self.metrics[name] = metric_data['history']
            if 'best' in metric_data:
                self.best_metrics[name] = metric_data['best']
                
            # Rebuild rolling metrics
            recent_values = [v for _, v in metric_data['history'][-self.window_size:]]
            self.rolling_metrics[name] = deque(recent_values, maxlen=self.window_size)


class ExperimentLogger:
    """
    Comprehensive experiment logger with multiple backends.
    
    Supports logging to files, Weights & Biases, TensorBoard, and provides
    visualization and checkpoint management capabilities.
    """
    
    def __init__(self,
                 experiment_name: str,
                 output_dir: Union[str, Path],
                 config: Dict[str, Any],
                 use_wandb: bool = False,
                 use_tensorboard: bool = False,
                 wandb_project: str = 'molecugen',
                 wandb_entity: Optional[str] = None,
                 wandb_tags: Optional[List[str]] = None):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment files
            config: Experiment configuration
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            wandb_project: W&B project name
            wandb_entity: W&B entity name
            wandb_tags: W&B tags
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.start_time = time.time()
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Setup logging backends
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        
        self.wandb_run = None
        self.tensorboard_writer = None
        
        # Initialize backends
        if self.use_wandb:
            self._init_wandb(wandb_project, wandb_entity, wandb_tags)
            
        if self.use_tensorboard:
            self._init_tensorboard()
            
        # Setup file logging
        self._setup_file_logging()
        
        # Log experiment start
        self.log_info(f"Experiment '{experiment_name}' started")
        self.log_config(config)
        
    def _init_wandb(self, project: str, entity: Optional[str], tags: Optional[List[str]]):
        """Initialize Weights & Biases."""
        try:
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=self.experiment_name,
                config=self.config,
                tags=tags or [],
                dir=str(self.output_dir)
            )
            logger.info("Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
            
    def _init_tensorboard(self):
        """Initialize TensorBoard."""
        try:
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.use_tensorboard = False
            
    def _setup_file_logging(self):
        """Setup file-based logging."""
        # Create log file
        log_file = self.output_dir / f"{self.experiment_name}.log"
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ''):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
            prefix: Prefix for metric names
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            
        # Update internal tracker
        self.metrics_tracker.update(metrics, step)
        
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
                
        # Log to TensorBoard
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                for name, value in metrics.items():
                    self.tensorboard_writer.add_scalar(name, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")
                
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        logger.info(f"Configuration saved to {config_file}")
        
    def log_info(self, message: str):
        """Log informational message."""
        logger.info(message)
        
    def log_warning(self, message: str):
        """Log warning message."""
        logger.warning(message)
        
    def log_error(self, message: str):
        """Log error message."""
        logger.error(message)
        
    def log_model_summary(self, model: torch.nn.Module):
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': str(model)
        }
        
        # Save to file
        summary_file = self.output_dir / "model_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Log key metrics
        self.log_metrics({
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/size_mb': summary['model_size_mb']
        })
        
        self.log_info(f"Model summary: {trainable_params:,} trainable parameters")
        
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        dataset_file = self.output_dir / "dataset_info.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
            
        # Log key metrics
        if 'train_size' in dataset_info:
            self.log_metrics({
                'data/train_size': dataset_info['train_size'],
                'data/val_size': dataset_info.get('val_size', 0),
                'data/test_size': dataset_info.get('test_size', 0)
            })
            
        self.log_info(f"Dataset info logged: {dataset_info.get('total_size', 'unknown')} samples")
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'experiment_name': self.experiment_name
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.log_info(f"Best checkpoint saved at epoch {epoch}")
            
        # Save latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        self.log_info(f"Checkpoint saved: epoch {epoch}")
        
    def create_plots(self, save_format: str = 'png'):
        """Create and save training plots."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot training curves
        self._plot_training_curves(plots_dir, save_format)
        
        # Plot learning rate schedule
        self._plot_learning_rate(plots_dir, save_format)
        
        # Plot metric distributions
        self._plot_metric_distributions(plots_dir, save_format)
        
        self.log_info(f"Training plots saved to {plots_dir}")
        
    def _plot_training_curves(self, plots_dir: Path, save_format: str):
        """Plot training and validation curves."""
        # Get loss metrics
        train_losses = self.metrics_tracker.get_metric_history('train_loss')
        val_losses = self.metrics_tracker.get_metric_history('val_loss')
        
        if not train_losses:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        steps, values = zip(*train_losses)
        ax.plot(steps, values, label='Training Loss', alpha=0.7)
        
        # Plot validation loss
        if val_losses:
            val_steps, val_values = zip(*val_losses)
            ax.plot(val_steps, val_values, label='Validation Loss', alpha=0.7)
            
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"training_curves.{save_format}", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_learning_rate(self, plots_dir: Path, save_format: str):
        """Plot learning rate schedule."""
        lr_history = self.metrics_tracker.get_metric_history('learning_rate')
        
        if not lr_history:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps, values = zip(*lr_history)
        ax.plot(steps, values, label='Learning Rate')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"learning_rate.{save_format}", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_metric_distributions(self, plots_dir: Path, save_format: str):
        """Plot distributions of recent metric values."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metric_names = ['train_loss', 'val_loss', 'grad_norm', 'learning_rate']
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
                
            history = self.metrics_tracker.get_metric_history(metric_name)
            if not history:
                axes[i].text(0.5, 0.5, f'No data for {metric_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
                
            # Get recent values
            recent_values = [v for _, v in history[-100:]]
            
            axes[i].hist(recent_values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric_name} Distribution (Recent 100)')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(plots_dir / f"metric_distributions.{save_format}", dpi=300, bbox_inches='tight')
        plt.close()
        
    def log_generated_samples(self, 
                             samples: List[Any], 
                             step: int,
                             sample_type: str = 'molecules'):
        """
        Log generated samples.
        
        Args:
            samples: Generated samples
            step: Current step
            sample_type: Type of samples
        """
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Save samples
        sample_file = samples_dir / f"{sample_type}_step_{step}.pkl"
        with open(sample_file, 'wb') as f:
            pickle.dump(samples, f)
            
        self.log_info(f"Saved {len(samples)} {sample_type} samples at step {step}")
        
        # Log sample count
        self.log_metrics({f'samples/{sample_type}_count': len(samples)}, step)
        
    def finalize(self):
        """Finalize experiment logging."""
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Save final metrics
        metrics_file = self.output_dir / "final_metrics.json"
        self.metrics_tracker.save_metrics(metrics_file)
        
        # Create final plots
        self.create_plots()
        
        # Log experiment summary
        summary = {
            'experiment_name': self.experiment_name,
            'total_time_seconds': total_time,
            'total_time_formatted': str(datetime.timedelta(seconds=int(total_time))),
            'best_metrics': self.metrics_tracker.best_metrics,
            'final_metrics': self.metrics_tracker.get_current_metrics()
        }
        
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Close backends
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
            
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
            
        self.log_info(f"Experiment completed in {summary['total_time_formatted']}")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


def create_experiment_logger(config: Dict[str, Any], 
                           experiment_name: Optional[str] = None) -> ExperimentLogger:
    """
    Create experiment logger from configuration.
    
    Args:
        config: Experiment configuration
        experiment_name: Override experiment name
        
    Returns:
        Configured experiment logger
    """
    logging_config = config.get('logging', {})
    
    if experiment_name is None:
        experiment_name = config.get('name', 'molecugen_experiment')
        
    output_dir = config.get('output_dir', 'experiments')
    
    return ExperimentLogger(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config,
        use_wandb=logging_config.get('use_wandb', False),
        use_tensorboard=logging_config.get('use_tensorboard', False),
        wandb_project=logging_config.get('wandb_project', 'molecugen'),
        wandb_entity=logging_config.get('wandb_entity'),
        wandb_tags=logging_config.get('wandb_tags', [])
    )