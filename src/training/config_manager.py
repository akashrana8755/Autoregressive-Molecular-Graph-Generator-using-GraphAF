"""
Configuration management for molecular generation experiments.

This module provides utilities for loading, validating, and managing
YAML-based configuration files for training molecular generation models.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass, asdict
from copy import deepcopy
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    type: str = 'diffusion'  # 'diffusion' or 'autoregressive_flow'
    node_dim: Optional[int] = None  # Will be set from data
    edge_dim: Optional[int] = None  # Will be set from data
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    max_nodes: int = 50
    
    # Diffusion-specific parameters
    num_timesteps: int = 1000
    beta_schedule: str = 'cosine'  # 'linear' or 'cosine'
    
    # Flow-specific parameters
    num_flow_layers: int = 4
    num_node_types: int = 10


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    type: str = 'adam'  # 'adam', 'adamw', 'sgd', 'rmsprop'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: List[float] = None
    eps: float = 1e-8
    momentum: float = 0.9  # For SGD
    nesterov: bool = False  # For SGD
    alpha: float = 0.99  # For RMSprop
    
    def __post_init__(self):
        if self.betas is None:
            self.betas = [0.9, 0.999]


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    type: str = 'cosine'  # 'cosine', 'plateau', 'step', 'multistep', 'exponential', 'warmup', 'none'
    
    # Cosine annealing parameters
    T_max: int = 100
    eta_min: float = 1e-6
    
    # Plateau parameters
    mode: str = 'min'
    factor: float = 0.5
    patience: int = 10
    threshold: float = 1e-4
    min_lr: float = 1e-6
    
    # Step parameters
    step_size: int = 30
    gamma: float = 0.1
    
    # MultiStep parameters
    milestones: List[int] = None
    
    # Warmup parameters
    warmup_steps: int = 1000
    warmup_factor: float = 0.1
    main_scheduler: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.milestones is None:
            self.milestones = [30, 60, 90]
        if self.main_scheduler is None:
            self.main_scheduler = {'type': 'cosine'}


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip: Optional[float] = 1.0
    patience: Optional[int] = 15
    seed: int = 42
    num_workers: int = 0
    save_every: int = 10
    validate_every: int = 1
    
    # Loss configuration
    loss_type: str = 'mse'  # For diffusion: 'mse', 'l1', 'huber'
    node_weight: float = 1.0
    edge_weight: float = 1.0
    timestep_weighting: str = 'uniform'  # 'uniform', 'snr', 'truncated_snr'
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None
    
    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig()


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset_path: Optional[str] = None
    dataset_type: str = 'zinc15'  # 'zinc15', 'qm9', 'custom'
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_molecules: Optional[int] = None
    filter_invalid: bool = True
    
    # Data augmentation
    augment_data: bool = False
    augmentation_factor: float = 2.0
    
    # Caching
    cache_processed: bool = True
    cache_dir: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    level: str = 'INFO'
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = 'molecugen'
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = None
    
    # TensorBoard
    use_tensorboard: bool = False
    tensorboard_dir: str = 'tensorboard'
    
    # Metrics logging
    log_every: int = 10
    save_plots: bool = True
    plot_format: str = 'png'
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = []


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = 'molecugen_experiment'
    description: str = ''
    output_dir: str = 'experiments'
    
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    logging: LoggingConfig = None
    
    # Additional metadata
    version: str = '1.0'
    author: str = ''
    tags: List[str] = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.tags is None:
            self.tags = []


class ConfigManager:
    """
    Manager for experiment configurations.
    
    Handles loading, validation, and manipulation of experiment configurations
    with support for inheritance, overrides, and environment variable substitution.
    """
    
    def __init__(self):
        self.config_cache = {}
        
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded experiment configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        logger.info(f"Loading configuration from {config_path}")
        
        # Load YAML
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Handle inheritance
        if 'inherit_from' in config_dict:
            config_dict = self._handle_inheritance(config_dict, config_path.parent)
            
        # Environment variable substitution
        config_dict = self._substitute_env_vars(config_dict)
        
        # Convert to dataclass
        config = self._dict_to_config(config_dict)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
        
    def save_config(self, config: ExperimentConfig, output_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save as YAML
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {output_path}")
        
    def create_default_config(self) -> ExperimentConfig:
        """Create default configuration."""
        return ExperimentConfig()
        
    def merge_configs(self, base_config: ExperimentConfig, 
                     override_config: Dict[str, Any]) -> ExperimentConfig:
        """
        Merge base configuration with overrides.
        
        Args:
            base_config: Base configuration
            override_config: Override values
            
        Returns:
            Merged configuration
        """
        # Convert base config to dict
        base_dict = asdict(base_config)
        
        # Deep merge
        merged_dict = self._deep_merge(base_dict, override_config)
        
        # Convert back to config
        return self._dict_to_config(merged_dict)
        
    def _handle_inheritance(self, config_dict: Dict[str, Any], 
                           config_dir: Path) -> Dict[str, Any]:
        """Handle configuration inheritance."""
        inherit_from = config_dict.pop('inherit_from')
        
        if isinstance(inherit_from, str):
            inherit_from = [inherit_from]
            
        # Load parent configurations
        merged_config = {}
        for parent_path in inherit_from:
            parent_path = config_dir / parent_path
            
            if parent_path in self.config_cache:
                parent_config = self.config_cache[parent_path]
            else:
                with open(parent_path, 'r') as f:
                    parent_config = yaml.safe_load(f)
                self.config_cache[parent_path] = parent_config
                
            merged_config = self._deep_merge(merged_config, parent_config)
            
        # Merge with current config
        return self._deep_merge(merged_config, config_dict)
        
    def _substitute_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration."""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Simple environment variable substitution
                if obj.startswith('${') and obj.endswith('}'):
                    env_var = obj[2:-1]
                    default_value = None
                    
                    if ':' in env_var:
                        env_var, default_value = env_var.split(':', 1)
                        
                    return os.getenv(env_var, default_value)
                return obj
            else:
                return obj
                
        return substitute_recursive(config_dict)
        
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
                
        return result
        
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to configuration dataclass."""
        # Extract sections
        model_dict = config_dict.get('model', {})
        training_dict = config_dict.get('training', {})
        data_dict = config_dict.get('data', {})
        logging_dict = config_dict.get('logging', {})
        
        # Create nested configs
        optimizer_dict = training_dict.pop('optimizer', {})
        scheduler_dict = training_dict.pop('scheduler', {})
        
        model_config = ModelConfig(**model_dict)
        optimizer_config = OptimizerConfig(**optimizer_dict)
        scheduler_config = SchedulerConfig(**scheduler_dict)
        training_config = TrainingConfig(
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            **training_dict
        )
        data_config = DataConfig(**data_dict)
        logging_config = LoggingConfig(**logging_dict)
        
        # Create main config
        main_dict = {k: v for k, v in config_dict.items() 
                    if k not in ['model', 'training', 'data', 'logging']}
        
        return ExperimentConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config,
            **main_dict
        )
        
    def _validate_config(self, config: ExperimentConfig):
        """Validate configuration values."""
        # Validate data splits
        total_split = config.data.train_split + config.data.val_split + config.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
            
        # Validate model type
        if config.model.type not in ['diffusion', 'autoregressive_flow']:
            raise ValueError(f"Unknown model type: {config.model.type}")
            
        # Validate optimizer type
        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad']
        if config.training.optimizer.type not in valid_optimizers:
            raise ValueError(f"Unknown optimizer type: {config.training.optimizer.type}")
            
        # Validate scheduler type
        valid_schedulers = ['cosine', 'plateau', 'step', 'multistep', 'exponential', 'warmup', 'none']
        if config.training.scheduler.type not in valid_schedulers:
            raise ValueError(f"Unknown scheduler type: {config.training.scheduler.type}")
            
        # Validate positive values
        if config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if config.training.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if config.training.optimizer.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
            
        logger.info("Configuration validation passed")


def load_config_from_file(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Convenience function to load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager()
    return manager.load_config(config_path)


def create_config_template(output_path: Union[str, Path]):
    """
    Create a template configuration file.
    
    Args:
        output_path: Path to save template
    """
    template_config = ExperimentConfig(
        name='molecugen_template',
        description='Template configuration for molecular generation',
        model=ModelConfig(
            type='diffusion',
            hidden_dim=256,
            num_layers=4,
            dropout=0.1,
            max_nodes=50,
            num_timesteps=1000,
            beta_schedule='cosine'
        ),
        training=TrainingConfig(
            batch_size=32,
            num_epochs=100,
            gradient_clip=1.0,
            patience=15,
            optimizer=OptimizerConfig(
                type='adam',
                learning_rate=1e-4,
                weight_decay=1e-5
            ),
            scheduler=SchedulerConfig(
                type='cosine',
                T_max=100,
                eta_min=1e-6
            )
        ),
        data=DataConfig(
            dataset_type='zinc15',
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            filter_invalid=True
        ),
        logging=LoggingConfig(
            level='INFO',
            use_wandb=False,
            wandb_project='molecugen'
        )
    )
    
    manager = ConfigManager()
    manager.save_config(template_config, output_path)
    
    print(f"Configuration template saved to {output_path}")


if __name__ == "__main__":
    # Create example configuration template
    create_config_template("config_template.yaml")