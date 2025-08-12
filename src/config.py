"""Configuration management system for MolecuGen."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """Manages configuration loading and validation for MolecuGen."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_name: str) -> DictConfig:
        """Load configuration from YAML file.
        
        Args:
            config_name: Name of configuration file (without .yaml extension)
            
        Returns:
            Loaded configuration as OmegaConf DictConfig
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return OmegaConf.create(config_dict)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    def save_config(self, config: DictConfig, config_name: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            config_name: Name of configuration file (without .yaml extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(config), f, default_flow_style=False)
    
    def merge_configs(self, base_config: DictConfig, override_config: DictConfig) -> DictConfig:
        """Merge two configurations with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        return OmegaConf.merge(base_config, override_config)
    
    def validate_config(self, config: DictConfig, required_keys: list) -> bool:
        """Validate that configuration contains required keys.
        
        Args:
            config: Configuration to validate
            required_keys: List of required configuration keys
            
        Returns:
            True if all required keys are present
            
        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = []
        for key in required_keys:
            if not OmegaConf.select(config, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def get_default_config(self) -> DictConfig:
        """Get default configuration for MolecuGen.
        
        Returns:
            Default configuration
        """
        default_config = {
            "model": {
                "type": "GraphDiffusion",
                "node_dim": 128,
                "edge_dim": 64,
                "hidden_dim": 256,
                "num_layers": 6,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 100,
                "scheduler": "cosine",
                "gradient_clip": 1.0,
                "save_every": 10,
                "validate_every": 5
            },
            "data": {
                "dataset": "zinc15",
                "max_nodes": 50,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "num_workers": 4
            },
            "generation": {
                "num_samples": 1000,
                "max_nodes": 50,
                "temperature": 1.0,
                "batch_size": 64
            },
            "constraints": {
                "lipinski": True,
                "qed_threshold": 0.5,
                "logp_range": [-2, 5],
                "mw_range": [150, 500]
            },
            "evaluation": {
                "metrics": ["validity", "uniqueness", "novelty", "qed", "lipinski"],
                "reference_dataset": "zinc15",
                "num_reference": 10000
            },
            "logging": {
                "level": "INFO",
                "use_wandb": False,
                "use_tensorboard": True,
                "log_dir": "logs"
            }
        }
        
        return OmegaConf.create(default_config)


def load_config_from_file(config_path: str) -> DictConfig:
    """Utility function to load configuration from file path.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    config_manager = ConfigManager()
    config_name = Path(config_path).stem
    return config_manager.load_config(config_name)


def create_default_configs():
    """Create default configuration files."""
    config_manager = ConfigManager()
    
    # Create default GraphDiffusion config
    default_config = config_manager.get_default_config()
    config_manager.save_config(default_config, "graphdiff_default")
    
    # Create GraphAF config variant
    graphaf_config = default_config.copy()
    graphaf_config.model.type = "GraphAF"
    graphaf_config.model.num_flows = 12
    config_manager.save_config(graphaf_config, "graphaf_default")
    
    # Create small dataset config for testing
    test_config = default_config.copy()
    test_config.training.batch_size = 8
    test_config.training.num_epochs = 5
    test_config.data.max_nodes = 20
    test_config.generation.num_samples = 100
    config_manager.save_config(test_config, "test_config")


if __name__ == "__main__":
    # Create default configuration files
    create_default_configs()
    print("Default configuration files created in config/ directory")