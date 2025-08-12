"""Tests for configuration management system."""

import pytest
import tempfile
import os
from pathlib import Path
from omegaconf import DictConfig

from src.config import ConfigManager, load_config_from_file


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_default_config(self):
        """Test default configuration generation."""
        config = self.config_manager.get_default_config()
        
        assert isinstance(config, DictConfig)
        assert "model" in config
        assert "training" in config
        assert "data" in config
        assert "generation" in config
        assert "constraints" in config
        assert "evaluation" in config
        assert "logging" in config
        
        # Test specific values
        assert config.model.type == "GraphDiffusion"
        assert config.training.batch_size == 32
        assert config.data.dataset == "zinc15"
        assert config.constraints.lipinski is True
    
    def test_save_and_load_config(self):
        """Test configuration saving and loading."""
        # Create test configuration
        test_config = self.config_manager.get_default_config()
        test_config.model.type = "TestModel"
        test_config.training.batch_size = 64
        
        # Save configuration
        self.config_manager.save_config(test_config, "test_config")
        
        # Verify file exists
        config_path = Path(self.temp_dir) / "test_config.yaml"
        assert config_path.exists()
        
        # Load configuration
        loaded_config = self.config_manager.load_config("test_config")
        
        # Verify loaded configuration
        assert loaded_config.model.type == "TestModel"
        assert loaded_config.training.batch_size == 64
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            self.config_manager.load_config("nonexistent_config")
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = self.config_manager.get_default_config()
        
        override_config = DictConfig({
            "model": {"type": "GraphAF", "num_flows": 12},
            "training": {"batch_size": 16}
        })
        
        merged_config = self.config_manager.merge_configs(base_config, override_config)
        
        # Check overridden values
        assert merged_config.model.type == "GraphAF"
        assert merged_config.model.num_flows == 12
        assert merged_config.training.batch_size == 16
        
        # Check preserved values
        assert merged_config.model.node_dim == 128  # From base config
        assert merged_config.data.dataset == "zinc15"  # From base config
    
    def test_validate_config(self):
        """Test configuration validation."""
        config = self.config_manager.get_default_config()
        
        # Test valid configuration
        required_keys = ["model.type", "training.batch_size", "data.dataset"]
        assert self.config_manager.validate_config(config, required_keys) is True
        
        # Test missing keys
        incomplete_config = DictConfig({"model": {"type": "GraphDiffusion"}})
        with pytest.raises(ValueError, match="Missing required configuration keys"):
            self.config_manager.validate_config(incomplete_config, required_keys)


def test_load_config_from_file():
    """Test utility function for loading config from file path."""
    # This test requires the actual config files to exist
    config_path = "config/graphdiff_default.yaml"
    
    if os.path.exists(config_path):
        config = load_config_from_file(config_path)
        assert isinstance(config, DictConfig)
        assert config.model.type == "GraphDiffusion"
    else:
        pytest.skip("Config file not found, skipping test")


if __name__ == "__main__":
    pytest.main([__file__])