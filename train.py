#!/usr/bin/env python3
"""
Main training script for molecular generation models.

This script provides a command-line interface for training GraphDiffusion
and GraphAF models with full configuration support, checkpoint resuming,
and model comparison utilities.

Usage:
    python train.py --config config/graphdiff_default.yaml
    python train.py --config config/graphaf_default.yaml --resume experiments/checkpoint.pt
    python train.py --model diffusion --dataset zinc15 --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.training.trainer import Trainer
from src.training.config_manager import ConfigManager, ExperimentConfig, create_config_template
from src.data.molecular_dataset import MolecularDataset
from src.data.dataset_downloader import DatasetDownloader
from src.models.base_model import BaseGenerativeModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train molecular generation models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for experiments (overrides config)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        choices=['diffusion', 'autoregressive_flow'],
        help='Model type to train'
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int,
        help='Hidden dimension size'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        help='Number of model layers'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Data configuration
    parser.add_argument(
        '--dataset',
        choices=['zinc15', 'qm9', 'custom'],
        help='Dataset to use for training'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to custom dataset'
    )
    
    parser.add_argument(
        '--max-molecules',
        type=int,
        help='Maximum number of molecules to use'
    )
    
    # Checkpoint and resuming
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        help='Path to pretrained model to fine-tune'
    )
    
    # Evaluation and comparison
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate model without training'
    )
    
    parser.add_argument(
        '--compare-models',
        nargs='+',
        help='Compare multiple model checkpoints'
    )
    
    # Utilities
    parser.add_argument(
        '--create-template',
        type=str,
        help='Create configuration template at specified path'
    )
    
    parser.add_argument(
        '--validate-config',
        type=str,
        help='Validate configuration file without training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help='Device to use (cuda, cpu, or auto)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Update root logger level
    logging.getLogger().setLevel(level)
    
    # Update specific loggers
    for logger_name in ['src.training', 'src.models', 'src.data']:
        logging.getLogger(logger_name).setLevel(level)


def load_or_create_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load configuration from file or create from arguments."""
    config_manager = ConfigManager()
    
    if args.config:
        # Load from file
        config = config_manager.load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create default config
        config = config_manager.create_default_config()
        logger.info("Using default configuration")
    
    # Apply command line overrides
    overrides = {}
    
    # Model overrides
    if args.model:
        overrides.setdefault('model', {})['type'] = args.model
    if args.hidden_dim:
        overrides.setdefault('model', {})['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        overrides.setdefault('model', {})['num_layers'] = args.num_layers
    
    # Training overrides
    if args.epochs:
        overrides.setdefault('training', {})['num_epochs'] = args.epochs
    if args.batch_size:
        overrides.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate:
        overrides.setdefault('training', {}).setdefault('optimizer', {})['learning_rate'] = args.learning_rate
    if args.seed:
        overrides.setdefault('training', {})['seed'] = args.seed
    if args.num_workers:
        overrides.setdefault('training', {})['num_workers'] = args.num_workers
    
    # Data overrides
    if args.dataset:
        overrides.setdefault('data', {})['dataset_type'] = args.dataset
    if args.dataset_path:
        overrides.setdefault('data', {})['dataset_path'] = args.dataset_path
    if args.max_molecules:
        overrides.setdefault('data', {})['max_molecules'] = args.max_molecules
    
    # Output directory override
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    
    # Apply overrides
    if overrides:
        config = config_manager.merge_configs(config, overrides)
        logger.info("Applied command line overrides")
    
    return config


def setup_device(device_arg: Optional[str]) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto' or device_arg is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def load_dataset(config: ExperimentConfig) -> MolecularDataset:
    """Load molecular dataset based on configuration."""
    logger.info(f"Loading {config.data.dataset_type} dataset...")
    
    if config.data.dataset_type == 'custom' and config.data.dataset_path:
        # Load custom dataset
        dataset = MolecularDataset.from_file(
            config.data.dataset_path,
            max_molecules=config.data.max_molecules,
            filter_invalid=config.data.filter_invalid
        )
    else:
        # Download and load standard dataset
        downloader = DatasetDownloader()
        
        if config.data.dataset_type == 'zinc15':
            dataset_path = downloader.download_zinc15()
        elif config.data.dataset_type == 'qm9':
            dataset_path = downloader.download_qm9()
        else:
            raise ValueError(f"Unknown dataset type: {config.data.dataset_type}")
        
        dataset = MolecularDataset.from_file(
            dataset_path,
            max_molecules=config.data.max_molecules,
            filter_invalid=config.data.filter_invalid
        )
    
    logger.info(f"Loaded dataset with {len(dataset)} molecules")
    return dataset


def train_model(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """Train molecular generation model."""
    # Setup device
    device = setup_device(args.device)
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Create trainer
    with Trainer(config, device=device) as trainer:
        # Setup data
        trainer.setup_data(dataset)
        
        # Setup model and optimizer
        trainer.setup_model()
        trainer.setup_optimizer()
        
        # Save configuration
        trainer.save_config()
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed training from {args.resume}")
        elif args.pretrained:
            # Load pretrained model for fine-tuning
            checkpoint_data = BaseGenerativeModel.load_checkpoint(args.pretrained, device)
            trainer.model = checkpoint_data['model']
            logger.info(f"Loaded pretrained model from {args.pretrained}")
        
        # Train model
        if not args.evaluate_only:
            logger.info("Starting training...")
            training_results = trainer.train()
            logger.info("Training completed successfully")
            
            # Log final results
            logger.info(f"Best validation loss: {training_results['best_val_loss']:.6f}")
            logger.info(f"Total epochs: {training_results['total_epochs']}")
            logger.info(f"Training time: {training_results['total_time']:.2f} seconds")
        
        # Evaluate on test set
        if trainer.test_loader is not None:
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate()
            logger.info(f"Test loss: {test_results['test_loss']:.6f}")
        
        return {
            'config': config,
            'training_results': training_results if not args.evaluate_only else None,
            'test_results': test_results if trainer.test_loader is not None else None
        }


def compare_models(model_paths: list) -> Dict[str, Any]:
    """Compare multiple trained models."""
    logger.info(f"Comparing {len(model_paths)} models...")
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_path in model_paths:
        try:
            # Load checkpoint
            checkpoint_data = BaseGenerativeModel.load_checkpoint(model_path, device)
            
            # Extract metrics
            model_name = Path(model_path).stem
            results[model_name] = {
                'path': model_path,
                'epoch': checkpoint_data.get('epoch', 'unknown'),
                'metrics': checkpoint_data.get('metrics', {}),
                'config': checkpoint_data.get('config', {})
            }
            
            logger.info(f"Loaded {model_name}: epoch {results[model_name]['epoch']}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}")
            results[Path(model_path).stem] = {'error': str(e)}
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print(f"{'Model':<20} {'Epoch':<8} {'Val Loss':<12} {'Test Loss':<12} {'Notes'}")
    print("-"*80)
    
    for name, data in results.items():
        if 'error' in data:
            print(f"{name:<20} {'ERROR':<8} {'-':<12} {'-':<12} {data['error']}")
        else:
            metrics = data.get('metrics', {})
            val_loss = metrics.get('val_loss', metrics.get('validation_loss', 'N/A'))
            test_loss = metrics.get('test_loss', 'N/A')
            epoch = data.get('epoch', 'N/A')
            
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.6f}"
            if isinstance(test_loss, float):
                test_loss = f"{test_loss:.6f}"
            
            print(f"{name:<20} {epoch:<8} {val_loss:<12} {test_loss:<12}")
    
    print("="*80)
    
    return results


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.debug)
    
    try:
        # Handle utility commands
        if args.create_template:
            create_config_template(args.create_template)
            logger.info(f"Configuration template created at {args.create_template}")
            return
        
        if args.validate_config:
            config_manager = ConfigManager()
            config = config_manager.load_config(args.validate_config)
            logger.info(f"Configuration {args.validate_config} is valid")
            return
        
        if args.compare_models:
            results = compare_models(args.compare_models)
            
            # Save comparison results
            output_file = "model_comparison.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Comparison results saved to {output_file}")
            return
        
        # Load configuration
        config = load_or_create_config(args)
        
        # Train model
        results = train_model(config, args)
        
        logger.info("Training script completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()