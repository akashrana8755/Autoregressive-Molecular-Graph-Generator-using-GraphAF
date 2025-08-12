#!/usr/bin/env python3
"""
Molecular generation script with constraint options.

This script provides a command-line interface for generating molecules
using trained models with various constraint filtering options.

Usage:
    python generate.py --model experiments/best_model.pt --num-molecules 1000
    python generate.py --model experiments/best_model.pt --num-molecules 500 --lipinski --qed-threshold 0.6
    python generate.py --config config/generation.yaml --output generated_molecules.smi
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.generate.molecular_generator import MolecularGenerator
from src.generate.constraint_filter import ConstraintFilter
from src.data.smiles_processor import SMILESProcessor
from src.models.base_model import BaseGenerativeModel
from src.models.graph_diffusion import GraphDiffusion
from src.models.graph_af import GraphAF

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('generation.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate molecules using trained models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and configuration
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to generation configuration file'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num-molecules', '-n',
        type=int,
        default=1000,
        help='Number of molecules to generate'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for generation'
    )
    
    parser.add_argument(
        '--max-nodes',
        type=int,
        default=50,
        help='Maximum number of nodes per molecule'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (higher = more diverse)'
    )
    
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=10,
        help='Maximum generation attempts per batch'
    )
    
    # Constraint options
    parser.add_argument(
        '--lipinski',
        action='store_true',
        help='Apply Lipinski Rule of Five filtering'
    )
    
    parser.add_argument(
        '--qed-threshold',
        type=float,
        help='Minimum QED score threshold (0.0-1.0)'
    )
    
    parser.add_argument(
        '--mw-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Molecular weight range (Da)'
    )
    
    parser.add_argument(
        '--logp-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='LogP range'
    )
    
    parser.add_argument(
        '--no-constraints',
        action='store_true',
        help='Generate without any constraint filtering'
    )
    
    # Property-based generation
    parser.add_argument(
        '--target-properties',
        type=str,
        help='JSON string or file path with target properties'
    )
    
    parser.add_argument(
        '--property-tolerance',
        type=float,
        default=0.1,
        help='Tolerance for property matching'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='generated_molecules.smi',
        help='Output file for generated SMILES'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['smi', 'csv', 'json'],
        default='smi',
        help='Output file format'
    )
    
    parser.add_argument(
        '--save-stats',
        action='store_true',
        help='Save generation statistics to JSON file'
    )
    
    parser.add_argument(
        '--save-all',
        action='store_true',
        help='Save all generated molecules (including filtered out)'
    )
    
    # Advanced options
    parser.add_argument(
        '--iterative',
        action='store_true',
        help='Use iterative constraint generation'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum iterations for iterative generation'
    )
    
    parser.add_argument(
        '--temperature-schedule',
        nargs='+',
        type=float,
        help='Temperature schedule for iterative generation'
    )
    
    # Device and performance
    parser.add_argument(
        '--device',
        type=str,
        help='Device to use (cuda, cpu, or auto)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
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
    logging.getLogger().setLevel(level)
    
    # Update specific loggers
    for logger_name in ['src.generate', 'src.models', 'src.data']:
        logging.getLogger(logger_name).setLevel(level)


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


def load_generation_config(config_path: str) -> Dict[str, Any]:
    """Load generation configuration from file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    logger.info(f"Loaded generation configuration from {config_path}")
    return config


def parse_target_properties(target_properties_str: str) -> Dict[str, float]:
    """Parse target properties from string or file."""
    if not target_properties_str:
        return {}
    
    # Check if it's a file path
    if Path(target_properties_str).exists():
        with open(target_properties_str, 'r') as f:
            if target_properties_str.endswith('.json'):
                return json.load(f)
            elif target_properties_str.endswith('.yaml') or target_properties_str.endswith('.yml'):
                return yaml.safe_load(f)
    
    # Try to parse as JSON string
    try:
        return json.loads(target_properties_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid target properties format: {target_properties_str}")


def create_constraint_filter(args: argparse.Namespace) -> Optional[ConstraintFilter]:
    """Create constraint filter based on arguments."""
    if args.no_constraints:
        return None
    
    # Default Lipinski thresholds
    constraint_filter = ConstraintFilter()
    
    # Update QED threshold if specified
    if args.qed_threshold is not None:
        constraint_filter.qed_threshold = args.qed_threshold
    
    return constraint_filter


def create_constraints_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Create constraints dictionary from arguments."""
    constraints = {}
    
    if not args.no_constraints:
        # Apply Lipinski rules by default unless explicitly disabled
        constraints['lipinski'] = args.lipinski or (
            args.qed_threshold is None and 
            args.mw_range is None and 
            args.logp_range is None
        )
        
        if args.qed_threshold is not None:
            constraints['qed_threshold'] = args.qed_threshold
        
        if args.mw_range is not None:
            constraints['mw_range'] = args.mw_range
        
        if args.logp_range is not None:
            constraints['logp_range'] = args.logp_range
    
    return constraints


def save_molecules(molecules: List[str], 
                  output_path: str, 
                  output_format: str,
                  stats: Optional[Dict] = None):
    """Save generated molecules to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'smi':
        # Save as SMILES file
        with open(output_path, 'w') as f:
            for smiles in molecules:
                f.write(f"{smiles}\n")
    
    elif output_format == 'csv':
        # Save as CSV with additional information
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SMILES', 'Index'])
            for i, smiles in enumerate(molecules):
                writer.writerow([smiles, i])
    
    elif output_format == 'json':
        # Save as JSON with metadata
        data = {
            'molecules': molecules,
            'count': len(molecules),
            'statistics': stats or {}
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(molecules)} molecules to {output_path}")


def generate_molecules(args: argparse.Namespace) -> Dict[str, Any]:
    """Main molecule generation function."""
    # Setup device and seed
    device = setup_device(args.device)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    
    # Create SMILES processor and constraint filter
    smiles_processor = SMILESProcessor()
    constraint_filter = create_constraint_filter(args)
    
    # Create molecular generator
    generator = MolecularGenerator.from_checkpoint(
        args.model,
        device=device,
        smiles_processor=smiles_processor,
        constraint_filter=constraint_filter
    )
    
    logger.info(f"Loaded molecular generator with {type(generator.model).__name__} model")
    
    # Parse target properties if specified
    target_properties = {}
    if args.target_properties:
        target_properties = parse_target_properties(args.target_properties)
        logger.info(f"Target properties: {target_properties}")
    
    # Create constraints dictionary
    constraints = create_constraints_dict(args)
    
    # Generate molecules
    logger.info(f"Starting generation of {args.num_molecules} molecules...")
    
    if target_properties:
        # Property-based generation
        generated_molecules = generator.generate_with_properties(
            target_properties=target_properties,
            num_molecules=args.num_molecules,
            tolerance=args.property_tolerance,
            max_nodes=args.max_nodes,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts
        )
        
    elif args.iterative and constraints:
        # Iterative constraint generation
        generated_molecules = generator.iterative_constraint_generation(
            num_molecules=args.num_molecules,
            constraints=constraints,
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            temperature_schedule=args.temperature_schedule,
            max_nodes=args.max_nodes
        )
        
    elif constraints:
        # Standard constraint generation
        result = generator.generate_with_constraints(
            num_molecules=args.num_molecules,
            constraints=constraints,
            max_nodes=args.max_nodes,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            return_all=args.save_all
        )
        
        if args.save_all and isinstance(result, dict):
            generated_molecules = result['constraint_passed']
            all_molecules = result['all_generated']
            failed_molecules = result['constraint_failed']
        else:
            generated_molecules = result
            all_molecules = []
            failed_molecules = []
            
    else:
        # Unconstrained generation
        generated_molecules = generator.generate(
            num_molecules=args.num_molecules,
            max_nodes=args.max_nodes,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts
        )
        all_molecules = []
        failed_molecules = []
    
    # Get generation statistics
    generation_stats = generator.get_generation_statistics()
    
    # Validate generated molecules
    validation_stats = generator.validate_molecules(generated_molecules)
    
    # Combine statistics
    combined_stats = {
        'generation': generation_stats,
        'validation': validation_stats,
        'constraints': constraints,
        'target_properties': target_properties,
        'final_count': len(generated_molecules)
    }
    
    logger.info(f"Generation completed: {len(generated_molecules)} molecules")
    logger.info(f"Validity rate: {validation_stats['validity_rate']:.3f}")
    logger.info(f"Uniqueness rate: {validation_stats['uniqueness_rate']:.3f}")
    
    # Save results
    save_molecules(generated_molecules, args.output, args.output_format, combined_stats)
    
    # Save additional files if requested
    if args.save_all and all_molecules:
        all_output = Path(args.output).with_suffix('.all' + Path(args.output).suffix)
        save_molecules(all_molecules, str(all_output), args.output_format)
        
        if failed_molecules:
            failed_output = Path(args.output).with_suffix('.failed' + Path(args.output).suffix)
            save_molecules(failed_molecules, str(failed_output), args.output_format)
    
    # Save statistics
    if args.save_stats:
        stats_output = Path(args.output).with_suffix('.stats.json')
        with open(stats_output, 'w') as f:
            json.dump(combined_stats, f, indent=2, default=str)
        logger.info(f"Saved statistics to {stats_output}")
    
    return {
        'generated_molecules': generated_molecules,
        'statistics': combined_stats,
        'all_molecules': all_molecules if args.save_all else [],
        'failed_molecules': failed_molecules if args.save_all else []
    }


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.debug)
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_generation_config(args.config)
            logger.info(f"Loaded configuration: {config}")
        
        # Generate molecules
        results = generate_molecules(args)
        
        # Print summary
        stats = results['statistics']
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Generated molecules: {stats['final_count']}")
        print(f"Validity rate: {stats['validation']['validity_rate']:.3f}")
        print(f"Uniqueness rate: {stats['validation']['uniqueness_rate']:.3f}")
        
        if stats['generation']['total_generated'] > 0:
            print(f"Total attempts: {stats['generation']['total_generated']}")
            print(f"Success rate: {stats['generation']['valid_molecules'] / stats['generation']['total_generated']:.3f}")
        
        if stats['constraints']:
            print(f"Constraints applied: {list(stats['constraints'].keys())}")
            if stats['generation']['constraint_passed'] > 0:
                print(f"Constraint pass rate: {stats['generation']['constraint_passed'] / stats['generation']['valid_molecules']:.3f}")
        
        print(f"Output saved to: {args.output}")
        print("="*60)
        
        logger.info("Generation script completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()