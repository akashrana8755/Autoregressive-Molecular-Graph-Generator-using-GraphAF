#!/usr/bin/env python3
"""
Molecular evaluation script for comprehensive molecule assessment.

This script provides comprehensive evaluation of generated molecules including
validity, uniqueness, novelty, drug-likeness, and property distribution analysis.

Usage:
    python evaluate.py --generated generated_molecules.smi --reference zinc15_subset.smi
    python evaluate.py --generated molecules.json --output evaluation_report.json
    python evaluate.py --generated molecules.smi --drug-likeness --visualize
"""

import argparse
import logging
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.evaluate.molecular_evaluator import MolecularEvaluator
from src.generate.constraint_filter import ConstraintFilter

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Visualization features disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate generated molecules comprehensively',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        '--generated', '-g',
        type=str,
        required=True,
        help='Path to generated molecules file (SMILES, CSV, or JSON)'
    )
    
    parser.add_argument(
        '--reference', '-r',
        type=str,
        help='Path to reference molecules file for novelty computation'
    )
    
    parser.add_argument(
        '--training-set',
        type=str,
        help='Path to training set molecules for novelty computation'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_report.json',
        help='Output file for evaluation report'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['json', 'csv', 'txt'],
        default='json',
        help='Output format for evaluation report'
    )
    
    # Evaluation options
    parser.add_argument(
        '--basic-metrics',
        action='store_true',
        help='Compute basic metrics (validity, uniqueness, novelty)'
    )
    
    parser.add_argument(
        '--drug-likeness',
        action='store_true',
        help='Compute drug-likeness metrics (QED, Lipinski)'
    )
    
    parser.add_argument(
        '--property-distributions',
        action='store_true',
        help='Compute molecular property distributions'
    )
    
    parser.add_argument(
        '--compare-distributions',
        action='store_true',
        help='Compare property distributions with reference set'
    )
    
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Run comprehensive evaluation (all metrics)'
    )
    
    # Drug-likeness thresholds
    parser.add_argument(
        '--qed-threshold',
        type=float,
        default=0.5,
        help='QED threshold for drug-likeness filtering'
    )
    
    parser.add_argument(
        '--lipinski-strict',
        action='store_true',
        help='Use strict Lipinski compliance (all 4 rules)'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--plot-format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Format for visualization plots'
    )
    
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='plots',
        help='Directory to save visualization plots'
    )
    
    # Filtering options
    parser.add_argument(
        '--filter-invalid',
        action='store_true',
        help='Filter out invalid molecules before evaluation'
    )
    
    parser.add_argument(
        '--max-molecules',
        type=int,
        help='Maximum number of molecules to evaluate'
    )
    
    # Advanced options
    parser.add_argument(
        '--save-filtered',
        action='store_true',
        help='Save filtered molecule sets'
    )
    
    parser.add_argument(
        '--detailed-report',
        action='store_true',
        help='Generate detailed evaluation report'
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
    for logger_name in ['src.evaluate', 'src.generate']:
        logging.getLogger(logger_name).setLevel(level)


def load_molecules_from_file(file_path: str, max_molecules: Optional[int] = None) -> List[str]:
    """Load molecules from various file formats."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    molecules = []
    
    if file_path.suffix.lower() == '.smi':
        # SMILES file
        with open(file_path, 'r') as f:
            for line in f:
                smiles = line.strip()
                if smiles and not smiles.startswith('#'):
                    molecules.append(smiles)
                    if max_molecules and len(molecules) >= max_molecules:
                        break
    
    elif file_path.suffix.lower() == '.csv':
        # CSV file
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Find SMILES column
            smiles_col = 0
            if header:
                for i, col_name in enumerate(header):
                    if 'smiles' in col_name.lower():
                        smiles_col = i
                        break
            
            for row in reader:
                if len(row) > smiles_col:
                    smiles = row[smiles_col].strip()
                    if smiles:
                        molecules.append(smiles)
                        if max_molecules and len(molecules) >= max_molecules:
                            break
    
    elif file_path.suffix.lower() == '.json':
        # JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                molecules = data[:max_molecules] if max_molecules else data
            elif isinstance(data, dict):
                if 'molecules' in data:
                    molecules = data['molecules'][:max_molecules] if max_molecules else data['molecules']
                elif 'smiles' in data:
                    molecules = data['smiles'][:max_molecules] if max_molecules else data['smiles']
                else:
                    raise ValueError("JSON file must contain 'molecules' or 'smiles' key")
            else:
                raise ValueError("Invalid JSON format")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(molecules)} molecules from {file_path}")
    return molecules


def create_visualizations(evaluation_results: Dict[str, Any], 
                         generated_molecules: List[str],
                         reference_molecules: Optional[List[str]] = None,
                         plot_dir: str = 'plots',
                         plot_format: str = 'png'):
    """Create visualization plots for evaluation results."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available, skipping visualizations")
        return
    
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Basic metrics bar plot
    if 'basic_metrics' in evaluation_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = evaluation_results['basic_metrics']
        metric_names = ['Validity', 'Uniqueness']
        metric_values = [metrics['validity'], metrics['uniqueness']]
        
        if metrics.get('novelty') is not None:
            metric_names.append('Novelty')
            metric_values.append(metrics['novelty'])
        
        bars = ax.bar(metric_names, metric_values, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Basic Evaluation Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'basic_metrics.{plot_format}', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Drug-likeness metrics
    if 'drug_likeness' in evaluation_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        drug_metrics = evaluation_results['drug_likeness']
        
        # QED distribution
        if 'qed_scores' in evaluation_results:
            qed_scores = [score for score in evaluation_results['qed_scores'] if not np.isnan(score)]
            if qed_scores:
                ax1.hist(qed_scores, bins=30, alpha=0.7, edgecolor='black')
                ax1.axvline(drug_metrics['mean_qed'], color='red', linestyle='--', 
                           label=f'Mean: {drug_metrics["mean_qed"]:.3f}')
                ax1.set_xlabel('QED Score')
                ax1.set_ylabel('Frequency')
                ax1.set_title('QED Score Distribution')
                ax1.legend()
        
        # Lipinski compliance
        lipinski_metrics = ['mw_pass_rate', 'logp_pass_rate', 'hbd_pass_rate', 'hba_pass_rate', 'lipinski_pass_rate']
        lipinski_names = ['MW ≤500', 'LogP ≤5', 'HBD ≤5', 'HBA ≤10', 'All Rules']
        lipinski_values = [drug_metrics.get(metric, 0) for metric in lipinski_metrics]
        
        bars = ax2.bar(lipinski_names, lipinski_values, alpha=0.7)
        ax2.set_ylabel('Pass Rate')
        ax2.set_title('Lipinski Rule Compliance')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, lipinski_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'drug_likeness.{plot_format}', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Property distributions
    if 'property_distributions' in evaluation_results:
        prop_dists = evaluation_results['property_distributions']
        
        # Create subplots for each property
        n_props = len(prop_dists)
        if n_props > 0:
            n_cols = min(3, n_props)
            n_rows = (n_props + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_props == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (prop_name, values) in enumerate(prop_dists.items()):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                if len(values) > 0:
                    ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_xlabel(prop_name.replace('_', ' ').title())
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{prop_name.replace("_", " ").title()} Distribution')
                    
                    # Add statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    ax.axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}±{std_val:.2f}')
                    ax.legend()
            
            # Hide unused subplots
            for i in range(n_props, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'property_distributions.{plot_format}', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Property comparison with reference
    if 'property_comparison' in evaluation_results and reference_molecules:
        comparison = evaluation_results['property_comparison']
        
        # Create comparison plots
        n_props = len(comparison)
        if n_props > 0:
            n_cols = min(2, n_props)
            n_rows = (n_props + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows))
            if n_props == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (prop_name, comp_data) in enumerate(comparison.items()):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                
                # Create comparison bar plot
                categories = ['Generated', 'Reference']
                means = [comp_data['generated_mean'], comp_data['reference_mean']]
                stds = [comp_data['generated_std'], comp_data['reference_std']]
                
                bars = ax.bar(categories, means, yerr=stds, alpha=0.7, capsize=5)
                ax.set_ylabel(prop_name.replace('_', ' ').title())
                ax.set_title(f'{prop_name.replace("_", " ").title()} Comparison')
                
                # Add KS statistic
                ks_stat = comp_data.get('ks_statistic', 0)
                ax.text(0.5, max(means) * 0.9, f'KS: {ks_stat:.3f}', 
                       ha='center', transform=ax.transData)
            
            # Hide unused subplots
            for i in range(n_props, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'property_comparison.{plot_format}', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Saved visualization plots to {plot_dir}")


def save_evaluation_report(results: Dict[str, Any], 
                          output_path: str, 
                          output_format: str):
    """Save evaluation results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif output_format == 'csv':
        # Flatten results for CSV format
        flattened = {}
        
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten_dict(v, f"{prefix}{k}_")
                elif isinstance(v, (list, np.ndarray)):
                    if len(v) > 0 and isinstance(v[0], (int, float)):
                        flattened[f"{prefix}{k}_mean"] = np.mean(v)
                        flattened[f"{prefix}{k}_std"] = np.std(v)
                    else:
                        flattened[f"{prefix}{k}_count"] = len(v)
                else:
                    flattened[f"{prefix}{k}"] = v
        
        flatten_dict(results)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in flattened.items():
                writer.writerow([key, value])
    
    elif output_format == 'txt':
        with open(output_path, 'w') as f:
            f.write("MOLECULAR EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            def write_section(data, title, indent=0):
                f.write("  " * indent + title + "\n")
                f.write("  " * indent + "-" * len(title) + "\n")
                
                for key, value in data.items():
                    if isinstance(value, dict):
                        f.write("\n")
                        write_section(value, key.replace('_', ' ').title(), indent + 1)
                    elif isinstance(value, (list, np.ndarray)):
                        if len(value) > 0 and isinstance(value[0], (int, float)):
                            f.write(f"{'  ' * (indent + 1)}{key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}\n")
                        else:
                            f.write(f"{'  ' * (indent + 1)}{key}: {len(value)} items\n")
                    else:
                        f.write(f"{'  ' * (indent + 1)}{key}: {value}\n")
                f.write("\n")
            
            write_section(results, "Results")
    
    logger.info(f"Saved evaluation report to {output_path}")


def evaluate_molecules(args: argparse.Namespace) -> Dict[str, Any]:
    """Main molecule evaluation function."""
    # Load generated molecules
    logger.info(f"Loading generated molecules from {args.generated}")
    generated_molecules = load_molecules_from_file(args.generated, args.max_molecules)
    
    if not generated_molecules:
        raise ValueError("No molecules found in generated file")
    
    # Load reference molecules if provided
    reference_molecules = None
    if args.reference:
        logger.info(f"Loading reference molecules from {args.reference}")
        reference_molecules = load_molecules_from_file(args.reference)
    elif args.training_set:
        logger.info(f"Loading training set molecules from {args.training_set}")
        reference_molecules = load_molecules_from_file(args.training_set)
    
    # Filter invalid molecules if requested
    if args.filter_invalid:
        evaluator = MolecularEvaluator()
        valid_molecules = evaluator.get_valid_molecules(generated_molecules)
        logger.info(f"Filtered to {len(valid_molecules)}/{len(generated_molecules)} valid molecules")
        generated_molecules = valid_molecules
    
    # Create evaluator
    evaluator = MolecularEvaluator(reference_molecules)
    
    # Determine which evaluations to run
    run_basic = args.basic_metrics or args.comprehensive
    run_drug_likeness = args.drug_likeness or args.comprehensive
    run_properties = args.property_distributions or args.comprehensive
    run_comparison = args.compare_distributions or args.comprehensive
    
    # If no specific options, run basic metrics
    if not any([run_basic, run_drug_likeness, run_properties, run_comparison]):
        run_basic = True
    
    results = {}
    
    # Basic metrics
    if run_basic:
        logger.info("Computing basic evaluation metrics...")
        basic_metrics = evaluator.evaluate(generated_molecules)
        results['basic_metrics'] = basic_metrics
    
    # Drug-likeness metrics
    if run_drug_likeness:
        logger.info("Computing drug-likeness metrics...")
        drug_likeness = evaluator.compute_drug_likeness_metrics(generated_molecules)
        results['drug_likeness'] = drug_likeness
        
        # Add detailed QED and Lipinski data if requested
        if args.detailed_report:
            results['qed_scores'] = evaluator.compute_qed_scores(generated_molecules)
            results['lipinski_compliance'] = evaluator.compute_lipinski_compliance(generated_molecules)
    
    # Property distributions
    if run_properties:
        logger.info("Computing property distributions...")
        property_distributions = evaluator.compute_property_distributions(generated_molecules)
        results['property_distributions'] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                           for k, v in property_distributions.items()}
    
    # Property comparison with reference
    if run_comparison and reference_molecules:
        logger.info("Comparing property distributions with reference...")
        property_comparison = evaluator.compare_property_distributions(
            generated_molecules, reference_molecules
        )
        results['property_comparison'] = property_comparison
    
    # Comprehensive report
    if args.comprehensive:
        logger.info("Generating comprehensive evaluation report...")
        comprehensive_report = evaluator.generate_comprehensive_report(
            generated_molecules, reference_molecules
        )
        results.update(comprehensive_report)
    
    # Add summary information
    results['summary'] = {
        'total_generated': len(generated_molecules),
        'evaluation_timestamp': str(np.datetime64('now')),
        'reference_molecules_count': len(reference_molecules) if reference_molecules else 0,
        'evaluation_options': {
            'basic_metrics': run_basic,
            'drug_likeness': run_drug_likeness,
            'property_distributions': run_properties,
            'property_comparison': run_comparison,
            'comprehensive': args.comprehensive
        }
    }
    
    # Save filtered molecules if requested
    if args.save_filtered:
        # Save drug-like molecules
        drug_like = evaluator.filter_drug_like_molecules(
            generated_molecules, 
            qed_threshold=args.qed_threshold,
            require_lipinski=args.lipinski_strict
        )
        
        if drug_like:
            drug_like_path = Path(args.output).with_suffix('.drug_like.smi')
            with open(drug_like_path, 'w') as f:
                for smiles in drug_like:
                    f.write(f"{smiles}\n")
            logger.info(f"Saved {len(drug_like)} drug-like molecules to {drug_like_path}")
    
    return results


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.debug)
    
    try:
        # Evaluate molecules
        results = evaluate_molecules(args)
        
        # Create visualizations if requested
        if args.visualize:
            generated_molecules = load_molecules_from_file(args.generated, args.max_molecules)
            reference_molecules = None
            if args.reference:
                reference_molecules = load_molecules_from_file(args.reference)
            elif args.training_set:
                reference_molecules = load_molecules_from_file(args.training_set)
            
            create_visualizations(
                results, generated_molecules, reference_molecules,
                args.plot_dir, args.plot_format
            )
        
        # Save evaluation report
        save_evaluation_report(results, args.output, args.output_format)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if 'basic_metrics' in results:
            basic = results['basic_metrics']
            print(f"Total molecules: {basic['total_molecules']}")
            print(f"Validity: {basic['validity']:.3f}")
            print(f"Uniqueness: {basic['uniqueness']:.3f}")
            if basic.get('novelty') is not None:
                print(f"Novelty: {basic['novelty']:.3f}")
        
        if 'drug_likeness' in results:
            drug = results['drug_likeness']
            print(f"Mean QED: {drug['mean_qed']:.3f}")
            print(f"Lipinski pass rate: {drug['lipinski_pass_rate']:.3f}")
        
        if 'property_comparison' in results:
            print(f"Property comparison with {results['summary']['reference_molecules_count']} reference molecules")
        
        print(f"Report saved to: {args.output}")
        if args.visualize:
            print(f"Plots saved to: {args.plot_dir}")
        print("="*60)
        
        logger.info("Evaluation script completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()