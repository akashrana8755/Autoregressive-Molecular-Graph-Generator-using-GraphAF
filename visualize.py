#!/usr/bin/env python3
"""
Molecular visualization utilities for generated molecules and metrics.

This script provides visualization capabilities for molecular structures,
property distributions, and evaluation metrics.

Usage:
    python visualize.py --molecules generated.smi --output-dir plots
    python visualize.py --evaluation-report report.json --plot-type metrics
    python visualize.py --molecules molecules.smi --draw-molecules --max-draw 20
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Error: Matplotlib and Seaborn are required for visualization")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Molecular structure drawing disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize molecules and evaluation metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        '--molecules',
        type=str,
        help='Path to molecules file (SMILES, CSV, or JSON)'
    )
    
    parser.add_argument(
        '--evaluation-report',
        type=str,
        help='Path to evaluation report JSON file'
    )
    
    parser.add_argument(
        '--reference-molecules',
        type=str,
        help='Path to reference molecules for comparison'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--plot-format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Format for output plots'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output plots'
    )
    
    # Plot types
    parser.add_argument(
        '--plot-type',
        choices=['all', 'metrics', 'properties', 'structures', 'comparison'],
        default='all',
        help='Type of plots to generate'
    )
    
    parser.add_argument(
        '--draw-molecules',
        action='store_true',
        help='Draw molecular structures'
    )
    
    parser.add_argument(
        '--max-draw',
        type=int,
        default=20,
        help='Maximum number of molecules to draw'
    )
    
    # Visualization options
    parser.add_argument(
        '--style',
        choices=['default', 'seaborn', 'ggplot', 'bmh'],
        default='seaborn',
        help='Matplotlib style'
    )
    
    parser.add_argument(
        '--color-palette',
        choices=['husl', 'Set2', 'viridis', 'plasma'],
        default='husl',
        help='Color palette for plots'
    )
    
    parser.add_argument(
        '--figure-size',
        nargs=2,
        type=float,
        default=[12, 8],
        help='Figure size (width height)'
    )
    
    # Property-specific options
    parser.add_argument(
        '--properties',
        nargs='+',
        choices=['molecular_weight', 'logp', 'qed', 'tpsa', 'num_atoms', 'num_bonds'],
        help='Specific properties to visualize'
    )
    
    parser.add_argument(
        '--bins',
        type=int,
        default=30,
        help='Number of bins for histograms'
    )
    
    parser.add_argument(
        '--max-molecules',
        type=int,
        help='Maximum number of molecules to process'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def setup_plotting_style(style: str, color_palette: str):
    """Setup matplotlib and seaborn styling."""
    if style == 'seaborn':
        sns.set_style("whitegrid")
    else:
        plt.style.use(style)
    
    sns.set_palette(color_palette)


def load_molecules_from_file(file_path: str, max_molecules: Optional[int] = None) -> List[str]:
    """Load molecules from file (same as in evaluate.py)."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    molecules = []
    
    if file_path.suffix.lower() == '.smi':
        with open(file_path, 'r') as f:
            for line in f:
                smiles = line.strip()
                if smiles and not smiles.startswith('#'):
                    molecules.append(smiles)
                    if max_molecules and len(molecules) >= max_molecules:
                        break
    
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                molecules = data[:max_molecules] if max_molecules else data
            elif isinstance(data, dict) and 'molecules' in data:
                molecules = data['molecules'][:max_molecules] if max_molecules else data['molecules']
    
    logger.info(f"Loaded {len(molecules)} molecules from {file_path}")
    return molecules


def compute_molecular_properties(molecules: List[str]) -> Dict[str, List[float]]:
    """Compute molecular properties for visualization."""
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available for property computation")
        return {}
    
    properties = {
        'molecular_weight': [],
        'logp': [],
        'qed': [],
        'tpsa': [],
        'num_atoms': [],
        'num_bonds': [],
        'num_rings': []
    }
    
    for smiles in molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            properties['molecular_weight'].append(Descriptors.MolWt(mol))
            properties['logp'].append(Descriptors.MolLogP(mol))
            properties['qed'].append(Descriptors.qed(mol))
            properties['tpsa'].append(Descriptors.TPSA(mol))
            properties['num_atoms'].append(mol.GetNumAtoms())
            properties['num_bonds'].append(mol.GetNumBonds())
            properties['num_rings'].append(Descriptors.RingCount(mol))
            
        except Exception as e:
            logger.debug(f"Error computing properties for {smiles}: {e}")
            continue
    
    return properties


def draw_molecular_structures(molecules: List[str], 
                            output_dir: Path, 
                            max_draw: int = 20,
                            plot_format: str = 'png'):
    """Draw molecular structures."""
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available for molecular drawing")
        return
    
    output_dir = output_dir / 'structures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Draw individual molecules
    molecules_to_draw = molecules[:max_draw]
    
    for i, smiles in enumerate(molecules_to_draw):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Draw molecule
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(output_dir / f'molecule_{i:03d}.{plot_format}')
            
        except Exception as e:
            logger.debug(f"Error drawing molecule {i}: {e}")
            continue
    
    # Create grid of molecules
    try:
        valid_mols = []
        valid_smiles = []
        
        for smiles in molecules_to_draw:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)
        
        if valid_mols:
            # Create grid image
            img = Draw.MolsToGridImage(
                valid_mols[:min(20, len(valid_mols))], 
                molsPerRow=4, 
                subImgSize=(200, 200),
                legends=[f"Mol {i}" for i in range(len(valid_mols[:20]))]
            )
            img.save(output_dir / f'molecules_grid.{plot_format}')
            
    except Exception as e:
        logger.error(f"Error creating molecule grid: {e}")
    
    logger.info(f"Drew {len(molecules_to_draw)} molecular structures in {output_dir}")


def plot_property_distributions(properties: Dict[str, List[float]], 
                               output_dir: Path,
                               bins: int = 30,
                               figure_size: tuple = (12, 8),
                               plot_format: str = 'png'):
    """Plot molecular property distributions."""
    n_props = len(properties)
    if n_props == 0:
        return
    
    # Create subplots
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figure_size[0], figure_size[1]))
    if n_props == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (prop_name, values) in enumerate(properties.items()):
        if i >= len(axes) or len(values) == 0:
            continue
        
        ax = axes[i]
        
        # Plot histogram
        n, bins_edges, patches = ax.hist(values, bins=bins, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(prop_name.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'{prop_name.replace("_", " ").title()} Distribution')
        ax.legend()
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'property_distributions.{plot_format}', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved property distribution plots to {output_dir}")


def plot_property_comparison(gen_properties: Dict[str, List[float]],
                           ref_properties: Dict[str, List[float]],
                           output_dir: Path,
                           bins: int = 30,
                           figure_size: tuple = (15, 10),
                           plot_format: str = 'png'):
    """Plot property comparison between generated and reference molecules."""
    common_props = set(gen_properties.keys()) & set(ref_properties.keys())
    if not common_props:
        logger.warning("No common properties found for comparison")
        return
    
    n_props = len(common_props)
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size)
    if n_props == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, prop_name in enumerate(sorted(common_props)):
        if i >= len(axes):
            break
        
        ax = axes[i]
        gen_values = gen_properties[prop_name]
        ref_values = ref_properties[prop_name]
        
        if len(gen_values) == 0 or len(ref_values) == 0:
            continue
        
        # Plot overlapping histograms
        ax.hist(gen_values, bins=bins, alpha=0.6, label='Generated', 
               density=True, edgecolor='black')
        ax.hist(ref_values, bins=bins, alpha=0.6, label='Reference', 
               density=True, edgecolor='black')
        
        # Add means
        gen_mean = np.mean(gen_values)
        ref_mean = np.mean(ref_values)
        
        ax.axvline(gen_mean, color='blue', linestyle='--', linewidth=2)
        ax.axvline(ref_mean, color='orange', linestyle='--', linewidth=2)
        
        ax.set_xlabel(prop_name.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{prop_name.replace("_", " ").title()} Comparison')
        ax.legend()
        
        # Add KS test if scipy available
        try:
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(gen_values, ref_values)
            ax.text(0.02, 0.98, f'KS: {ks_stat:.3f}\np: {p_value:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        except ImportError:
            pass
    
    # Hide unused subplots
    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'property_comparison.{plot_format}', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved property comparison plots to {output_dir}")


def plot_evaluation_metrics(evaluation_data: Dict[str, Any],
                          output_dir: Path,
                          figure_size: tuple = (12, 8),
                          plot_format: str = 'png'):
    """Plot evaluation metrics from evaluation report."""
    
    # Basic metrics
    if 'basic_metrics' in evaluation_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = evaluation_data['basic_metrics']
        metric_names = []
        metric_values = []
        
        for key, value in metrics.items():
            if key in ['validity', 'uniqueness', 'novelty'] and value is not None:
                metric_names.append(key.title())
                metric_values.append(value)
        
        if metric_values:
            bars = ax.bar(metric_names, metric_values, alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title('Basic Evaluation Metrics')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'basic_metrics.{plot_format}', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Drug-likeness metrics
    if 'drug_likeness' in evaluation_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        drug_metrics = evaluation_data['drug_likeness']
        
        # QED metrics
        qed_metrics = ['mean_qed', 'median_qed']
        qed_values = [drug_metrics.get(metric, 0) for metric in qed_metrics]
        qed_names = ['Mean QED', 'Median QED']
        
        bars1 = ax1.bar(qed_names, qed_values, alpha=0.7)
        ax1.set_ylabel('QED Score')
        ax1.set_title('QED Drug-likeness Scores')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars1, qed_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Lipinski compliance
        lipinski_metrics = ['mw_pass_rate', 'logp_pass_rate', 'hbd_pass_rate', 
                           'hba_pass_rate', 'lipinski_pass_rate']
        lipinski_names = ['MW ≤500', 'LogP ≤5', 'HBD ≤5', 'HBA ≤10', 'All Rules']
        lipinski_values = [drug_metrics.get(metric, 0) for metric in lipinski_metrics]
        
        bars2 = ax2.bar(lipinski_names, lipinski_values, alpha=0.7)
        ax2.set_ylabel('Pass Rate')
        ax2.set_title('Lipinski Rule Compliance')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars2, lipinski_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'drug_likeness_metrics.{plot_format}', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved evaluation metric plots to {output_dir}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup plotting
    setup_plotting_style(args.style, args.color_palette)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load molecules if provided
        molecules = []
        if args.molecules:
            molecules = load_molecules_from_file(args.molecules, args.max_molecules)
        
        # Load reference molecules if provided
        reference_molecules = []
        if args.reference_molecules:
            reference_molecules = load_molecules_from_file(args.reference_molecules, args.max_molecules)
        
        # Load evaluation report if provided
        evaluation_data = {}
        if args.evaluation_report:
            with open(args.evaluation_report, 'r') as f:
                evaluation_data = json.load(f)
        
        # Generate visualizations based on plot type
        if args.plot_type in ['all', 'structures'] and args.draw_molecules and molecules:
            logger.info("Drawing molecular structures...")
            draw_molecular_structures(molecules, output_dir, args.max_draw, args.plot_format)
        
        if args.plot_type in ['all', 'properties'] and molecules:
            logger.info("Computing and plotting property distributions...")
            properties = compute_molecular_properties(molecules)
            
            # Filter properties if specified
            if args.properties:
                properties = {k: v for k, v in properties.items() if k in args.properties}
            
            plot_property_distributions(
                properties, output_dir, args.bins, 
                tuple(args.figure_size), args.plot_format
            )
        
        if args.plot_type in ['all', 'comparison'] and molecules and reference_molecules:
            logger.info("Plotting property comparison...")
            gen_properties = compute_molecular_properties(molecules)
            ref_properties = compute_molecular_properties(reference_molecules)
            
            plot_property_comparison(
                gen_properties, ref_properties, output_dir,
                args.bins, tuple(args.figure_size), args.plot_format
            )
        
        if args.plot_type in ['all', 'metrics'] and evaluation_data:
            logger.info("Plotting evaluation metrics...")
            plot_evaluation_metrics(
                evaluation_data, output_dir, 
                tuple(args.figure_size), args.plot_format
            )
        
        logger.info(f"Visualization completed. Plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()