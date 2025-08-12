#!/usr/bin/env python3
"""
Molecular Visualization Examples for MolecuGen

This script demonstrates various ways to visualize generated molecules,
their properties, and evaluation metrics.
"""

import sys
import os
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# MolecuGen imports
from src.generate.molecular_generator import MolecularGenerator
from src.evaluate.molecular_evaluator import MolecularEvaluator
from src.generate.constraint_filter import ConstraintFilter

# Optional RDKit imports for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Some visualizations will be disabled.")
    RDKIT_AVAILABLE = False

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MolecularVisualizer:
    """
    Comprehensive molecular visualization toolkit.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_molecular_structures(self, smiles_list: List[str], 
                                 titles: List[str] = None,
                                 mols_per_row: int = 4,
                                 mol_size: tuple = (300, 300),
                                 save_path: str = None) -> None:
        """
        Plot molecular structures from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            titles: Optional titles for each molecule
            mols_per_row: Number of molecules per row
            mol_size: Size of each molecule image
            save_path: Path to save the image
        """
        if not RDKIT_AVAILABLE:
            print("RDKit not available - cannot plot molecular structures")
            return
            
        # Convert SMILES to molecules
        mols = []
        valid_titles = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                if titles:
                    valid_titles.append(titles[i])
                else:
                    valid_titles.append(f"Molecule {i+1}")
        
        if not mols:
            print("No valid molecules to visualize")
            return
            
        # Create grid image
        img = Draw.MolsToGridImage(
            mols, 
            molsPerRow=mols_per_row,
            subImgSize=mol_size,
            legends=valid_titles
        )
        
        # Save or display
        if save_path:
            img.save(self.output_dir / save_path)
            print(f"Molecular structures saved to {self.output_dir / save_path}")
        else:
            img.show()
    
    def plot_property_distributions(self, molecules: List[str],
                                  reference_molecules: List[str] = None,
                                  properties: List[str] = None,
                                  save_path: str = None) -> None:
        """
        Plot property distributions for generated molecules.
        
        Args:
            molecules: Generated molecules
            reference_molecules: Reference molecules for comparison
            properties: Properties to plot
            save_path: Path to save the plot
        """
        if properties is None:
            properties = ['molecular_weight', 'logp', 'num_atoms', 'num_bonds', 'tpsa']
        
        # Calculate properties
        evaluator = MolecularEvaluator()
        gen_props = evaluator.compute_property_distributions(molecules)
        
        if reference_molecules:
            ref_props = evaluator.compute_property_distributions(reference_molecules)
        else:
            ref_props = {}
        
        # Create subplots
        n_props = len(properties)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        property_labels = {
            'molecular_weight': 'Molecular Weight (Da)',
            'logp': 'LogP',
            'num_atoms': 'Number of Atoms',
            'num_bonds': 'Number of Bonds',
            'tpsa': 'TPSA (Ų)',
            'num_rings': 'Number of Rings'
        }
        
        for i, prop in enumerate(properties):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if prop in gen_props and len(gen_props[prop]) > 0:
                # Plot generated molecules
                ax.hist(gen_props[prop], bins=20, alpha=0.7, 
                       label='Generated', color='skyblue', density=True)
                
                # Plot reference molecules if available
                if prop in ref_props and len(ref_props[prop]) > 0:
                    ax.hist(ref_props[prop], bins=20, alpha=0.7,
                           label='Reference', color='orange', density=True)
                
                ax.set_xlabel(property_labels.get(prop, prop))
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {property_labels.get(prop, prop)}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for\n{property_labels.get(prop, prop)}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{property_labels.get(prop, prop)} (No Data)')
        
        # Hide empty subplots
        for i in range(n_props, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Property distributions saved to {self.output_dir / save_path}")
        else:
            plt.show()
    
    def plot_drug_likeness_analysis(self, molecules: List[str],
                                   save_path: str = None) -> None:
        """
        Plot comprehensive drug-likeness analysis.
        
        Args:
            molecules: List of SMILES strings
            save_path: Path to save the plot
        """
        evaluator = MolecularEvaluator()
        constraint_filter = ConstraintFilter()
        
        # Calculate QED scores
        qed_scores = evaluator.compute_qed_scores(molecules)
        valid_qed_scores = [score for score in qed_scores if score > 0]
        
        # Calculate Lipinski compliance
        lipinski_compliance = evaluator.compute_lipinski_compliance(molecules)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # QED distribution
        if valid_qed_scores:
            axes[0, 0].hist(valid_qed_scores, bins=20, alpha=0.7, 
                           color='lightgreen', edgecolor='black')
            axes[0, 0].axvline(np.mean(valid_qed_scores), color='red', linestyle='--',
                              label=f'Mean: {np.mean(valid_qed_scores):.3f}')
            axes[0, 0].axvline(0.5, color='orange', linestyle='--',
                              label='Drug-like threshold (0.5)')
            axes[0, 0].set_xlabel('QED Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('QED Score Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No valid QED scores',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # Lipinski compliance
        if lipinski_compliance['lipinski_pass']:
            rules = ['molecular_weight_ok', 'logp_ok', 'hbd_ok', 'hba_ok', 'lipinski_pass']
            rule_labels = ['MW ≤ 500', 'LogP ≤ 5', 'HBD ≤ 5', 'HBA ≤ 10', 'All Rules']
            pass_rates = [np.mean(lipinski_compliance[rule]) for rule in rules]
            
            bars = axes[0, 1].bar(rule_labels, pass_rates, 
                                 color=['lightblue', 'lightgreen', 'lightcoral', 
                                       'lightyellow', 'lightpink'])
            axes[0, 1].set_ylabel('Pass Rate')
            axes[0, 1].set_title('Lipinski Rule Compliance')
            axes[0, 1].set_ylim(0, 1.1)
            
            # Add percentage labels
            for bar, rate in zip(bars, pass_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + 0.02,
                               f'{rate:.1%}', ha='center', va='bottom')
            
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Molecular weight vs LogP scatter plot
        mw_values = []
        logp_values = []
        colors = []
        
        for smiles in molecules:
            mw = constraint_filter.calculate_molecular_weight(smiles)
            logp = constraint_filter.calculate_logp(smiles)
            qed = constraint_filter.calculate_qed_score(smiles)
            
            if mw and logp and qed:
                mw_values.append(mw)
                logp_values.append(logp)
                colors.append(qed)
        
        if mw_values and logp_values:
            scatter = axes[1, 0].scatter(mw_values, logp_values, c=colors, 
                                       cmap='viridis', alpha=0.7)
            axes[1, 0].set_xlabel('Molecular Weight (Da)')
            axes[1, 0].set_ylabel('LogP')
            axes[1, 0].set_title('Molecular Weight vs LogP (colored by QED)')
            
            # Add Lipinski boundaries
            axes[1, 0].axvline(500, color='red', linestyle='--', alpha=0.5, label='MW limit')
            axes[1, 0].axhline(5, color='red', linestyle='--', alpha=0.5, label='LogP limit')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('QED Score')
        
        # Property correlation heatmap
        prop_data = []
        prop_names = ['MW', 'LogP', 'HBD', 'HBA', 'QED']
        
        for smiles in molecules:
            mw = constraint_filter.calculate_molecular_weight(smiles)
            logp = constraint_filter.calculate_logp(smiles)
            hbd = constraint_filter.calculate_hbd(smiles)
            hba = constraint_filter.calculate_hba(smiles)
            qed = constraint_filter.calculate_qed_score(smiles)
            
            if all(x is not None for x in [mw, logp, hbd, hba, qed]):
                prop_data.append([mw, logp, hbd, hba, qed])
        
        if prop_data:
            prop_df = pd.DataFrame(prop_data, columns=prop_names)
            corr_matrix = prop_df.corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1, 1], square=True)
            axes[1, 1].set_title('Property Correlation Matrix')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for correlation',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Drug-likeness analysis saved to {self.output_dir / save_path}")
        else:
            plt.show()
    
    def plot_generation_comparison(self, generation_results: Dict[str, List[str]],
                                 save_path: str = None) -> None:
        """
        Compare different generation conditions.
        
        Args:
            generation_results: Dict mapping condition names to molecule lists
            save_path: Path to save the plot
        """
        evaluator = MolecularEvaluator()
        
        # Calculate metrics for each condition
        metrics_data = []
        
        for condition, molecules in generation_results.items():
            if molecules:
                # Basic metrics
                basic_metrics = evaluator.evaluate(molecules)
                
                # Drug-likeness metrics
                drug_metrics = evaluator.compute_drug_likeness_metrics(molecules)
                
                # Property statistics
                prop_dist = evaluator.compute_property_distributions(molecules)
                
                metrics_data.append({
                    'condition': condition,
                    'validity': basic_metrics['validity'],
                    'uniqueness': basic_metrics['uniqueness'],
                    'mean_qed': drug_metrics['mean_qed'],
                    'lipinski_pass_rate': drug_metrics['lipinski_pass_rate'],
                    'mean_mw': np.mean(prop_dist['molecular_weight']) if 'molecular_weight' in prop_dist else 0,
                    'mean_logp': np.mean(prop_dist['logp']) if 'logp' in prop_dist else 0,
                    'count': len(molecules)
                })
        
        if not metrics_data:
            print("No data to plot")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Validity and uniqueness
        x_pos = np.arange(len(df))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, df['validity'], width, 
                      label='Validity', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, df['uniqueness'], width,
                      label='Uniqueness', alpha=0.7)
        axes[0, 0].set_xlabel('Condition')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_title('Validity and Uniqueness')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(df['condition'], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # QED scores
        axes[0, 1].bar(df['condition'], df['mean_qed'], alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Condition')
        axes[0, 1].set_ylabel('Mean QED Score')
        axes[0, 1].set_title('Drug-likeness (QED)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Lipinski pass rate
        axes[0, 2].bar(df['condition'], df['lipinski_pass_rate'], alpha=0.7, color='orange')
        axes[0, 2].set_xlabel('Condition')
        axes[0, 2].set_ylabel('Lipinski Pass Rate')
        axes[0, 2].set_title('Lipinski Compliance')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Molecular weight
        axes[1, 0].bar(df['condition'], df['mean_mw'], alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Condition')
        axes[1, 0].set_ylabel('Mean Molecular Weight (Da)')
        axes[1, 0].set_title('Molecular Weight')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # LogP
        axes[1, 1].bar(df['condition'], df['mean_logp'], alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Condition')
        axes[1, 1].set_ylabel('Mean LogP')
        axes[1, 1].set_title('Lipophilicity (LogP)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Molecule count
        axes[1, 2].bar(df['condition'], df['count'], alpha=0.7, color='gray')
        axes[1, 2].set_xlabel('Condition')
        axes[1, 2].set_ylabel('Number of Molecules')
        axes[1, 2].set_title('Generation Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Generation comparison saved to {self.output_dir / save_path}")
        else:
            plt.show()
    
    def create_summary_report(self, molecules: List[str],
                            reference_molecules: List[str] = None,
                            title: str = "Molecular Generation Report",
                            save_path: str = None) -> None:
        """
        Create a comprehensive summary report.
        
        Args:
            molecules: Generated molecules
            reference_molecules: Reference molecules for comparison
            title: Report title
            save_path: Path to save the report
        """
        evaluator = MolecularEvaluator(reference_molecules)
        
        # Calculate all metrics
        basic_metrics = evaluator.evaluate(molecules)
        drug_metrics = evaluator.compute_drug_likeness_metrics(molecules)
        prop_dist = evaluator.compute_property_distributions(molecules)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Basic metrics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Validity', 'Uniqueness', 'Novelty']
        values = [basic_metrics['validity'], basic_metrics['uniqueness'], 
                 basic_metrics.get('novelty', 0)]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Rate')
        ax1.set_title('Basic Metrics')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if value is not None:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.2%}', ha='center', va='bottom')
        
        # Drug-likeness summary (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        drug_labels = ['Mean QED', 'Lipinski Pass']
        drug_values = [drug_metrics['mean_qed'], drug_metrics['lipinski_pass_rate']]
        
        bars = ax2.bar(drug_labels, drug_values, color=['green', 'orange'], alpha=0.7)
        ax2.set_ylabel('Score/Rate')
        ax2.set_title('Drug-likeness')
        ax2.set_ylim(0, 1.1)
        
        for bar, value in zip(bars, drug_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Property statistics (top right span)
        ax3 = fig.add_subplot(gs[0, 2:])
        if 'molecular_weight' in prop_dist and 'logp' in prop_dist:
            mw_data = prop_dist['molecular_weight']
            logp_data = prop_dist['logp']
            
            # Create table of statistics
            stats_data = [
                ['Property', 'Mean', 'Std', 'Min', 'Max'],
                ['MW (Da)', f'{np.mean(mw_data):.1f}', f'{np.std(mw_data):.1f}',
                 f'{np.min(mw_data):.1f}', f'{np.max(mw_data):.1f}'],
                ['LogP', f'{np.mean(logp_data):.2f}', f'{np.std(logp_data):.2f}',
                 f'{np.min(logp_data):.2f}', f'{np.max(logp_data):.2f}']
            ]
            
            table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax3.axis('off')
            ax3.set_title('Property Statistics')
        
        # Molecular weight distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0:2])
        if 'molecular_weight' in prop_dist:
            ax4.hist(prop_dist['molecular_weight'], bins=20, alpha=0.7, 
                    color='skyblue', edgecolor='black')
            ax4.axvline(500, color='red', linestyle='--', alpha=0.7, label='Lipinski limit')
            ax4.set_xlabel('Molecular Weight (Da)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Molecular Weight Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # LogP distribution (middle right)
        ax5 = fig.add_subplot(gs[1, 2:])
        if 'logp' in prop_dist:
            ax5.hist(prop_dist['logp'], bins=20, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            ax5.axvline(5, color='red', linestyle='--', alpha=0.7, label='Lipinski limit')
            ax5.set_xlabel('LogP')
            ax5.set_ylabel('Frequency')
            ax5.set_title('LogP Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # QED distribution (bottom left)
        ax6 = fig.add_subplot(gs[2, 0:2])
        qed_scores = evaluator.compute_qed_scores(molecules)
        valid_qed = [score for score in qed_scores if score > 0]
        
        if valid_qed:
            ax6.hist(valid_qed, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax6.axvline(0.5, color='orange', linestyle='--', alpha=0.7, 
                       label='Drug-like threshold')
            ax6.axvline(np.mean(valid_qed), color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {np.mean(valid_qed):.3f}')
            ax6.set_xlabel('QED Score')
            ax6.set_ylabel('Frequency')
            ax6.set_title('QED Score Distribution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Summary text (bottom right)
        ax7 = fig.add_subplot(gs[2, 2:])
        summary_text = f"""
        Summary Statistics:
        
        Total molecules: {len(molecules)}
        Valid molecules: {int(basic_metrics['validity'] * len(molecules))}
        Unique molecules: {int(basic_metrics['uniqueness'] * len(molecules))}
        
        Drug-likeness:
        Mean QED: {drug_metrics['mean_qed']:.3f}
        Lipinski pass: {drug_metrics['lipinski_pass_rate']:.1%}
        
        Properties:
        Mean MW: {np.mean(prop_dist['molecular_weight']):.1f} Da
        Mean LogP: {np.mean(prop_dist['logp']):.2f}
        """
        
        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax7.axis('off')
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Summary report saved to {self.output_dir / save_path}")
        else:
            plt.show()


def main():
    """
    Main function demonstrating visualization capabilities.
    """
    print("MolecuGen Visualization Examples")
    print("=" * 40)
    
    # Sample molecules for demonstration
    sample_molecules = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Salbutamol
    ]
    
    # Initialize visualizer
    visualizer = MolecularVisualizer("example_visualizations")
    
    print("1. Plotting molecular structures...")
    visualizer.plot_molecular_structures(
        sample_molecules[:6],
        titles=[f"Molecule {i+1}" for i in range(6)],
        save_path="molecular_structures.png"
    )
    
    print("2. Plotting property distributions...")
    visualizer.plot_property_distributions(
        sample_molecules,
        properties=['molecular_weight', 'logp', 'num_atoms', 'tpsa'],
        save_path="property_distributions.png"
    )
    
    print("3. Drug-likeness analysis...")
    visualizer.plot_drug_likeness_analysis(
        sample_molecules,
        save_path="drug_likeness_analysis.png"
    )
    
    print("4. Generation comparison...")
    # Simulate different generation conditions
    generation_results = {
        "Low Temp (0.5)": sample_molecules[:4],
        "Medium Temp (1.0)": sample_molecules[2:6],
        "High Temp (1.5)": sample_molecules[4:8],
    }
    
    visualizer.plot_generation_comparison(
        generation_results,
        save_path="generation_comparison.png"
    )
    
    print("5. Creating summary report...")
    visualizer.create_summary_report(
        sample_molecules,
        title="Sample Molecules Analysis",
        save_path="summary_report.png"
    )
    
    print("\nVisualization examples completed!")
    print(f"All plots saved to: {visualizer.output_dir}")
    
    # If running with actual generator, uncomment below:
    """
    # Load trained model and generate molecules
    generator = MolecularGenerator.from_checkpoint("path/to/model.pt")
    
    # Generate molecules with different conditions
    temp_results = {}
    for temp in [0.8, 1.0, 1.2]:
        molecules = generator.generate(num_molecules=100, temperature=temp)
        temp_results[f"Temperature {temp}"] = molecules
    
    # Visualize results
    visualizer.plot_generation_comparison(temp_results, 
                                        save_path="real_generation_comparison.png")
    
    # Create comprehensive report
    all_molecules = []
    for molecules in temp_results.values():
        all_molecules.extend(molecules)
    
    visualizer.create_summary_report(all_molecules,
                                   title="Generated Molecules Analysis",
                                   save_path="generated_molecules_report.png")
    """


if __name__ == "__main__":
    main()