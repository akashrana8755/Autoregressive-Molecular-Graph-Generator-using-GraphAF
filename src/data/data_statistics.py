"""
Data statistics and visualization utilities for molecular datasets.

This module provides utilities for computing detailed statistics and
creating visualizations for molecular datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MolecularStatistics:
    """Compute detailed statistics for molecular datasets."""
    
    def __init__(self, use_rdkit: bool = True):
        """
        Initialize molecular statistics calculator.
        
        Args:
            use_rdkit: Whether to use RDKit for molecular property calculations
        """
        self.use_rdkit = use_rdkit
        
        if self.use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
                self.Chem = Chem
                self.Descriptors = Descriptors
                self.rdMolDescriptors = rdMolDescriptors
                self.Fragments = Fragments
                logger.info("RDKit available for molecular statistics")
            except ImportError:
                logger.warning("RDKit not available, using basic statistics only")
                self.use_rdkit = False
                
    def compute_basic_statistics(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Compute basic statistics for SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with basic statistics
        """
        if not smiles_list:
            return {}
            
        # SMILES length statistics
        lengths = [len(smiles) for smiles in smiles_list]
        
        # Character frequency analysis
        char_counter = Counter()
        for smiles in smiles_list:
            char_counter.update(smiles)
            
        # Most common characters
        common_chars = dict(char_counter.most_common(20))
        
        # Basic patterns
        patterns = {
            'contains_rings': sum(1 for s in smiles_list if any(c in s for c in '()[]')),
            'contains_aromatic': sum(1 for s in smiles_list if any(c in s for c in 'cnops')),
            'contains_charges': sum(1 for s in smiles_list if any(c in s for c in '+-')),
            'contains_stereochemistry': sum(1 for s in smiles_list if any(c in s for c in '@\\/'))
        }
        
        return {
            'total_molecules': len(smiles_list),
            'smiles_length': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths),
                'q25': np.percentile(lengths, 25),
                'q75': np.percentile(lengths, 75)
            },
            'character_frequency': common_chars,
            'structural_patterns': patterns
        }
        
    def compute_molecular_properties(self, smiles_list: List[str],
                                   batch_size: int = 1000,
                                   show_progress: bool = True) -> Dict[str, Any]:
        """
        Compute molecular properties using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with molecular property statistics
        """
        if not self.use_rdkit:
            logger.warning("RDKit not available, cannot compute molecular properties")
            return {}
            
        logger.info(f"Computing molecular properties for {len(smiles_list)} molecules...")
        
        # Property collectors
        properties = defaultdict(list)
        failed_count = 0
        
        # Process in batches
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(smiles_list), batch_size), desc="Computing properties")
            except ImportError:
                iterator = range(0, len(smiles_list), batch_size)
        else:
            iterator = range(0, len(smiles_list), batch_size)
            
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(smiles_list))
            batch = smiles_list[start_idx:end_idx]
            
            for smiles in batch:
                try:
                    mol = self.Chem.MolFromSmiles(smiles)
                    if mol is None:
                        failed_count += 1
                        continue
                        
                    # Basic molecular properties
                    properties['mol_weight'].append(self.Descriptors.MolWt(mol))
                    properties['logp'].append(self.Descriptors.MolLogP(mol))
                    properties['tpsa'].append(self.Descriptors.TPSA(mol))
                    properties['num_atoms'].append(mol.GetNumAtoms())
                    properties['num_bonds'].append(mol.GetNumBonds())
                    properties['num_heavy_atoms'].append(mol.GetNumHeavyAtoms())
                    
                    # Hydrogen bond donors/acceptors
                    properties['num_hbd'].append(self.Descriptors.NumHDonors(mol))
                    properties['num_hba'].append(self.Descriptors.NumHAcceptors(mol))
                    
                    # Rotatable bonds
                    properties['num_rotatable_bonds'].append(self.Descriptors.NumRotatableBonds(mol))
                    
                    # Ring properties
                    properties['num_rings'].append(self.rdMolDescriptors.CalcNumRings(mol))
                    properties['num_aromatic_rings'].append(self.rdMolDescriptors.CalcNumAromaticRings(mol))
                    properties['num_saturated_rings'].append(self.rdMolDescriptors.CalcNumSaturatedRings(mol))
                    
                    # Formal charge
                    properties['formal_charge'].append(self.Chem.rdmolops.GetFormalCharge(mol))
                    
                    # Fraction of sp3 carbons
                    properties['frac_csp3'].append(self.rdMolDescriptors.CalcFractionCsp3(mol))
                    
                    # Molar refractivity
                    properties['molar_refractivity'].append(self.Descriptors.MolMR(mol))
                    
                    # Balaban J index
                    try:
                        properties['balaban_j'].append(self.Descriptors.BalabanJ(mol))
                    except:
                        properties['balaban_j'].append(np.nan)
                        
                    # Bertz complexity index
                    try:
                        properties['bertz_ct'].append(self.Descriptors.BertzCT(mol))
                    except:
                        properties['bertz_ct'].append(np.nan)
                        
                except Exception as e:
                    failed_count += 1
                    logger.debug(f"Failed to compute properties for {smiles}: {e}")
                    
        # Compute statistics for each property
        property_stats = {}
        for prop_name, values in properties.items():
            # Remove NaN values
            clean_values = [v for v in values if not pd.isna(v)]
            
            if clean_values:
                property_stats[prop_name] = {
                    'count': len(clean_values),
                    'mean': np.mean(clean_values),
                    'std': np.std(clean_values),
                    'min': np.min(clean_values),
                    'max': np.max(clean_values),
                    'median': np.median(clean_values),
                    'q25': np.percentile(clean_values, 25),
                    'q75': np.percentile(clean_values, 75),
                    'skewness': self._compute_skewness(clean_values),
                    'kurtosis': self._compute_kurtosis(clean_values)
                }
                
        logger.info(f"Computed properties for {len(smiles_list) - failed_count}/{len(smiles_list)} molecules")
        
        return {
            'property_statistics': property_stats,
            'successful_calculations': len(smiles_list) - failed_count,
            'failed_calculations': failed_count,
            'success_rate': (len(smiles_list) - failed_count) / len(smiles_list) if smiles_list else 0
        }
        
    def compute_fragment_statistics(self, smiles_list: List[str],
                                  batch_size: int = 1000) -> Dict[str, Any]:
        """
        Compute molecular fragment statistics using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with fragment statistics
        """
        if not self.use_rdkit:
            logger.warning("RDKit not available, cannot compute fragment statistics")
            return {}
            
        logger.info(f"Computing fragment statistics for {len(smiles_list)} molecules...")
        
        # Fragment counters
        fragment_counts = defaultdict(int)
        functional_groups = defaultdict(int)
        failed_count = 0
        
        # Common functional group patterns (SMARTS)
        functional_group_patterns = {
            'alcohol': '[OH]',
            'aldehyde': '[CX3H1](=O)',
            'ketone': '[CX3](=O)[CX4]',
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'ester': '[CX3](=O)[OX2H0]',
            'ether': '[OD2]([#6])[#6]',
            'amine_primary': '[NX3;H2;!$(NC=O)]',
            'amine_secondary': '[NX3;H1;!$(NC=O)]',
            'amine_tertiary': '[NX3;H0;!$(NC=O)]',
            'amide': '[NX3][CX3](=[OX1])',
            'nitrile': '[NX1]#[CX2]',
            'nitro': '[NX3+](=O)[O-]',
            'sulfone': '[SX4](=O)(=O)',
            'sulfoxide': '[SX3](=O)',
            'phenol': '[OH][cX3]:[c]',
            'benzene': 'c1ccccc1',
            'pyridine': 'n1ccccc1',
            'furan': 'o1cccc1',
            'thiophene': 's1cccc1',
            'imidazole': 'n1ccnc1'
        }
        
        # Process molecules
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            
            for smiles in batch:
                try:
                    mol = self.Chem.MolFromSmiles(smiles)
                    if mol is None:
                        failed_count += 1
                        continue
                        
                    # Count functional groups
                    for group_name, pattern in functional_group_patterns.items():
                        pattern_mol = self.Chem.MolFromSmarts(pattern)
                        if pattern_mol:
                            matches = mol.GetSubstructMatches(pattern_mol)
                            if matches:
                                functional_groups[group_name] += len(matches)
                                
                    # Count RDKit fragments
                    try:
                        fragment_counts['fr_Al_COO'] += self.Fragments.fr_Al_COO(mol)
                        fragment_counts['fr_Al_OH'] += self.Fragments.fr_Al_OH(mol)
                        fragment_counts['fr_Ar_COO'] += self.Fragments.fr_Ar_COO(mol)
                        fragment_counts['fr_Ar_N'] += self.Fragments.fr_Ar_N(mol)
                        fragment_counts['fr_Ar_NH'] += self.Fragments.fr_Ar_NH(mol)
                        fragment_counts['fr_Ar_OH'] += self.Fragments.fr_Ar_OH(mol)
                        fragment_counts['fr_COO'] += self.Fragments.fr_COO(mol)
                        fragment_counts['fr_COO2'] += self.Fragments.fr_COO2(mol)
                        fragment_counts['fr_C_O'] += self.Fragments.fr_C_O(mol)
                        fragment_counts['fr_C_S'] += self.Fragments.fr_C_S(mol)
                        fragment_counts['fr_HOCCN'] += self.Fragments.fr_HOCCN(mol)
                        fragment_counts['fr_NH0'] += self.Fragments.fr_NH0(mol)
                        fragment_counts['fr_NH1'] += self.Fragments.fr_NH1(mol)
                        fragment_counts['fr_NH2'] += self.Fragments.fr_NH2(mol)
                        fragment_counts['fr_N_O'] += self.Fragments.fr_N_O(mol)
                        fragment_counts['fr_Ndealkylation1'] += self.Fragments.fr_Ndealkylation1(mol)
                        fragment_counts['fr_Ndealkylation2'] += self.Fragments.fr_Ndealkylation2(mol)
                        fragment_counts['fr_Nhpyrrole'] += self.Fragments.fr_Nhpyrrole(mol)
                        fragment_counts['fr_SH'] += self.Fragments.fr_SH(mol)
                        fragment_counts['fr_aldehyde'] += self.Fragments.fr_aldehyde(mol)
                        fragment_counts['fr_benzene'] += self.Fragments.fr_benzene(mol)
                        fragment_counts['fr_furan'] += self.Fragments.fr_furan(mol)
                        fragment_counts['fr_imidazole'] += self.Fragments.fr_imidazole(mol)
                        fragment_counts['fr_ketone'] += self.Fragments.fr_ketone(mol)
                        fragment_counts['fr_lactam'] += self.Fragments.fr_lactam(mol)
                        fragment_counts['fr_lactone'] += self.Fragments.fr_lactone(mol)
                        fragment_counts['fr_methoxy'] += self.Fragments.fr_methoxy(mol)
                        fragment_counts['fr_nitro'] += self.Fragments.fr_nitro(mol)
                        fragment_counts['fr_phenol'] += self.Fragments.fr_phenol(mol)
                        fragment_counts['fr_pyridine'] += self.Fragments.fr_pyridine(mol)
                        fragment_counts['fr_sulfide'] += self.Fragments.fr_sulfide(mol)
                        fragment_counts['fr_sulfonamd'] += self.Fragments.fr_sulfonamd(mol)
                        fragment_counts['fr_sulfone'] += self.Fragments.fr_sulfone(mol)
                        fragment_counts['fr_thiophene'] += self.Fragments.fr_thiophene(mol)
                    except Exception as e:
                        logger.debug(f"Fragment counting error for {smiles}: {e}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.debug(f"Failed to process molecule {smiles}: {e}")
                    
        # Compute fragment frequencies
        total_molecules = len(smiles_list) - failed_count
        fragment_frequencies = {name: count / total_molecules 
                              for name, count in fragment_counts.items()}
        functional_group_frequencies = {name: count / total_molecules 
                                      for name, count in functional_groups.items()}
        
        return {
            'fragment_counts': dict(fragment_counts),
            'fragment_frequencies': fragment_frequencies,
            'functional_group_counts': dict(functional_groups),
            'functional_group_frequencies': functional_group_frequencies,
            'processed_molecules': total_molecules,
            'failed_molecules': failed_count
        }
        
    def _compute_skewness(self, values: List[float]) -> float:
        """Compute skewness of values."""
        if len(values) < 3:
            return 0.0
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
            
        skew = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
        return skew
        
    def _compute_kurtosis(self, values: List[float]) -> float:
        """Compute kurtosis of values."""
        if len(values) < 4:
            return 0.0
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
            
        kurt = np.mean([((x - mean_val) / std_val) ** 4 for x in values]) - 3
        return kurt
        
    def generate_comprehensive_report(self, smiles_list: List[str],
                                    properties: Optional[Dict[str, List[float]]] = None,
                                    dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Generate comprehensive statistical report.
        
        Args:
            smiles_list: List of SMILES strings
            properties: Optional molecular properties
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with comprehensive statistics
        """
        logger.info(f"Generating comprehensive statistics for {dataset_name}...")
        
        report = {
            'dataset_name': dataset_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'basic_statistics': {},
            'molecular_properties': {},
            'fragment_statistics': {},
            'property_statistics': {}
        }
        
        # Basic statistics
        report['basic_statistics'] = self.compute_basic_statistics(smiles_list)
        
        # Molecular properties
        if self.use_rdkit:
            report['molecular_properties'] = self.compute_molecular_properties(smiles_list)
            report['fragment_statistics'] = self.compute_fragment_statistics(smiles_list)
            
        # External property statistics
        if properties:
            prop_stats = {}
            for prop_name, prop_values in properties.items():
                clean_values = [v for v in prop_values if not pd.isna(v)]
                if clean_values:
                    prop_stats[prop_name] = {
                        'count': len(clean_values),
                        'mean': np.mean(clean_values),
                        'std': np.std(clean_values),
                        'min': np.min(clean_values),
                        'max': np.max(clean_values),
                        'median': np.median(clean_values),
                        'q25': np.percentile(clean_values, 25),
                        'q75': np.percentile(clean_values, 75),
                        'missing_count': len(prop_values) - len(clean_values),
                        'missing_rate': (len(prop_values) - len(clean_values)) / len(prop_values)
                    }
            report['property_statistics'] = prop_stats
            
        logger.info(f"Comprehensive statistics generated for {dataset_name}")
        
        return report


def create_statistical_visualizations(statistics: Dict[str, Any],
                                    output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Create statistical visualizations for molecular dataset.
    
    Args:
        statistics: Statistics from MolecularStatistics.generate_comprehensive_report
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    saved_plots = {}
    
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Basic statistics visualization
        if 'basic_statistics' in statistics:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # SMILES length distribution
            basic_stats = statistics['basic_statistics']
            if 'smiles_length' in basic_stats:
                length_stats = basic_stats['smiles_length']
                # Create synthetic data for visualization
                lengths = np.random.normal(length_stats['mean'], length_stats['std'], 1000)
                lengths = np.clip(lengths, length_stats['min'], length_stats['max'])
                
                ax1.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
                ax1.set_title('SMILES Length Distribution')
                ax1.set_xlabel('SMILES Length')
                ax1.set_ylabel('Frequency')
                ax1.axvline(length_stats['mean'], color='red', linestyle='--', label='Mean')
                ax1.legend()
                
            # Character frequency
            if 'character_frequency' in basic_stats:
                char_freq = basic_stats['character_frequency']
                top_chars = dict(list(char_freq.items())[:15])
                
                chars = list(top_chars.keys())
                freqs = list(top_chars.values())
                
                ax2.bar(chars, freqs)
                ax2.set_title('Most Common Characters')
                ax2.set_xlabel('Character')
                ax2.set_ylabel('Frequency')
                ax2.tick_params(axis='x', rotation=45)
                
            # Structural patterns
            if 'structural_patterns' in basic_stats:
                patterns = basic_stats['structural_patterns']
                pattern_names = list(patterns.keys())
                pattern_counts = list(patterns.values())
                
                ax3.bar(pattern_names, pattern_counts)
                ax3.set_title('Structural Patterns')
                ax3.set_ylabel('Count')
                ax3.tick_params(axis='x', rotation=45)
                
            # Dataset overview
            total_mols = basic_stats.get('total_molecules', 0)
            ax4.text(0.1, 0.8, f"Dataset: {statistics.get('dataset_name', 'Unknown')}", 
                    fontsize=14, transform=ax4.transAxes)
            ax4.text(0.1, 0.6, f"Total Molecules: {total_mols:,}", 
                    fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.4, f"Analysis Date: {statistics.get('timestamp', 'Unknown')}", 
                    fontsize=10, transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Dataset Overview')
            
            plt.tight_layout()
            
            if output_dir:
                plot_path = output_dir / 'basic_statistics.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                saved_plots['basic_statistics'] = plot_path
                
            plt.close()
            
        # 2. Molecular properties visualization
        if 'molecular_properties' in statistics and 'property_statistics' in statistics['molecular_properties']:
            prop_stats = statistics['molecular_properties']['property_statistics']
            
            # Select key properties for visualization
            key_properties = ['mol_weight', 'logp', 'tpsa', 'num_atoms', 'num_hbd', 'num_hba']
            available_props = [prop for prop in key_properties if prop in prop_stats]
            
            if available_props:
                n_props = len(available_props)
                n_cols = 3
                n_rows = (n_props + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                    
                for i, prop in enumerate(available_props):
                    stats = prop_stats[prop]
                    
                    # Create synthetic data for visualization
                    data = np.random.normal(stats['mean'], stats['std'], 1000)
                    data = np.clip(data, stats['min'], stats['max'])
                    
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{prop.replace("_", " ").title()} Distribution')
                    axes[i].set_xlabel(prop.replace("_", " ").title())
                    axes[i].set_ylabel('Frequency')
                    axes[i].axvline(stats['mean'], color='red', linestyle='--', label='Mean')
                    axes[i].legend()
                    
                # Hide unused subplots
                for i in range(len(available_props), len(axes)):
                    axes[i].axis('off')
                    
                plt.tight_layout()
                
                if output_dir:
                    plot_path = output_dir / 'molecular_properties.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plots['molecular_properties'] = plot_path
                    
                plt.close()
                
        # 3. Fragment statistics visualization
        if 'fragment_statistics' in statistics:
            frag_stats = statistics['fragment_statistics']
            
            if 'functional_group_frequencies' in frag_stats:
                func_groups = frag_stats['functional_group_frequencies']
                
                # Top functional groups
                top_groups = dict(sorted(func_groups.items(), key=lambda x: x[1], reverse=True)[:15])
                
                if top_groups:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    groups = list(top_groups.keys())
                    freqs = list(top_groups.values())
                    
                    bars = ax.barh(groups, freqs)
                    ax.set_title('Most Common Functional Groups')
                    ax.set_xlabel('Frequency (per molecule)')
                    
                    # Add value labels
                    for bar, freq in zip(bars, freqs):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{freq:.3f}', ha='left', va='center')
                               
                    plt.tight_layout()
                    
                    if output_dir:
                        plot_path = output_dir / 'functional_groups.png'
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        saved_plots['functional_groups'] = plot_path
                        
                    plt.close()
                    
    except Exception as e:
        logger.warning(f"Failed to create statistical visualizations: {e}")
        
    return saved_plots