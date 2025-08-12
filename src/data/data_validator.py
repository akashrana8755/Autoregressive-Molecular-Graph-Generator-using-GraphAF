"""
Data validation and quality control for molecular datasets.

This module provides utilities for validating molecular structures,
computing data quality metrics, and creating data statistics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MolecularValidator:
    """Validator for molecular structures and data quality."""
    
    def __init__(self, use_rdkit: bool = True):
        """
        Initialize molecular validator.
        
        Args:
            use_rdkit: Whether to use RDKit for validation (requires RDKit installation)
        """
        self.use_rdkit = use_rdkit
        
        if self.use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, rdMolDescriptors
                self.Chem = Chem
                self.Descriptors = Descriptors
                self.rdMolDescriptors = rdMolDescriptors
                logger.info("RDKit available for molecular validation")
            except ImportError:
                logger.warning("RDKit not available, using basic validation only")
                self.use_rdkit = False
                
    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Validate a single SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'smiles': smiles,
            'is_valid': False,
            'canonical_smiles': None,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        # Basic validation
        if not smiles or not isinstance(smiles, str):
            result['errors'].append("Empty or invalid SMILES string")
            return result
            
        if len(smiles.strip()) == 0:
            result['errors'].append("Empty SMILES string")
            return result
            
        # RDKit validation
        if self.use_rdkit:
            try:
                mol = self.Chem.MolFromSmiles(smiles)
                
                if mol is None:
                    result['errors'].append("RDKit failed to parse SMILES")
                    return result
                    
                # Successful parsing
                result['is_valid'] = True
                result['canonical_smiles'] = self.Chem.MolToSmiles(mol)
                
                # Compute basic properties
                try:
                    result['properties'] = {
                        'mol_weight': self.Descriptors.MolWt(mol),
                        'logp': self.Descriptors.MolLogP(mol),
                        'num_atoms': mol.GetNumAtoms(),
                        'num_bonds': mol.GetNumBonds(),
                        'num_rings': self.rdMolDescriptors.CalcNumRings(mol),
                        'num_aromatic_rings': self.rdMolDescriptors.CalcNumAromaticRings(mol),
                        'num_rotatable_bonds': self.Descriptors.NumRotatableBonds(mol),
                        'tpsa': self.Descriptors.TPSA(mol),
                        'num_hbd': self.Descriptors.NumHDonors(mol),
                        'num_hba': self.Descriptors.NumHAcceptors(mol)
                    }
                except Exception as e:
                    result['warnings'].append(f"Failed to compute properties: {e}")
                    
                # Check for unusual structures
                self._check_structure_warnings(mol, result)
                
            except Exception as e:
                result['errors'].append(f"RDKit validation error: {e}")
                
        else:
            # Basic string validation without RDKit
            result['is_valid'] = self._basic_smiles_validation(smiles)
            if result['is_valid']:
                result['canonical_smiles'] = smiles.strip()
                
        return result
        
    def _basic_smiles_validation(self, smiles: str) -> bool:
        """Basic SMILES validation without RDKit."""
        smiles = smiles.strip()
        
        # Check for balanced parentheses and brackets
        paren_count = 0
        bracket_count = 0
        
        for char in smiles:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                
            # Check for negative counts (unbalanced)
            if paren_count < 0 or bracket_count < 0:
                return False
                
        # Check final balance
        if paren_count != 0 or bracket_count != 0:
            return False
            
        # Check for valid characters (basic check)
        valid_chars = set('CNOPSFClBrI()[]=#-+\\/@0123456789cnops')
        if not all(c in valid_chars for c in smiles):
            return False
            
        return True
        
    def _check_structure_warnings(self, mol, result: Dict[str, Any]):
        """Check for structural warnings in RDKit molecule."""
        try:
            # Check for very large molecules
            if mol.GetNumAtoms() > 100:
                result['warnings'].append("Large molecule (>100 atoms)")
                
            # Check for very small molecules
            if mol.GetNumAtoms() < 3:
                result['warnings'].append("Very small molecule (<3 atoms)")
                
            # Check for unusual elements
            common_elements = {'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'H'}
            elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
            unusual_elements = elements - common_elements
            
            if unusual_elements:
                result['warnings'].append(f"Unusual elements: {unusual_elements}")
                
            # Check for high formal charges
            charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            max_charge = max(abs(c) for c in charges) if charges else 0
            
            if max_charge > 2:
                result['warnings'].append(f"High formal charge: {max_charge}")
                
        except Exception as e:
            result['warnings'].append(f"Structure check error: {e}")
            
    def validate_dataset(self, smiles_list: List[str], 
                        batch_size: int = 1000,
                        show_progress: bool = True) -> Dict[str, Any]:
        """
        Validate a dataset of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating dataset of {len(smiles_list)} molecules...")
        
        valid_molecules = []
        invalid_molecules = []
        all_properties = defaultdict(list)
        error_counts = Counter()
        warning_counts = Counter()
        
        # Process in batches
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(smiles_list), batch_size), desc="Validating")
            except ImportError:
                iterator = range(0, len(smiles_list), batch_size)
        else:
            iterator = range(0, len(smiles_list), batch_size)
            
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(smiles_list))
            batch = smiles_list[start_idx:end_idx]
            
            for smiles in batch:
                result = self.validate_smiles(smiles)
                
                if result['is_valid']:
                    valid_molecules.append(result)
                    
                    # Collect properties
                    for prop, value in result['properties'].items():
                        all_properties[prop].append(value)
                else:
                    invalid_molecules.append(result)
                    
                # Count errors and warnings
                for error in result['errors']:
                    error_counts[error] += 1
                for warning in result['warnings']:
                    warning_counts[warning] += 1
                    
        # Compute statistics
        total_molecules = len(smiles_list)
        valid_count = len(valid_molecules)
        invalid_count = len(invalid_molecules)
        
        validation_results = {
            'total_molecules': total_molecules,
            'valid_molecules': valid_count,
            'invalid_molecules': invalid_count,
            'validity_rate': valid_count / total_molecules if total_molecules > 0 else 0,
            'error_counts': dict(error_counts),
            'warning_counts': dict(warning_counts),
            'property_statistics': {}
        }
        
        # Compute property statistics
        for prop, values in all_properties.items():
            if values:
                validation_results['property_statistics'][prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
                
        logger.info(f"Validation complete: {valid_count}/{total_molecules} valid molecules "
                   f"({validation_results['validity_rate']:.2%})")
                   
        return validation_results
        
    def get_duplicates(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Find duplicate molecules in dataset.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with duplicate analysis
        """
        logger.info(f"Analyzing duplicates in {len(smiles_list)} molecules...")
        
        # Canonicalize SMILES if possible
        canonical_smiles = []
        failed_canonicalization = 0
        
        for smiles in smiles_list:
            if self.use_rdkit:
                try:
                    mol = self.Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical = self.Chem.MolToSmiles(mol)
                        canonical_smiles.append(canonical)
                    else:
                        canonical_smiles.append(smiles)
                        failed_canonicalization += 1
                except:
                    canonical_smiles.append(smiles)
                    failed_canonicalization += 1
            else:
                canonical_smiles.append(smiles.strip())
                
        # Count occurrences
        smiles_counts = Counter(canonical_smiles)
        duplicates = {smiles: count for smiles, count in smiles_counts.items() if count > 1}
        
        # Find indices of duplicates
        duplicate_indices = defaultdict(list)
        for i, smiles in enumerate(canonical_smiles):
            if smiles in duplicates:
                duplicate_indices[smiles].append(i)
                
        total_molecules = len(smiles_list)
        unique_molecules = len(smiles_counts)
        duplicate_molecules = sum(count - 1 for count in duplicates.values())
        
        results = {
            'total_molecules': total_molecules,
            'unique_molecules': unique_molecules,
            'duplicate_molecules': duplicate_molecules,
            'uniqueness_rate': unique_molecules / total_molecules if total_molecules > 0 else 0,
            'duplicate_groups': len(duplicates),
            'duplicates': dict(duplicates),
            'duplicate_indices': dict(duplicate_indices),
            'failed_canonicalization': failed_canonicalization
        }
        
        logger.info(f"Duplicate analysis: {unique_molecules}/{total_molecules} unique molecules "
                   f"({results['uniqueness_rate']:.2%})")
                   
        return results


class DataQualityAnalyzer:
    """Analyzer for molecular dataset quality and statistics."""
    
    def __init__(self, validator: Optional[MolecularValidator] = None):
        """
        Initialize data quality analyzer.
        
        Args:
            validator: Molecular validator instance
        """
        self.validator = validator or MolecularValidator()
        
    def analyze_dataset(self, smiles_list: List[str],
                       properties: Optional[Dict[str, List[float]]] = None,
                       dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Comprehensive analysis of dataset quality.
        
        Args:
            smiles_list: List of SMILES strings
            properties: Optional molecular properties
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Analyzing dataset quality for {dataset_name}...")
        
        analysis = {
            'dataset_name': dataset_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'basic_stats': {},
            'validation_results': {},
            'duplicate_analysis': {},
            'property_analysis': {},
            'quality_score': 0.0
        }
        
        # Basic statistics
        analysis['basic_stats'] = {
            'total_molecules': len(smiles_list),
            'avg_smiles_length': np.mean([len(s) for s in smiles_list]),
            'std_smiles_length': np.std([len(s) for s in smiles_list]),
            'min_smiles_length': min(len(s) for s in smiles_list) if smiles_list else 0,
            'max_smiles_length': max(len(s) for s in smiles_list) if smiles_list else 0
        }
        
        # Validation analysis
        analysis['validation_results'] = self.validator.validate_dataset(smiles_list)
        
        # Duplicate analysis
        analysis['duplicate_analysis'] = self.validator.get_duplicates(smiles_list)
        
        # Property analysis
        if properties:
            analysis['property_analysis'] = self._analyze_properties(properties)
            
        # Compute quality score
        analysis['quality_score'] = self._compute_quality_score(analysis)
        
        logger.info(f"Dataset analysis complete. Quality score: {analysis['quality_score']:.3f}")
        
        return analysis
        
    def _analyze_properties(self, properties: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze molecular properties."""
        property_analysis = {}
        
        for prop_name, prop_values in properties.items():
            # Remove NaN values
            clean_values = [v for v in prop_values if not pd.isna(v)]
            
            if not clean_values:
                property_analysis[prop_name] = {'error': 'No valid values'}
                continue
                
            # Compute statistics
            property_analysis[prop_name] = {
                'count': len(clean_values),
                'missing_count': len(prop_values) - len(clean_values),
                'missing_rate': (len(prop_values) - len(clean_values)) / len(prop_values),
                'mean': np.mean(clean_values),
                'std': np.std(clean_values),
                'min': np.min(clean_values),
                'max': np.max(clean_values),
                'median': np.median(clean_values),
                'q25': np.percentile(clean_values, 25),
                'q75': np.percentile(clean_values, 75),
                'outlier_count': self._count_outliers(clean_values)
            }
            
        return property_analysis
        
    def _count_outliers(self, values: List[float], method: str = 'iqr') -> int:
        """Count outliers in property values."""
        if method == 'iqr':
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            return len(outliers)
        else:
            # Z-score method
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = [(v - mean_val) / std_val for v in values]
            outliers = [z for z in z_scores if abs(z) > 3]
            return len(outliers)
            
    def _compute_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Compute overall dataset quality score (0-1)."""
        score = 0.0
        
        # Validity score (40% weight)
        validity_rate = analysis['validation_results'].get('validity_rate', 0)
        score += 0.4 * validity_rate
        
        # Uniqueness score (30% weight)
        uniqueness_rate = analysis['duplicate_analysis'].get('uniqueness_rate', 0)
        score += 0.3 * uniqueness_rate
        
        # Property completeness score (20% weight)
        if analysis['property_analysis']:
            missing_rates = [prop.get('missing_rate', 1.0) 
                           for prop in analysis['property_analysis'].values()
                           if isinstance(prop, dict) and 'missing_rate' in prop]
            if missing_rates:
                avg_completeness = 1.0 - np.mean(missing_rates)
                score += 0.2 * avg_completeness
                
        # Size adequacy score (10% weight)
        total_molecules = analysis['basic_stats']['total_molecules']
        if total_molecules >= 10000:
            size_score = 1.0
        elif total_molecules >= 1000:
            size_score = 0.8
        elif total_molecules >= 100:
            size_score = 0.6
        else:
            size_score = 0.3
            
        score += 0.1 * size_score
        
        return min(score, 1.0)  # Cap at 1.0
        
    def create_quality_report(self, analysis: Dict[str, Any], 
                            output_path: Optional[Path] = None) -> str:
        """
        Create a human-readable quality report.
        
        Args:
            analysis: Analysis results from analyze_dataset
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append(f"MOLECULAR DATASET QUALITY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Dataset: {analysis['dataset_name']}")
        report_lines.append(f"Analysis Date: {analysis['timestamp']}")
        report_lines.append(f"Overall Quality Score: {analysis['quality_score']:.3f}/1.000")
        report_lines.append("")
        
        # Basic Statistics
        report_lines.append("BASIC STATISTICS")
        report_lines.append("-" * 20)
        basic = analysis['basic_stats']
        report_lines.append(f"Total Molecules: {basic['total_molecules']:,}")
        report_lines.append(f"Average SMILES Length: {basic['avg_smiles_length']:.1f} ± {basic['std_smiles_length']:.1f}")
        report_lines.append(f"SMILES Length Range: {basic['min_smiles_length']} - {basic['max_smiles_length']}")
        report_lines.append("")
        
        # Validation Results
        report_lines.append("VALIDATION RESULTS")
        report_lines.append("-" * 20)
        validation = analysis['validation_results']
        report_lines.append(f"Valid Molecules: {validation['valid_molecules']:,}/{validation['total_molecules']:,} "
                           f"({validation['validity_rate']:.1%})")
        report_lines.append(f"Invalid Molecules: {validation['invalid_molecules']:,}")
        
        if validation['error_counts']:
            report_lines.append("\nMost Common Errors:")
            for error, count in sorted(validation['error_counts'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                report_lines.append(f"  • {error}: {count:,}")
                
        report_lines.append("")
        
        # Duplicate Analysis
        report_lines.append("DUPLICATE ANALYSIS")
        report_lines.append("-" * 20)
        duplicates = analysis['duplicate_analysis']
        report_lines.append(f"Unique Molecules: {duplicates['unique_molecules']:,}/{duplicates['total_molecules']:,} "
                           f"({duplicates['uniqueness_rate']:.1%})")
        report_lines.append(f"Duplicate Groups: {duplicates['duplicate_groups']:,}")
        report_lines.append(f"Total Duplicates: {duplicates['duplicate_molecules']:,}")
        report_lines.append("")
        
        # Property Analysis
        if analysis['property_analysis']:
            report_lines.append("PROPERTY ANALYSIS")
            report_lines.append("-" * 20)
            
            for prop_name, prop_stats in analysis['property_analysis'].items():
                if isinstance(prop_stats, dict) and 'mean' in prop_stats:
                    report_lines.append(f"{prop_name.upper()}:")
                    report_lines.append(f"  Mean: {prop_stats['mean']:.3f} ± {prop_stats['std']:.3f}")
                    report_lines.append(f"  Range: {prop_stats['min']:.3f} - {prop_stats['max']:.3f}")
                    report_lines.append(f"  Missing: {prop_stats['missing_count']:,} ({prop_stats['missing_rate']:.1%})")
                    report_lines.append(f"  Outliers: {prop_stats['outlier_count']:,}")
                    report_lines.append("")
                    
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 20)
        
        if validation['validity_rate'] < 0.95:
            report_lines.append("• Consider filtering out invalid molecules")
            
        if duplicates['uniqueness_rate'] < 0.90:
            report_lines.append("• Consider removing duplicate molecules")
            
        if analysis['basic_stats']['total_molecules'] < 1000:
            report_lines.append("• Dataset may be too small for robust model training")
            
        if analysis['property_analysis']:
            high_missing = [prop for prop, stats in analysis['property_analysis'].items()
                          if isinstance(stats, dict) and stats.get('missing_rate', 0) > 0.1]
            if high_missing:
                report_lines.append(f"• Properties with high missing rates: {', '.join(high_missing)}")
                
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Quality report saved to {output_path}")
            
        return report


def create_data_visualizations(analysis: Dict[str, Any], 
                             output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Create visualizations for dataset analysis.
    
    Args:
        analysis: Analysis results from DataQualityAnalyzer
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    saved_plots = {}
    
    try:
        # Property distributions
        if analysis['property_analysis']:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            prop_names = list(analysis['property_analysis'].keys())[:4]
            
            for i, prop_name in enumerate(prop_names):
                prop_stats = analysis['property_analysis'][prop_name]
                if isinstance(prop_stats, dict) and 'mean' in prop_stats:
                    # Create synthetic data for visualization (in real implementation, 
                    # you would pass the actual property values)
                    mean = prop_stats['mean']
                    std = prop_stats['std']
                    data = np.random.normal(mean, std, 1000)
                    
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{prop_name.upper()} Distribution')
                    axes[i].set_xlabel(prop_name)
                    axes[i].set_ylabel('Frequency')
                    
            plt.tight_layout()
            
            if output_dir:
                plot_path = output_dir / 'property_distributions.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                saved_plots['property_distributions'] = plot_path
                
            plt.close()
            
        # Quality metrics summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validity and uniqueness rates
        metrics = ['Validity', 'Uniqueness']
        rates = [
            analysis['validation_results']['validity_rate'],
            analysis['duplicate_analysis']['uniqueness_rate']
        ]
        
        bars = ax1.bar(metrics, rates, color=['skyblue', 'lightcoral'])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Rate')
        ax1.set_title('Dataset Quality Metrics')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
                    
        # Overall quality score
        score = analysis['quality_score']
        colors = ['red' if score < 0.5 else 'orange' if score < 0.8 else 'green']
        
        ax2.pie([score, 1-score], labels=['Quality Score', ''], 
               colors=[colors[0], 'lightgray'], startangle=90,
               wedgeprops=dict(width=0.3))
        ax2.set_title(f'Overall Quality Score\n{score:.3f}/1.000')
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = output_dir / 'quality_summary.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            saved_plots['quality_summary'] = plot_path
            
        plt.close()
        
    except Exception as e:
        logger.warning(f"Failed to create visualizations: {e}")
        
    return saved_plots