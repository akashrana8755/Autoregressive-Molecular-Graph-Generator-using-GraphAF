#!/usr/bin/env python3
"""
Performance Benchmarking for MolecuGen

This script provides comprehensive benchmarking tools for evaluating
the performance of molecular generation models across different metrics.
"""

import sys
import os
sys.path.append('..')

import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import psutil
import gc

# MolecuGen imports
from src.generate.molecular_generator import MolecularGenerator
from src.evaluate.molecular_evaluator import MolecularEvaluator
from src.generate.constraint_filter import ConstraintFilter
from src.training.trainer import Trainer
from src.data.molecular_dataset import MolecularDataset

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for MolecuGen.
    """
    
    def __init__(self, output_dir: str = "benchmarks"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def benchmark_generation_speed(self, generator: MolecularGenerator,
                                 batch_sizes: List[int] = None,
                                 num_molecules: int = 100,
                                 temperatures: List[float] = None,
                                 num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark molecular generation speed across different parameters.
        
        Args:
            generator: MolecularGenerator instance
            batch_sizes: List of batch sizes to test
            num_molecules: Number of molecules to generate per test
            temperatures: List of temperatures to test
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        if temperatures is None:
            temperatures = [0.5, 1.0, 1.5, 2.0]
            
        print("Benchmarking generation speed...")
        results = []
        
        for batch_size in batch_sizes:
            for temperature in temperatures:
                print(f"Testing batch_size={batch_size}, temperature={temperature}")
                
                run_times = []
                molecules_generated = []
                valid_molecules = []
                
                for run in range(num_runs):
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Measure generation time
                    start_time = time.time()
                    
                    try:
                        molecules = generator.generate(
                            num_molecules=num_molecules,
                            batch_size=batch_size,
                            temperature=temperature,
                            max_attempts=3
                        )
                        
                        end_time = time.time()
                        run_time = end_time - start_time
                        
                        # Validate molecules
                        valid_count = sum(1 for mol in molecules 
                                        if generator.smiles_processor.validate_molecule(mol))
                        
                        run_times.append(run_time)
                        molecules_generated.append(len(molecules))
                        valid_molecules.append(valid_count)
                        
                    except Exception as e:
                        print(f"Error in run {run}: {e}")
                        continue
                
                if run_times:
                    avg_time = np.mean(run_times)
                    std_time = np.std(run_times)
                    avg_generated = np.mean(molecules_generated)
                    avg_valid = np.mean(valid_molecules)
                    
                    results.append({
                        'batch_size': batch_size,
                        'temperature': temperature,
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'avg_generated': avg_generated,
                        'avg_valid': avg_valid,
                        'validity_rate': avg_valid / avg_generated if avg_generated > 0 else 0,
                        'molecules_per_second': avg_generated / avg_time if avg_time > 0 else 0,
                        'valid_per_second': avg_valid / avg_time if avg_time > 0 else 0
                    })
        
        self.results['generation_speed'] = results
        return results
    
    def benchmark_memory_usage(self, generator: MolecularGenerator,
                             batch_sizes: List[int] = None,
                             num_molecules: int = 50) -> Dict[str, Any]:
        """
        Benchmark memory usage during generation.
        
        Args:
            generator: MolecularGenerator instance
            batch_sizes: List of batch sizes to test
            num_molecules: Number of molecules to generate per test
            
        Returns:
            Dictionary containing memory usage results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
            
        print("Benchmarking memory usage...")
        results = []
        
        for batch_size in batch_sizes:
            print(f"Testing batch_size={batch_size}")
            
            # Clear memory before test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Measure memory before generation
            process = psutil.Process()
            cpu_memory_before = process.memory_info().rss / 1024**2  # MB
            
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                gpu_memory_before = 0
            
            try:
                # Generate molecules
                molecules = generator.generate(
                    num_molecules=num_molecules,
                    batch_size=batch_size,
                    temperature=1.0,
                    max_attempts=3
                )
                
                # Measure memory after generation
                cpu_memory_after = process.memory_info().rss / 1024**2  # MB
                
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                    gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
                else:
                    gpu_memory_after = 0
                    gpu_memory_peak = 0
                
                results.append({
                    'batch_size': batch_size,
                    'molecules_generated': len(molecules),
                    'cpu_memory_before': cpu_memory_before,
                    'cpu_memory_after': cpu_memory_after,
                    'cpu_memory_increase': cpu_memory_after - cpu_memory_before,
                    'gpu_memory_before': gpu_memory_before,
                    'gpu_memory_after': gpu_memory_after,
                    'gpu_memory_increase': gpu_memory_after - gpu_memory_before,
                    'gpu_memory_peak': gpu_memory_peak,
                    'memory_per_molecule': (gpu_memory_peak - gpu_memory_before) / len(molecules) if molecules else 0
                })
                
            except Exception as e:
                print(f"Error testing batch_size {batch_size}: {e}")
                continue
        
        self.results['memory_usage'] = results
        return results
    
    def benchmark_quality_metrics(self, generator: MolecularGenerator,
                                 reference_molecules: List[str],
                                 generation_configs: List[Dict] = None,
                                 num_molecules: int = 1000) -> Dict[str, Any]:
        """
        Benchmark generation quality across different configurations.
        
        Args:
            generator: MolecularGenerator instance
            reference_molecules: Reference molecules for comparison
            generation_configs: List of generation configurations to test
            num_molecules: Number of molecules to generate per config
            
        Returns:
            Dictionary containing quality benchmark results
        """
        if generation_configs is None:
            generation_configs = [
                {'temperature': 0.5, 'name': 'Conservative'},
                {'temperature': 1.0, 'name': 'Balanced'},
                {'temperature': 1.5, 'name': 'Diverse'},
                {'temperature': 2.0, 'name': 'Exploratory'}
            ]
        
        print("Benchmarking generation quality...")
        evaluator = MolecularEvaluator(reference_molecules)
        constraint_filter = ConstraintFilter()
        
        results = []
        
        for config in generation_configs:
            print(f"Testing configuration: {config['name']}")
            
            try:
                # Generate molecules
                molecules = generator.generate(
                    num_molecules=num_molecules,
                    temperature=config['temperature'],
                    batch_size=32,
                    max_attempts=5
                )
                
                if not molecules:
                    print(f"No molecules generated for {config['name']}")
                    continue
                
                # Basic evaluation metrics
                basic_metrics = evaluator.evaluate(molecules)
                
                # Drug-likeness metrics
                drug_metrics = evaluator.compute_drug_likeness_metrics(molecules)
                
                # Property distributions
                prop_dist = evaluator.compute_property_distributions(molecules)
                
                # Constraint statistics
                constraint_stats = constraint_filter.get_comprehensive_statistics(molecules)
                
                # Diversity metrics
                diversity_metrics = evaluator.compute_diversity_metrics(molecules)
                
                # Compile results
                result = {
                    'config_name': config['name'],
                    'temperature': config['temperature'],
                    'molecules_generated': len(molecules),
                    
                    # Basic metrics
                    'validity': basic_metrics['validity'],
                    'uniqueness': basic_metrics['uniqueness'],
                    'novelty': basic_metrics.get('novelty', 0),
                    
                    # Drug-likeness
                    'mean_qed': drug_metrics['mean_qed'],
                    'median_qed': drug_metrics['median_qed'],
                    'lipinski_pass_rate': drug_metrics['lipinski_pass_rate'],
                    
                    # Property statistics
                    'mean_mw': np.mean(prop_dist['molecular_weight']) if 'molecular_weight' in prop_dist else 0,
                    'std_mw': np.std(prop_dist['molecular_weight']) if 'molecular_weight' in prop_dist else 0,
                    'mean_logp': np.mean(prop_dist['logp']) if 'logp' in prop_dist else 0,
                    'std_logp': np.std(prop_dist['logp']) if 'logp' in prop_dist else 0,
                    
                    # Diversity
                    'diversity_score': diversity_metrics['diversity_score'],
                    
                    # Constraint satisfaction
                    'constraint_pass_rate': constraint_stats.get('combined_pass_rate', 0)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error testing {config['name']}: {e}")
                continue
        
        self.results['quality_metrics'] = results
        return results
    
    def benchmark_constraint_satisfaction(self, generator: MolecularGenerator,
                                        constraint_configs: List[Dict] = None,
                                        num_molecules: int = 500) -> Dict[str, Any]:
        """
        Benchmark constraint satisfaction rates.
        
        Args:
            generator: MolecularGenerator instance
            constraint_configs: List of constraint configurations to test
            num_molecules: Number of molecules to generate per config
            
        Returns:
            Dictionary containing constraint satisfaction results
        """
        if constraint_configs is None:
            constraint_configs = [
                {
                    'name': 'Lipinski Only',
                    'constraints': {'lipinski': True}
                },
                {
                    'name': 'QED > 0.5',
                    'constraints': {'qed_threshold': 0.5}
                },
                {
                    'name': 'Drug-like',
                    'constraints': {'lipinski': True, 'qed_threshold': 0.5}
                },
                {
                    'name': 'Strict Drug-like',
                    'constraints': {
                        'lipinski': True,
                        'qed_threshold': 0.6,
                        'mw_range': [200, 400],
                        'logp_range': [0, 3]
                    }
                }
            ]
        
        print("Benchmarking constraint satisfaction...")
        results = []
        
        for config in constraint_configs:
            print(f"Testing constraints: {config['name']}")
            
            try:
                start_time = time.time()
                
                # Generate with constraints
                molecules = generator.generate_with_constraints(
                    num_molecules=num_molecules,
                    constraints=config['constraints'],
                    temperature=1.0,
                    max_attempts=10,
                    iterative_filtering=True
                )
                
                generation_time = time.time() - start_time
                
                # Analyze results
                constraint_filter = ConstraintFilter()
                
                # Verify constraint satisfaction
                satisfied_count = 0
                for mol in molecules:
                    if generator._molecule_passes_constraints(mol, config['constraints']):
                        satisfied_count += 1
                
                satisfaction_rate = satisfied_count / len(molecules) if molecules else 0
                
                # Get detailed statistics
                stats = constraint_filter.get_comprehensive_statistics(molecules)
                
                result = {
                    'config_name': config['name'],
                    'constraints': str(config['constraints']),
                    'molecules_requested': num_molecules,
                    'molecules_generated': len(molecules),
                    'generation_rate': len(molecules) / num_molecules,
                    'satisfaction_rate': satisfaction_rate,
                    'generation_time': generation_time,
                    'molecules_per_second': len(molecules) / generation_time if generation_time > 0 else 0,
                    'lipinski_pass_rate': stats.get('all_rules_pass_rate', 0),
                    'mean_qed': stats.get('qed_statistics', {}).get('mean_qed', 0)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error testing {config['name']}: {e}")
                continue
        
        self.results['constraint_satisfaction'] = results
        return results
    
    def benchmark_scalability(self, generator: MolecularGenerator,
                            molecule_counts: List[int] = None,
                            batch_size: int = 32) -> Dict[str, Any]:
        """
        Benchmark scalability for large-scale generation.
        
        Args:
            generator: MolecularGenerator instance
            molecule_counts: List of molecule counts to test
            batch_size: Batch size to use
            
        Returns:
            Dictionary containing scalability results
        """
        if molecule_counts is None:
            molecule_counts = [100, 500, 1000, 2000, 5000]
        
        print("Benchmarking scalability...")
        results = []
        
        for count in molecule_counts:
            print(f"Testing generation of {count} molecules...")
            
            try:
                start_time = time.time()
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate molecules
                molecules = generator.generate(
                    num_molecules=count,
                    batch_size=batch_size,
                    temperature=1.0,
                    max_attempts=3
                )
                
                generation_time = time.time() - start_time
                
                # Validate molecules
                valid_count = sum(1 for mol in molecules 
                                if generator.smiles_processor.validate_molecule(mol))
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                else:
                    memory_used = 0
                
                result = {
                    'molecule_count': count,
                    'molecules_generated': len(molecules),
                    'valid_molecules': valid_count,
                    'generation_time': generation_time,
                    'molecules_per_second': len(molecules) / generation_time if generation_time > 0 else 0,
                    'validity_rate': valid_count / len(molecules) if molecules else 0,
                    'memory_used_mb': memory_used,
                    'memory_per_molecule': memory_used / len(molecules) if molecules else 0
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error testing {count} molecules: {e}")
                continue
        
        self.results['scalability'] = results
        return results
    
    def plot_benchmark_results(self, save_plots: bool = True) -> None:
        """
        Create comprehensive plots of all benchmark results.
        
        Args:
            save_plots: Whether to save plots to files
        """
        print("Creating benchmark plots...")
        
        # Generation speed plots
        if 'generation_speed' in self.results:
            self._plot_generation_speed(save_plots)
        
        # Memory usage plots
        if 'memory_usage' in self.results:
            self._plot_memory_usage(save_plots)
        
        # Quality metrics plots
        if 'quality_metrics' in self.results:
            self._plot_quality_metrics(save_plots)
        
        # Constraint satisfaction plots
        if 'constraint_satisfaction' in self.results:
            self._plot_constraint_satisfaction(save_plots)
        
        # Scalability plots
        if 'scalability' in self.results:
            self._plot_scalability(save_plots)
    
    def _plot_generation_speed(self, save_plots: bool) -> None:
        """Plot generation speed benchmark results."""
        df = pd.DataFrame(self.results['generation_speed'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Speed vs batch size
        for temp in df['temperature'].unique():
            temp_data = df[df['temperature'] == temp]
            axes[0, 0].plot(temp_data['batch_size'], temp_data['molecules_per_second'],
                           marker='o', label=f'T={temp}')
        
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Molecules per Second')
        axes[0, 0].set_title('Generation Speed vs Batch Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validity vs temperature
        for batch_size in sorted(df['batch_size'].unique()):
            batch_data = df[df['batch_size'] == batch_size]
            axes[0, 1].plot(batch_data['temperature'], batch_data['validity_rate'],
                           marker='s', label=f'Batch={batch_size}')
        
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Validity Rate')
        axes[0, 1].set_title('Validity Rate vs Temperature')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Heatmap of generation speed
        pivot_speed = df.pivot(index='temperature', columns='batch_size', values='molecules_per_second')
        sns.heatmap(pivot_speed, annot=True, fmt='.1f', ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title('Generation Speed Heatmap')
        
        # Heatmap of validity rate
        pivot_validity = df.pivot(index='temperature', columns='batch_size', values='validity_rate')
        sns.heatmap(pivot_validity, annot=True, fmt='.2f', ax=axes[1, 1], cmap='RdYlGn')
        axes[1, 1].set_title('Validity Rate Heatmap')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'generation_speed_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_memory_usage(self, save_plots: bool) -> None:
        """Plot memory usage benchmark results."""
        df = pd.DataFrame(self.results['memory_usage'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Memory usage vs batch size
        axes[0].plot(df['batch_size'], df['gpu_memory_peak'], marker='o', label='GPU Peak')
        axes[0].plot(df['batch_size'], df['cpu_memory_increase'], marker='s', label='CPU Increase')
        axes[0].set_xlabel('Batch Size')
        axes[0].set_ylabel('Memory Usage (MB)')
        axes[0].set_title('Memory Usage vs Batch Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Memory per molecule
        axes[1].bar(df['batch_size'], df['memory_per_molecule'], alpha=0.7)
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Memory per Molecule (MB)')
        axes[1].set_title('Memory Efficiency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'memory_usage_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_quality_metrics(self, save_plots: bool) -> None:
        """Plot quality metrics benchmark results."""
        df = pd.DataFrame(self.results['quality_metrics'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Validity and uniqueness
        x_pos = np.arange(len(df))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, df['validity'], width, label='Validity', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, df['uniqueness'], width, label='Uniqueness', alpha=0.7)
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_title('Validity and Uniqueness')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(df['config_name'], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # QED scores
        axes[0, 1].bar(df['config_name'], df['mean_qed'], alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('Mean QED Score')
        axes[0, 1].set_title('Drug-likeness (QED)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Lipinski compliance
        axes[0, 2].bar(df['config_name'], df['lipinski_pass_rate'], alpha=0.7, color='orange')
        axes[0, 2].set_xlabel('Configuration')
        axes[0, 2].set_ylabel('Lipinski Pass Rate')
        axes[0, 2].set_title('Lipinski Compliance')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Molecular weight
        axes[1, 0].bar(df['config_name'], df['mean_mw'], alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Configuration')
        axes[1, 0].set_ylabel('Mean Molecular Weight (Da)')
        axes[1, 0].set_title('Molecular Weight')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # LogP
        axes[1, 1].bar(df['config_name'], df['mean_logp'], alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Configuration')
        axes[1, 1].set_ylabel('Mean LogP')
        axes[1, 1].set_title('Lipophilicity (LogP)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Diversity
        axes[1, 2].bar(df['config_name'], df['diversity_score'], alpha=0.7, color='cyan')
        axes[1, 2].set_xlabel('Configuration')
        axes[1, 2].set_ylabel('Diversity Score')
        axes[1, 2].set_title('Molecular Diversity')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'quality_metrics_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_constraint_satisfaction(self, save_plots: bool) -> None:
        """Plot constraint satisfaction benchmark results."""
        df = pd.DataFrame(self.results['constraint_satisfaction'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Generation and satisfaction rates
        x_pos = np.arange(len(df))
        width = 0.35
        
        axes[0].bar(x_pos - width/2, df['generation_rate'], width, 
                   label='Generation Rate', alpha=0.7)
        axes[0].bar(x_pos + width/2, df['satisfaction_rate'], width,
                   label='Satisfaction Rate', alpha=0.7)
        axes[0].set_xlabel('Constraint Configuration')
        axes[0].set_ylabel('Rate')
        axes[0].set_title('Constraint Satisfaction Performance')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(df['config_name'], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Generation speed
        axes[1].bar(df['config_name'], df['molecules_per_second'], alpha=0.7, color='green')
        axes[1].set_xlabel('Constraint Configuration')
        axes[1].set_ylabel('Molecules per Second')
        axes[1].set_title('Constrained Generation Speed')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'constraint_satisfaction_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_scalability(self, save_plots: bool) -> None:
        """Plot scalability benchmark results."""
        df = pd.DataFrame(self.results['scalability'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Generation speed vs scale
        axes[0].plot(df['molecule_count'], df['molecules_per_second'], marker='o', linewidth=2)
        axes[0].set_xlabel('Number of Molecules')
        axes[0].set_ylabel('Molecules per Second')
        axes[0].set_title('Generation Speed vs Scale')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Memory usage vs scale
        axes[1].plot(df['molecule_count'], df['memory_per_molecule'], marker='s', linewidth=2, color='red')
        axes[1].set_xlabel('Number of Molecules')
        axes[1].set_ylabel('Memory per Molecule (MB)')
        axes[1].set_title('Memory Efficiency vs Scale')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'scalability_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save all benchmark results to JSON file."""
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        clean_results = json.loads(json.dumps(self.results, default=convert_numpy))
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Benchmark results saved to {self.output_dir / filename}")
    
    def generate_report(self, filename: str = "benchmark_report.txt") -> None:
        """Generate a comprehensive text report of benchmark results."""
        report_lines = []
        report_lines.append("MolecuGen Performance Benchmark Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Generation speed summary
        if 'generation_speed' in self.results:
            df = pd.DataFrame(self.results['generation_speed'])
            report_lines.append("Generation Speed Benchmark:")
            report_lines.append(f"  Best speed: {df['molecules_per_second'].max():.2f} molecules/sec")
            report_lines.append(f"  Average speed: {df['molecules_per_second'].mean():.2f} molecules/sec")
            report_lines.append(f"  Best validity: {df['validity_rate'].max():.2%}")
            report_lines.append("")
        
        # Memory usage summary
        if 'memory_usage' in self.results:
            df = pd.DataFrame(self.results['memory_usage'])
            report_lines.append("Memory Usage Benchmark:")
            report_lines.append(f"  Peak GPU memory: {df['gpu_memory_peak'].max():.1f} MB")
            report_lines.append(f"  Most efficient: {df['memory_per_molecule'].min():.3f} MB/molecule")
            report_lines.append("")
        
        # Quality metrics summary
        if 'quality_metrics' in self.results:
            df = pd.DataFrame(self.results['quality_metrics'])
            report_lines.append("Quality Metrics Benchmark:")
            report_lines.append(f"  Best validity: {df['validity'].max():.2%}")
            report_lines.append(f"  Best uniqueness: {df['uniqueness'].max():.2%}")
            report_lines.append(f"  Best QED score: {df['mean_qed'].max():.3f}")
            report_lines.append(f"  Best Lipinski compliance: {df['lipinski_pass_rate'].max():.2%}")
            report_lines.append("")
        
        # Constraint satisfaction summary
        if 'constraint_satisfaction' in self.results:
            df = pd.DataFrame(self.results['constraint_satisfaction'])
            report_lines.append("Constraint Satisfaction Benchmark:")
            report_lines.append(f"  Best generation rate: {df['generation_rate'].max():.2%}")
            report_lines.append(f"  Best satisfaction rate: {df['satisfaction_rate'].max():.2%}")
            report_lines.append("")
        
        # Scalability summary
        if 'scalability' in self.results:
            df = pd.DataFrame(self.results['scalability'])
            report_lines.append("Scalability Benchmark:")
            report_lines.append(f"  Largest scale tested: {df['molecule_count'].max()} molecules")
            report_lines.append(f"  Speed at largest scale: {df.iloc[-1]['molecules_per_second']:.2f} molecules/sec")
            report_lines.append("")
        
        # Write report
        with open(self.output_dir / filename, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Benchmark report saved to {self.output_dir / filename}")


def main():
    """
    Main function demonstrating benchmark usage.
    """
    print("MolecuGen Performance Benchmarking")
    print("=" * 40)
    
    # Note: This is a demonstration. In practice, you would load a trained model:
    # generator = MolecularGenerator.from_checkpoint("path/to/model.pt")
    
    print("This is a demonstration script.")
    print("To run actual benchmarks, you need:")
    print("1. A trained MolecularGenerator model")
    print("2. Reference molecules for comparison")
    print("")
    print("Example usage:")
    print("""
    # Load trained model
    generator = MolecularGenerator.from_checkpoint("model.pt")
    
    # Load reference molecules
    with open("reference_molecules.smi", "r") as f:
        reference_molecules = [line.strip() for line in f]
    
    # Initialize benchmark suite
    benchmark = PerformanceBenchmark("benchmark_results")
    
    # Run benchmarks
    benchmark.benchmark_generation_speed(generator)
    benchmark.benchmark_memory_usage(generator)
    benchmark.benchmark_quality_metrics(generator, reference_molecules)
    benchmark.benchmark_constraint_satisfaction(generator)
    benchmark.benchmark_scalability(generator)
    
    # Generate plots and reports
    benchmark.plot_benchmark_results()
    benchmark.save_results()
    benchmark.generate_report()
    """)


if __name__ == "__main__":
    main()