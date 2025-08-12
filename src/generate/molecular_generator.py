"""
Molecular generation engine for drug-like molecule generation.

This module provides the main interface for generating molecular structures
using trained generative models with constraint filtering and validation.
"""

import logging
from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
import torch
from torch_geometric.data import Data, Batch
import numpy as np

from ..models.base_model import BaseGenerativeModel
from ..models.graph_diffusion import GraphDiffusion
from ..models.graph_af import GraphAF
from ..data.smiles_processor import SMILESProcessor
from .constraint_filter import ConstraintFilter

logger = logging.getLogger(__name__)


class MolecularGenerator:
    """
    Main interface for molecular generation with constraint filtering.
    
    This class provides batch generation capabilities with configurable parameters,
    graph-to-SMILES conversion with validation, and constraint-aware generation.
    """
    
    def __init__(self, 
                 model: BaseGenerativeModel,
                 smiles_processor: Optional[SMILESProcessor] = None,
                 constraint_filter: Optional[ConstraintFilter] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the molecular generator.
        
        Args:
            model: Trained generative model (GraphDiffusion or GraphAF)
            smiles_processor: SMILES processor for graph-to-SMILES conversion
            constraint_filter: Constraint filter for drug-likeness filtering
            device: Device to run generation on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize SMILES processor
        self.smiles_processor = smiles_processor or SMILESProcessor(
            add_self_loops=False,
            explicit_hydrogens=False,
            sanitize=True
        )
        
        # Initialize constraint filter
        self.constraint_filter = constraint_filter or ConstraintFilter()
        
        # Generation statistics
        self.generation_stats = {
            'total_generated': 0,
            'valid_molecules': 0,
            'constraint_passed': 0,
            'unique_molecules': 0
        }
        
    @classmethod
    def from_checkpoint(cls, 
                       checkpoint_path: Union[str, Path],
                       model_class: Optional[type] = None,
                       device: Optional[torch.device] = None,
                       **kwargs) -> 'MolecularGenerator':
        """
        Create generator from a saved model checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            model_class: Model class (GraphDiffusion or GraphAF)
            device: Device to load model on
            **kwargs: Additional arguments for generator initialization
            
        Returns:
            Initialized MolecularGenerator instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Determine model class from checkpoint if not provided
        if model_class is None:
            model_name = checkpoint_data.get('model_name', '')
            if 'GraphDiffusion' in model_name:
                model_class = GraphDiffusion
            elif 'GraphAF' in model_name:
                model_class = GraphAF
            else:
                raise ValueError(f"Cannot determine model class from checkpoint: {model_name}")
        
        # Create model instance
        model = model_class(checkpoint_data['config'])
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model.to(device)
        model.eval()
        
        return cls(model=model, device=device, **kwargs)
    
    def generate(self, 
                 num_molecules: int,
                 max_nodes: Optional[int] = None,
                 temperature: float = 1.0,
                 batch_size: int = 32,
                 max_attempts: int = 10,
                 return_graphs: bool = False,
                 **kwargs) -> Union[List[str], Tuple[List[str], List[Data]]]:
        """
        Generate molecular structures as SMILES strings.
        
        Args:
            num_molecules: Number of molecules to generate
            max_nodes: Maximum number of nodes per molecule
            temperature: Sampling temperature (higher = more diverse)
            batch_size: Batch size for generation
            max_attempts: Maximum attempts per molecule
            return_graphs: Whether to return molecular graphs along with SMILES
            **kwargs: Additional sampling parameters
            
        Returns:
            List of SMILES strings, optionally with molecular graphs
        """
        logger.info(f"Generating {num_molecules} molecules with max_nodes={max_nodes}, "
                   f"temperature={temperature}")
        
        generated_smiles = []
        generated_graphs = []
        
        self.model.eval()
        with torch.no_grad():
            remaining = num_molecules
            attempts = 0
            
            while remaining > 0 and attempts < max_attempts:
                attempts += 1
                current_batch_size = min(batch_size, remaining)
                
                # Generate molecular graphs
                try:
                    graphs = self.model.sample(
                        num_samples=current_batch_size,
                        max_nodes=max_nodes,
                        temperature=temperature,
                        **kwargs
                    )
                    
                    # Convert graphs to SMILES
                    batch_smiles, batch_graphs = self._graphs_to_smiles(graphs)
                    
                    # Update statistics
                    self.generation_stats['total_generated'] += len(graphs)
                    self.generation_stats['valid_molecules'] += len(batch_smiles)
                    
                    generated_smiles.extend(batch_smiles)
                    generated_graphs.extend(batch_graphs)
                    
                    remaining -= len(batch_smiles)
                    
                    logger.debug(f"Batch {attempts}: Generated {len(batch_smiles)}/{current_batch_size} "
                               f"valid molecules. Remaining: {remaining}")
                    
                except Exception as e:
                    logger.warning(f"Generation attempt {attempts} failed: {str(e)}")
                    continue
        
        # Remove duplicates and keep only unique molecules
        unique_smiles, unique_graphs = self._remove_duplicates(generated_smiles, generated_graphs)
        self.generation_stats['unique_molecules'] = len(unique_smiles)
        
        # Truncate to requested number
        final_smiles = unique_smiles[:num_molecules]
        final_graphs = unique_graphs[:num_molecules] if unique_graphs else []
        
        logger.info(f"Generated {len(final_smiles)} unique molecules "
                   f"(requested: {num_molecules}, attempts: {attempts})")
        
        if return_graphs:
            return final_smiles, final_graphs
        return final_smiles
    
    def generate_batch(self, 
                      batch_size: int,
                      **kwargs) -> Tuple[List[str], List[Data]]:
        """
        Generate a single batch of molecules.
        
        Args:
            batch_size: Number of molecules to generate in batch
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (SMILES list, molecular graphs list)
        """
        return self.generate(
            num_molecules=batch_size,
            batch_size=batch_size,
            max_attempts=1,
            return_graphs=True,
            **kwargs
        )
    
    def _graphs_to_smiles(self, graphs: List[Data]) -> Tuple[List[str], List[Data]]:
        """
        Convert molecular graphs to SMILES strings with validation.
        
        Args:
            graphs: List of molecular graphs
            
        Returns:
            Tuple of (valid SMILES list, corresponding graphs list)
        """
        valid_smiles = []
        valid_graphs = []
        
        for graph in graphs:
            try:
                # Convert graph to SMILES
                smiles = self.smiles_processor.graph_to_smiles(graph)
                
                if smiles is not None:
                    # Validate the SMILES
                    if self.smiles_processor.validate_molecule(smiles):
                        # Canonicalize SMILES
                        canonical_smiles = self.smiles_processor.sanitize_smiles(smiles)
                        if canonical_smiles is not None:
                            valid_smiles.append(canonical_smiles)
                            valid_graphs.append(graph)
                        else:
                            logger.debug("Failed to canonicalize SMILES")
                    else:
                        logger.debug(f"Invalid SMILES generated: {smiles}")
                else:
                    logger.debug("Failed to convert graph to SMILES")
                    
            except Exception as e:
                logger.debug(f"Error converting graph to SMILES: {str(e)}")
                continue
        
        return valid_smiles, valid_graphs
    
    def _remove_duplicates(self, 
                          smiles_list: List[str], 
                          graphs_list: List[Data]) -> Tuple[List[str], List[Data]]:
        """
        Remove duplicate SMILES while preserving order.
        
        Args:
            smiles_list: List of SMILES strings
            graphs_list: Corresponding list of molecular graphs
            
        Returns:
            Tuple of (unique SMILES, corresponding unique graphs)
        """
        seen = set()
        unique_smiles = []
        unique_graphs = []
        
        for i, smiles in enumerate(smiles_list):
            if smiles not in seen:
                seen.add(smiles)
                unique_smiles.append(smiles)
                if i < len(graphs_list):
                    unique_graphs.append(graphs_list[i])
        
        return unique_smiles, unique_graphs
    
    def validate_molecules(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Validate a list of molecules and return statistics.
        
        Args:
            smiles_list: List of SMILES strings to validate
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'total_molecules': len(smiles_list),
            'valid_molecules': 0,
            'invalid_molecules': 0,
            'validity_rate': 0.0,
            'unique_molecules': 0,
            'uniqueness_rate': 0.0
        }
        
        if not smiles_list:
            return stats
        
        # Check validity
        valid_count = 0
        for smiles in smiles_list:
            if self.smiles_processor.validate_molecule(smiles):
                valid_count += 1
        
        stats['valid_molecules'] = valid_count
        stats['invalid_molecules'] = len(smiles_list) - valid_count
        stats['validity_rate'] = valid_count / len(smiles_list)
        
        # Check uniqueness
        unique_smiles = set(smiles_list)
        stats['unique_molecules'] = len(unique_smiles)
        stats['uniqueness_rate'] = len(unique_smiles) / len(smiles_list)
        
        return stats
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get current generation statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        stats = self.generation_stats.copy()
        
        if stats['total_generated'] > 0:
            stats['validity_rate'] = stats['valid_molecules'] / stats['total_generated']
        else:
            stats['validity_rate'] = 0.0
            
        if stats['valid_molecules'] > 0:
            stats['uniqueness_rate'] = stats['unique_molecules'] / stats['valid_molecules']
        else:
            stats['uniqueness_rate'] = 0.0
            
        return stats
    
    def reset_statistics(self) -> None:
        """Reset generation statistics."""
        self.generation_stats = {
            'total_generated': 0,
            'valid_molecules': 0,
            'constraint_passed': 0,
            'unique_molecules': 0
        }
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the default sampling temperature.
        
        Args:
            temperature: Sampling temperature (higher = more diverse)
        """
        self.temperature = temperature
        logger.info(f"Set sampling temperature to {temperature}")
    
    def generate_with_constraints(self,
                                 num_molecules: int,
                                 constraints: Optional[Dict[str, Any]] = None,
                                 max_nodes: Optional[int] = None,
                                 temperature: float = 1.0,
                                 batch_size: int = 32,
                                 max_attempts: int = 20,
                                 iterative_filtering: bool = True,
                                 return_all: bool = False,
                                 **kwargs) -> Union[List[str], Dict[str, List[str]]]:
        """
        Generate molecules with constraint filtering.
        
        Args:
            num_molecules: Number of constraint-passing molecules to generate
            constraints: Dictionary of constraints to apply
            max_nodes: Maximum number of nodes per molecule
            temperature: Sampling temperature
            batch_size: Batch size for generation
            max_attempts: Maximum generation attempts
            iterative_filtering: Whether to filter during generation or after
            return_all: Whether to return all generated molecules (passed and failed)
            **kwargs: Additional generation parameters
            
        Returns:
            List of constraint-passing SMILES, or dict with all results if return_all=True
        """
        logger.info(f"Generating {num_molecules} molecules with constraints: {constraints}")
        
        if constraints is None:
            constraints = {}
        
        constraint_passed = []
        constraint_failed = []
        all_generated = []
        
        self.model.eval()
        with torch.no_grad():
            attempts = 0
            
            while len(constraint_passed) < num_molecules and attempts < max_attempts:
                attempts += 1
                
                # Generate a batch of molecules
                current_batch_size = min(batch_size, num_molecules * 2)  # Generate extra for filtering
                
                try:
                    batch_smiles = self.generate(
                        num_molecules=current_batch_size,
                        max_nodes=max_nodes,
                        temperature=temperature,
                        batch_size=current_batch_size,
                        max_attempts=1,
                        return_graphs=False,
                        **kwargs
                    )
                    
                    all_generated.extend(batch_smiles)
                    
                    # Apply constraints
                    if iterative_filtering:
                        passed, failed = self._apply_constraints(batch_smiles, constraints)
                        constraint_passed.extend(passed)
                        constraint_failed.extend(failed)
                        
                        # Update statistics
                        self.generation_stats['constraint_passed'] += len(passed)
                        
                        logger.debug(f"Attempt {attempts}: {len(passed)}/{len(batch_smiles)} "
                                   f"passed constraints. Total passed: {len(constraint_passed)}")
                    else:
                        constraint_passed.extend(batch_smiles)
                        
                except Exception as e:
                    logger.warning(f"Constraint generation attempt {attempts} failed: {str(e)}")
                    continue
        
        # Apply constraints to all generated molecules if not done iteratively
        if not iterative_filtering:
            constraint_passed, constraint_failed = self._apply_constraints(all_generated, constraints)
            self.generation_stats['constraint_passed'] = len(constraint_passed)
        
        # Truncate to requested number
        final_passed = constraint_passed[:num_molecules]
        
        logger.info(f"Generated {len(final_passed)} constraint-passing molecules "
                   f"(requested: {num_molecules}, total generated: {len(all_generated)}, "
                   f"attempts: {attempts})")
        
        if return_all:
            return {
                'constraint_passed': final_passed,
                'constraint_failed': constraint_failed,
                'all_generated': all_generated
            }
        
        return final_passed
    
    def generate_with_properties(self,
                                target_properties: Dict[str, float],
                                num_molecules: int,
                                property_predictor: Optional[Any] = None,
                                tolerance: float = 0.1,
                                max_nodes: Optional[int] = None,
                                temperature: float = 1.0,
                                batch_size: int = 32,
                                max_attempts: int = 30,
                                **kwargs) -> List[str]:
        """
        Generate molecules targeting specific properties.
        
        Args:
            target_properties: Dictionary of target property values
            num_molecules: Number of molecules to generate
            property_predictor: Property prediction model (optional)
            tolerance: Tolerance for property matching
            max_nodes: Maximum number of nodes per molecule
            temperature: Sampling temperature
            batch_size: Batch size for generation
            max_attempts: Maximum generation attempts
            **kwargs: Additional generation parameters
            
        Returns:
            List of SMILES strings with properties close to targets
        """
        logger.info(f"Generating {num_molecules} molecules with target properties: {target_properties}")
        
        if property_predictor is None:
            logger.warning("No property predictor provided, using constraint-based generation")
            # Convert properties to constraints
            constraints = self._properties_to_constraints(target_properties, tolerance)
            return self.generate_with_constraints(
                num_molecules=num_molecules,
                constraints=constraints,
                max_nodes=max_nodes,
                temperature=temperature,
                batch_size=batch_size,
                max_attempts=max_attempts,
                **kwargs
            )
        
        property_matched = []
        
        self.model.eval()
        with torch.no_grad():
            attempts = 0
            
            while len(property_matched) < num_molecules and attempts < max_attempts:
                attempts += 1
                
                # Generate a batch of molecules
                current_batch_size = min(batch_size, num_molecules * 3)  # Generate extra for filtering
                
                try:
                    batch_smiles = self.generate(
                        num_molecules=current_batch_size,
                        max_nodes=max_nodes,
                        temperature=temperature,
                        batch_size=current_batch_size,
                        max_attempts=1,
                        return_graphs=False,
                        **kwargs
                    )
                    
                    # Predict properties and filter
                    matched = self._filter_by_properties(
                        batch_smiles, target_properties, property_predictor, tolerance
                    )
                    property_matched.extend(matched)
                    
                    logger.debug(f"Attempt {attempts}: {len(matched)}/{len(batch_smiles)} "
                               f"matched properties. Total matched: {len(property_matched)}")
                    
                except Exception as e:
                    logger.warning(f"Property generation attempt {attempts} failed: {str(e)}")
                    continue
        
        # Truncate to requested number
        final_matched = property_matched[:num_molecules]
        
        logger.info(f"Generated {len(final_matched)} property-matched molecules "
                   f"(requested: {num_molecules}, attempts: {attempts})")
        
        return final_matched
    
    def iterative_constraint_generation(self,
                                      num_molecules: int,
                                      constraints: Dict[str, Any],
                                      max_iterations: int = 10,
                                      batch_size: int = 32,
                                      temperature_schedule: Optional[List[float]] = None,
                                      **kwargs) -> List[str]:
        """
        Generate molecules using iterative constraint satisfaction.
        
        Args:
            num_molecules: Number of molecules to generate
            constraints: Dictionary of constraints to satisfy
            max_iterations: Maximum number of iterations
            batch_size: Batch size per iteration
            temperature_schedule: List of temperatures for each iteration
            **kwargs: Additional generation parameters
            
        Returns:
            List of constraint-satisfying SMILES strings
        """
        logger.info(f"Starting iterative constraint generation for {num_molecules} molecules")
        
        if temperature_schedule is None:
            # Default: start high, decrease over iterations
            temperature_schedule = [2.0 - 1.5 * i / max_iterations for i in range(max_iterations)]
        
        all_passed = []
        
        for iteration in range(max_iterations):
            if len(all_passed) >= num_molecules:
                break
                
            temperature = temperature_schedule[iteration] if iteration < len(temperature_schedule) else 1.0
            remaining = num_molecules - len(all_passed)
            current_batch_size = min(batch_size, remaining * 2)
            
            logger.debug(f"Iteration {iteration + 1}/{max_iterations}: "
                        f"temperature={temperature:.2f}, batch_size={current_batch_size}")
            
            try:
                # Generate batch with current temperature
                batch_passed = self.generate_with_constraints(
                    num_molecules=current_batch_size,
                    constraints=constraints,
                    temperature=temperature,
                    batch_size=current_batch_size,
                    max_attempts=3,
                    iterative_filtering=True,
                    **kwargs
                )
                
                all_passed.extend(batch_passed)
                
                logger.debug(f"Iteration {iteration + 1}: Generated {len(batch_passed)} "
                           f"constraint-passing molecules. Total: {len(all_passed)}")
                
            except Exception as e:
                logger.warning(f"Iteration {iteration + 1} failed: {str(e)}")
                continue
        
        # Remove duplicates and truncate
        unique_passed = list(dict.fromkeys(all_passed))  # Preserve order
        final_molecules = unique_passed[:num_molecules]
        
        logger.info(f"Iterative generation completed: {len(final_molecules)} molecules "
                   f"(requested: {num_molecules}, iterations: {max_iterations})")
        
        return final_molecules
    
    def _apply_constraints(self, 
                          smiles_list: List[str], 
                          constraints: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Apply constraints to filter molecules.
        
        Args:
            smiles_list: List of SMILES strings to filter
            constraints: Dictionary of constraints
            
        Returns:
            Tuple of (passed molecules, failed molecules)
        """
        if not constraints:
            return smiles_list, []
        
        passed = []
        failed = []
        
        for smiles in smiles_list:
            if self._molecule_passes_constraints(smiles, constraints):
                passed.append(smiles)
            else:
                failed.append(smiles)
        
        return passed, failed
    
    def _molecule_passes_constraints(self, smiles: str, constraints: Dict[str, Any]) -> bool:
        """
        Check if a molecule passes all constraints.
        
        Args:
            smiles: SMILES string to check
            constraints: Dictionary of constraints
            
        Returns:
            True if molecule passes all constraints
        """
        try:
            # Lipinski rules
            if constraints.get('lipinski', True):
                if not self.constraint_filter.passes_lipinski_filter(smiles):
                    return False
            
            # QED threshold
            qed_threshold = constraints.get('qed_threshold')
            if qed_threshold is not None:
                qed_score = self.constraint_filter.calculate_qed_score(smiles)
                if qed_score is None or qed_score < qed_threshold:
                    return False
            
            # Molecular weight range
            mw_range = constraints.get('mw_range')
            if mw_range is not None:
                mw = self.constraint_filter.calculate_molecular_weight(smiles)
                if mw is None or mw < mw_range[0] or mw > mw_range[1]:
                    return False
            
            # LogP range
            logp_range = constraints.get('logp_range')
            if logp_range is not None:
                logp = self.constraint_filter.calculate_logp(smiles)
                if logp is None or logp < logp_range[0] or logp > logp_range[1]:
                    return False
            
            # Custom property ranges
            for prop_name, prop_range in constraints.items():
                if prop_name.endswith('_range') and prop_name not in ['mw_range', 'logp_range']:
                    # Handle custom property ranges (would need property calculator)
                    continue
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking constraints for {smiles}: {str(e)}")
            return False
    
    def _properties_to_constraints(self, 
                                  target_properties: Dict[str, float], 
                                  tolerance: float) -> Dict[str, Any]:
        """
        Convert target properties to constraint ranges.
        
        Args:
            target_properties: Dictionary of target property values
            tolerance: Tolerance for property matching
            
        Returns:
            Dictionary of constraints
        """
        constraints = {}
        
        for prop_name, target_value in target_properties.items():
            if prop_name.lower() in ['mw', 'molecular_weight']:
                constraints['mw_range'] = [
                    target_value * (1 - tolerance),
                    target_value * (1 + tolerance)
                ]
            elif prop_name.lower() in ['logp', 'log_p']:
                constraints['logp_range'] = [
                    target_value - tolerance * abs(target_value),
                    target_value + tolerance * abs(target_value)
                ]
            elif prop_name.lower() == 'qed':
                constraints['qed_threshold'] = max(0, target_value - tolerance)
            else:
                # Generic property range
                constraints[f'{prop_name}_range'] = [
                    target_value - tolerance * abs(target_value),
                    target_value + tolerance * abs(target_value)
                ]
        
        return constraints
    
    def _filter_by_properties(self,
                             smiles_list: List[str],
                             target_properties: Dict[str, float],
                             property_predictor: Any,
                             tolerance: float) -> List[str]:
        """
        Filter molecules by predicted properties.
        
        Args:
            smiles_list: List of SMILES strings
            target_properties: Target property values
            property_predictor: Property prediction model
            tolerance: Tolerance for property matching
            
        Returns:
            List of molecules matching target properties
        """
        matched = []
        
        try:
            # Convert SMILES to graphs for property prediction
            graphs = []
            valid_smiles = []
            
            for smiles in smiles_list:
                graph = self.smiles_processor.smiles_to_graph(smiles)
                if graph is not None:
                    graphs.append(graph)
                    valid_smiles.append(smiles)
            
            if not graphs:
                return matched
            
            # Predict properties
            predicted_properties = property_predictor.predict_properties(graphs)
            
            # Filter by property matching
            for i, smiles in enumerate(valid_smiles):
                if self._properties_match(predicted_properties[i], target_properties, tolerance):
                    matched.append(smiles)
                    
        except Exception as e:
            logger.warning(f"Error in property-based filtering: {str(e)}")
        
        return matched
    
    def _properties_match(self,
                         predicted: Dict[str, float],
                         target: Dict[str, float],
                         tolerance: float) -> bool:
        """
        Check if predicted properties match targets within tolerance.
        
        Args:
            predicted: Dictionary of predicted property values
            target: Dictionary of target property values
            tolerance: Tolerance for matching
            
        Returns:
            True if all properties match within tolerance
        """
        for prop_name, target_value in target.items():
            if prop_name not in predicted:
                return False
            
            predicted_value = predicted[prop_name]
            if abs(predicted_value - target_value) > tolerance * abs(target_value):
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return self.model.get_model_info()