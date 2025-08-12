"""
Batch prediction utilities for molecular property prediction.

This module provides efficient batch prediction capabilities for
trained property prediction models, with support for large-scale
molecular screening and evaluation.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from .property_predictor import PropertyPredictor, MultiTaskPropertyPredictor
from ..data.smiles_processor import SMILESProcessor
from ..data.feature_extractor import FeatureExtractor
from ..data.property_calculator import PropertyCalculator

logger = logging.getLogger(__name__)


class BatchPropertyPredictor:
    """
    Batch predictor for molecular properties using trained GNN models.
    
    This class provides efficient batch prediction capabilities for
    large sets of molecules, with support for both neural network
    predictions and RDKit-based property calculations.
    """
    
    def __init__(self,
                 model: Optional[Union[PropertyPredictor, MultiTaskPropertyPredictor]] = None,
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 batch_size: int = 64,
                 use_rdkit_fallback: bool = True):
        """
        Initialize the batch predictor.
        
        Args:
            model: Trained property prediction model
            model_path: Path to saved model checkpoint
            device: Device to run predictions on
            batch_size: Batch size for predictions
            use_rdkit_fallback: Whether to use RDKit for fallback calculations
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.use_rdkit_fallback = use_rdkit_fallback
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = None
            logger.warning("No model provided. Only RDKit calculations will be available.")
            
        # Initialize processors
        self.smiles_processor = SMILESProcessor()
        self.feature_extractor = FeatureExtractor()
        self.property_calculator = PropertyCalculator() if use_rdkit_fallback else None
        
        # Model properties
        self.property_names = getattr(self.model, 'property_names', []) if self.model else []
        
    def _load_model(self, model_path: str) -> Union[PropertyPredictor, MultiTaskPropertyPredictor]:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Determine model type and create instance
        model_type = model_config.get('type', 'single_task')
        
        # Get feature dimensions from checkpoint or use defaults
        node_dim = model_config.get('node_dim', 128)
        edge_dim = model_config.get('edge_dim', 64)
        
        if model_type == 'multi_task':
            # Multi-task model
            property_configs = model_config.get('property_configs', {})
            model = MultiTaskPropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                property_configs=property_configs,
                **{k: v for k, v in model_config.items() 
                   if k not in ['type', 'property_configs', 'node_dim', 'edge_dim']}
            )
        else:
            # Single-task model
            properties = config.get('data', {}).get('properties', ['property'])
            model = PropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_properties=len(properties),
                property_names=properties,
                **{k: v for k, v in model_config.items() 
                   if k not in ['type', 'node_dim', 'edge_dim']}
            )
            
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded {model_type} model with {len(model.property_names)} properties")
        
        return model
        
    def predict_neural(self, 
                      smiles_list: List[str],
                      return_embeddings: bool = False) -> Dict[str, Union[List[float], np.ndarray]]:
        """
        Predict properties using the neural network model.
        
        Args:
            smiles_list: List of SMILES strings
            return_embeddings: Whether to return graph embeddings
            
        Returns:
            Dictionary with predictions and optionally embeddings
        """
        if self.model is None:
            raise ValueError("No neural network model available")
            
        logger.info(f"Predicting properties for {len(smiles_list)} molecules using neural network")
        
        # Convert SMILES to graphs
        graphs = []
        valid_indices = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Converting SMILES")):
            try:
                graph = self.smiles_processor.smiles_to_graph(smiles)
                if graph is not None:
                    # Enhance with detailed features
                    mol = self._smiles_to_mol(smiles)
                    if mol is not None:
                        enhanced_features = self.feature_extractor.get_graph_features(mol)
                        if enhanced_features:
                            graph.x = enhanced_features['x']
                            graph.edge_attr = enhanced_features['edge_attr']
                            graph.edge_index = enhanced_features['edge_index']
                            
                    graphs.append(graph)
                    valid_indices.append(i)
                    
            except Exception as e:
                logger.warning(f"Failed to process SMILES {i}: {smiles}, Error: {e}")
                continue
                
        if not graphs:
            logger.warning("No valid graphs generated")
            return {}
            
        # Batch prediction
        all_predictions = {prop: [] for prop in self.property_names}
        all_embeddings = [] if return_embeddings else None
        
        with torch.no_grad():
            for i in range(0, len(graphs), self.batch_size):
                batch_graphs = graphs[i:i + self.batch_size]
                batch = Batch.from_data_list(batch_graphs).to(self.device)
                
                # Get predictions
                if isinstance(self.model, MultiTaskPropertyPredictor):
                    predictions = self.model(batch)
                    
                    for prop in self.property_names:
                        pred_values = predictions[prop].cpu().numpy().flatten()
                        all_predictions[prop].extend(pred_values)
                        
                else:
                    predictions = self.model(batch).cpu().numpy()
                    
                    for j, prop in enumerate(self.property_names):
                        pred_values = predictions[:, j]
                        all_predictions[prop].extend(pred_values)
                        
                # Get embeddings if requested
                if return_embeddings:
                    embeddings = self.model.get_embeddings(batch).cpu().numpy()
                    all_embeddings.append(embeddings)
                    
        # Organize results
        results = {}
        
        # Create full-length arrays with NaN for invalid molecules
        for prop in self.property_names:
            full_predictions = np.full(len(smiles_list), np.nan)
            full_predictions[valid_indices] = all_predictions[prop]
            results[prop] = full_predictions.tolist()
            
        if return_embeddings and all_embeddings:
            full_embeddings = np.full((len(smiles_list), all_embeddings[0].shape[1]), np.nan)
            embeddings_array = np.vstack(all_embeddings)
            full_embeddings[valid_indices] = embeddings_array
            results['embeddings'] = full_embeddings
            
        return results
        
    def predict_rdkit(self, 
                     smiles_list: List[str],
                     properties: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        Predict properties using RDKit calculations.
        
        Args:
            smiles_list: List of SMILES strings
            properties: List of properties to calculate
            
        Returns:
            Dictionary with RDKit-calculated properties
        """
        if self.property_calculator is None:
            raise ValueError("RDKit property calculator not available")
            
        logger.info(f"Calculating RDKit properties for {len(smiles_list)} molecules")
        
        # Default properties if not specified
        if properties is None:
            properties = ['molecular_weight', 'logp', 'qed', 'num_hbd', 'num_hba', 'tpsa']
            
        # Calculate properties
        results = {prop: [] for prop in properties}
        
        for smiles in tqdm(smiles_list, desc="Calculating RDKit properties"):
            props = self.property_calculator.calculate_properties(smiles, properties)
            
            for prop in properties:
                value = props.get(prop, np.nan)
                results[prop].append(value)
                
        return results
        
    def predict_combined(self, 
                        smiles_list: List[str],
                        neural_properties: Optional[List[str]] = None,
                        rdkit_properties: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        Predict properties using both neural network and RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            neural_properties: Properties to predict with neural network
            rdkit_properties: Properties to calculate with RDKit
            
        Returns:
            Combined dictionary of predictions
        """
        results = {}
        
        # Neural network predictions
        if self.model is not None and neural_properties is not None:
            # Filter to available properties
            available_neural = [p for p in neural_properties if p in self.property_names]
            if available_neural:
                neural_results = self.predict_neural(smiles_list)
                for prop in available_neural:
                    if prop in neural_results:
                        results[f'neural_{prop}'] = neural_results[prop]
                        
        # RDKit calculations
        if self.property_calculator is not None and rdkit_properties is not None:
            rdkit_results = self.predict_rdkit(smiles_list, rdkit_properties)
            for prop, values in rdkit_results.items():
                results[f'rdkit_{prop}'] = values
                
        return results
        
    def predict_with_uncertainty(self, 
                                smiles_list: List[str],
                                num_samples: int = 10) -> Dict[str, Dict[str, List[float]]]:
        """
        Predict properties with uncertainty estimation using dropout.
        
        Args:
            smiles_list: List of SMILES strings
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            Dictionary with mean predictions and uncertainties
        """
        if self.model is None:
            raise ValueError("No neural network model available")
            
        logger.info(f"Predicting with uncertainty for {len(smiles_list)} molecules")
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        # Multiple predictions
        all_predictions = []
        
        for _ in range(num_samples):
            predictions = self.predict_neural(smiles_list)
            all_predictions.append(predictions)
            
        # Compute statistics
        results = {}
        
        for prop in self.property_names:
            prop_predictions = np.array([pred[prop] for pred in all_predictions])
            
            # Handle NaN values
            valid_mask = ~np.isnan(prop_predictions).all(axis=0)
            
            means = np.nanmean(prop_predictions, axis=0)
            stds = np.nanstd(prop_predictions, axis=0)
            
            results[prop] = {
                'mean': means.tolist(),
                'std': stds.tolist(),
                'valid': valid_mask.tolist()
            }
            
        # Set model back to eval mode
        self.model.eval()
        
        return results
        
    def screen_molecules(self, 
                        smiles_list: List[str],
                        criteria: Dict[str, Dict[str, float]],
                        use_neural: bool = True) -> Dict[str, Any]:
        """
        Screen molecules based on property criteria.
        
        Args:
            smiles_list: List of SMILES strings
            criteria: Dictionary of property criteria (e.g., {'logp': {'min': 0, 'max': 5}})
            use_neural: Whether to use neural predictions or RDKit
            
        Returns:
            Dictionary with screening results
        """
        logger.info(f"Screening {len(smiles_list)} molecules")
        
        # Get predictions
        if use_neural and self.model is not None:
            predictions = self.predict_neural(smiles_list)
        elif self.property_calculator is not None:
            properties = list(criteria.keys())
            predictions = self.predict_rdkit(smiles_list, properties)
        else:
            raise ValueError("No prediction method available")
            
        # Apply criteria
        passed_indices = []
        failed_criteria = {prop: 0 for prop in criteria.keys()}
        
        for i in range(len(smiles_list)):
            passes = True
            
            for prop, limits in criteria.items():
                if prop not in predictions:
                    continue
                    
                value = predictions[prop][i]
                
                if np.isnan(value):
                    passes = False
                    break
                    
                if 'min' in limits and value < limits['min']:
                    passes = False
                    failed_criteria[prop] += 1
                    
                if 'max' in limits and value > limits['max']:
                    passes = False
                    failed_criteria[prop] += 1
                    
            if passes:
                passed_indices.append(i)
                
        # Results
        passed_smiles = [smiles_list[i] for i in passed_indices]
        
        results = {
            'total_molecules': len(smiles_list),
            'passed_molecules': len(passed_smiles),
            'pass_rate': len(passed_smiles) / len(smiles_list),
            'passed_indices': passed_indices,
            'passed_smiles': passed_smiles,
            'failed_criteria': failed_criteria,
            'predictions': predictions
        }
        
        return results
        
    def save_predictions(self, 
                        smiles_list: List[str],
                        predictions: Dict[str, List[float]],
                        output_path: str):
        """
        Save predictions to CSV file.
        
        Args:
            smiles_list: List of SMILES strings
            predictions: Dictionary of predictions
            output_path: Path to save CSV file
        """
        # Create DataFrame
        data = {'smiles': smiles_list}
        data.update(predictions)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved predictions to {output_path}")
        
    def _smiles_to_mol(self, smiles: str):
        """Convert SMILES to RDKit molecule."""
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles)
        except:
            return None


def batch_predict_from_file(model_path: str,
                           smiles_file: str,
                           output_file: str,
                           batch_size: int = 64,
                           smiles_column: str = 'smiles') -> None:
    """
    Convenience function to predict properties from a file.
    
    Args:
        model_path: Path to trained model
        smiles_file: Path to CSV file with SMILES
        output_file: Path to save predictions
        batch_size: Batch size for predictions
        smiles_column: Name of SMILES column
    """
    # Load SMILES
    df = pd.read_csv(smiles_file)
    smiles_list = df[smiles_column].tolist()
    
    # Create predictor
    predictor = BatchPropertyPredictor(
        model_path=model_path,
        batch_size=batch_size
    )
    
    # Predict
    predictions = predictor.predict_neural(smiles_list)
    
    # Save results
    predictor.save_predictions(smiles_list, predictions, output_file)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch property prediction")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--smiles_file", required=True, help="CSV file with SMILES")
    parser.add_argument("--output_file", required=True, help="Output CSV file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--smiles_column", default="smiles", help="SMILES column name")
    
    args = parser.parse_args()
    
    batch_predict_from_file(
        model_path=args.model_path,
        smiles_file=args.smiles_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        smiles_column=args.smiles_column
    )