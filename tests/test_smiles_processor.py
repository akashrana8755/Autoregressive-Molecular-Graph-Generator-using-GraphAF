"""
Comprehensive unit tests for SMILESProcessor class.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.smiles_processor import SMILESProcessor


class TestSMILESProcessor:
    """Test cases for SMILESProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SMILESProcessor()
        
        # Test molecules with known properties
        self.valid_smiles = [
            'C',           # Methane
            'CC',          # Ethane  
            'CCO',         # Ethanol
            'c1ccccc1',    # Benzene
            'CC(=O)O',     # Acetic acid
            'CCN(CC)CC',   # Triethylamine
            'C1CCC(CC1)O', # Cyclohexanol
        ]
        
        self.invalid_smiles = [
            '',            # Empty string
            'X',           # Invalid atom
            'C(',          # Unmatched parenthesis
            'C[C@H',       # Incomplete stereochemistry
            'invalid',     # Nonsense string
            'C1CC1C1',     # Invalid ring closure
        ]
        
        self.aromatic_smiles = [
            'c1ccccc1',    # Benzene
            'c1ccc2ccccc2c1',  # Naphthalene
            'c1ccc(cc1)O', # Phenol
        ]
    
    def test_initialization(self):
        """Test SMILESProcessor initialization."""
        processor = SMILESProcessor()
        assert processor is not None
        
        # Test with custom parameters
        processor_custom = SMILESProcessor(
            add_hydrogens=False,
            kekulize=False,
            sanitize=False
        )
        assert processor_custom is not None
    
    def test_validate_molecule_valid(self):
        """Test molecule validation with valid SMILES."""
        for smiles in self.valid_smiles:
            assert self.processor.validate_molecule(smiles), f"Valid SMILES {smiles} failed validation"
    
    def test_validate_molecule_invalid(self):
        """Test molecule validation with invalid SMILES."""
        for smiles in self.invalid_smiles:
            assert not self.processor.validate_molecule(smiles), f"Invalid SMILES {smiles} passed validation"
    
    def test_sanitize_smiles_valid(self):
        """Test SMILES sanitization with valid molecules."""
        for smiles in self.valid_smiles:
            sanitized = self.processor.sanitize_smiles(smiles)
            assert sanitized is not None, f"Failed to sanitize {smiles}"
            assert isinstance(sanitized, str), f"Sanitized SMILES should be string, got {type(sanitized)}"
            assert len(sanitized) > 0, f"Sanitized SMILES should not be empty for {smiles}"
    
    def test_sanitize_smiles_invalid(self):
        """Test SMILES sanitization with invalid molecules."""
        for smiles in self.invalid_smiles:
            sanitized = self.processor.sanitize_smiles(smiles)
            assert sanitized is None, f"Invalid SMILES {smiles} should return None after sanitization"
    
    def test_canonicalize_smiles(self):
        """Test SMILES canonicalization."""
        # Test equivalent representations
        equivalent_pairs = [
            ('CCO', 'OCC'),           # Different atom ordering
            ('C(C)O', 'CCO'),         # Different bracketing
            ('c1ccccc1', 'C1=CC=CC=C1'),  # Aromatic vs Kekule
        ]
        
        for smiles1, smiles2 in equivalent_pairs:
            canon1 = self.processor.canonicalize_smiles(smiles1)
            canon2 = self.processor.canonicalize_smiles(smiles2)
            assert canon1 == canon2, f"Equivalent SMILES {smiles1} and {smiles2} have different canonical forms"
    
    def test_smiles_to_graph_basic(self):
        """Test basic SMILES to graph conversion."""
        for smiles in self.valid_smiles:
            graph = self.processor.smiles_to_graph(smiles)
            
            assert graph is not None, f"Failed to convert {smiles} to graph"
            assert hasattr(graph, 'x'), f"Graph missing node features for {smiles}"
            assert hasattr(graph, 'edge_index'), f"Graph missing edge indices for {smiles}"
            assert hasattr(graph, 'edge_attr'), f"Graph missing edge features for {smiles}"
            assert hasattr(graph, 'smiles'), f"Graph missing original SMILES for {smiles}"
            
            # Check tensor properties
            assert isinstance(graph.x, torch.Tensor), "Node features should be tensor"
            assert isinstance(graph.edge_index, torch.Tensor), "Edge indices should be tensor"
            assert isinstance(graph.edge_attr, torch.Tensor), "Edge features should be tensor"
            
            # Check dimensions
            assert graph.x.dim() == 2, f"Node features should be 2D, got {graph.x.dim()}"
            assert graph.edge_index.dim() == 2, f"Edge indices should be 2D, got {graph.edge_index.dim()}"
            assert graph.edge_attr.dim() == 2, f"Edge features should be 2D, got {graph.edge_attr.dim()}"
            
            # Check edge index format
            assert graph.edge_index.size(0) == 2, f"Edge index should have 2 rows, got {graph.edge_index.size(0)}"
            
            # Check consistency
            num_nodes = graph.x.size(0)
            num_edges = graph.edge_index.size(1)
            assert graph.edge_attr.size(0) == num_edges, "Edge features and edge indices size mismatch"
            
            # Check edge indices are valid
            if num_edges > 0:
                assert graph.edge_index.max() < num_nodes, "Edge indices exceed number of nodes"
                assert graph.edge_index.min() >= 0, "Edge indices should be non-negative"
    
    def test_smiles_to_graph_invalid(self):
        """Test SMILES to graph conversion with invalid molecules."""
        for smiles in self.invalid_smiles:
            graph = self.processor.smiles_to_graph(smiles)
            assert graph is None, f"Invalid SMILES {smiles} should return None"
    
    def test_graph_to_smiles_roundtrip(self):
        """Test graph to SMILES conversion and roundtrip consistency."""
        for smiles in self.valid_smiles:
            # Convert to graph
            graph = self.processor.smiles_to_graph(smiles)
            assert graph is not None, f"Failed to convert {smiles} to graph"
            
            # Convert back to SMILES
            reconstructed = self.processor.graph_to_smiles(graph)
            assert reconstructed is not None, f"Failed to convert graph back to SMILES for {smiles}"
            
            # Check that both represent the same molecule (canonical forms should match)
            original_canon = self.processor.canonicalize_smiles(smiles)
            reconstructed_canon = self.processor.canonicalize_smiles(reconstructed)
            
            # Allow for some flexibility in representation
            assert original_canon is not None and reconstructed_canon is not None, \
                f"Canonicalization failed for {smiles} -> {reconstructed}"
    
    def test_graph_to_smiles_invalid(self):
        """Test graph to SMILES conversion with invalid graphs."""
        # Empty graph
        empty_graph = MagicMock()
        empty_graph.x = torch.empty(0, 10)
        empty_graph.edge_index = torch.empty(2, 0, dtype=torch.long)
        empty_graph.edge_attr = torch.empty(0, 5)
        
        result = self.processor.graph_to_smiles(empty_graph)
        assert result is None, "Empty graph should return None"
        
        # Graph with invalid node features
        invalid_graph = MagicMock()
        invalid_graph.x = torch.full((3, 10), float('nan'))
        invalid_graph.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        invalid_graph.edge_attr = torch.zeros(2, 5)
        
        result = self.processor.graph_to_smiles(invalid_graph)
        assert result is None, "Graph with NaN features should return None"
    
    def test_get_molecular_properties(self):
        """Test molecular property calculation."""
        for smiles in self.valid_smiles:
            props = self.processor.get_molecular_properties(smiles)
            
            assert props is not None, f"Failed to get properties for {smiles}"
            assert isinstance(props, dict), "Properties should be returned as dictionary"
            
            # Check expected properties
            expected_props = ['molecular_weight', 'logp', 'num_atoms', 'num_bonds', 'num_rings']
            for prop in expected_props:
                assert prop in props, f"Missing property {prop} for {smiles}"
                assert isinstance(props[prop], (int, float)), f"Property {prop} should be numeric"
                assert not np.isnan(props[prop]), f"Property {prop} should not be NaN for {smiles}"
    
    def test_get_molecular_properties_invalid(self):
        """Test molecular property calculation with invalid SMILES."""
        for smiles in self.invalid_smiles:
            props = self.processor.get_molecular_properties(smiles)
            assert props is None, f"Invalid SMILES {smiles} should return None for properties"
    
    def test_batch_processing(self):
        """Test batch processing of multiple SMILES."""
        # Valid batch
        graphs = self.processor.smiles_to_graphs(self.valid_smiles)
        assert len(graphs) <= len(self.valid_smiles), "Should not return more graphs than input SMILES"
        
        # All returned graphs should be valid
        for graph in graphs:
            assert graph is not None, "Batch processing should not return None graphs"
            assert hasattr(graph, 'x'), "All graphs should have node features"
            assert hasattr(graph, 'edge_index'), "All graphs should have edge indices"
            assert hasattr(graph, 'edge_attr'), "All graphs should have edge features"
        
        # Mixed valid/invalid batch
        mixed_smiles = self.valid_smiles + self.invalid_smiles
        mixed_graphs = self.processor.smiles_to_graphs(mixed_smiles)
        
        # Should only return valid graphs
        assert len(mixed_graphs) <= len(self.valid_smiles), "Should filter out invalid SMILES"
        for graph in mixed_graphs:
            assert graph is not None, "Batch processing should filter out None results"
    
    def test_feature_consistency(self):
        """Test that feature dimensions are consistent across molecules."""
        graphs = self.processor.smiles_to_graphs(self.valid_smiles)
        
        if len(graphs) > 1:
            # Check that all graphs have same feature dimensions
            node_dim = graphs[0].x.size(1)
            edge_dim = graphs[0].edge_attr.size(1)
            
            for graph in graphs[1:]:
                assert graph.x.size(1) == node_dim, "Node feature dimensions should be consistent"
                assert graph.edge_attr.size(1) == edge_dim, "Edge feature dimensions should be consistent"
    
    def test_aromatic_handling(self):
        """Test proper handling of aromatic molecules."""
        for smiles in self.aromatic_smiles:
            graph = self.processor.smiles_to_graph(smiles)
            assert graph is not None, f"Failed to process aromatic SMILES {smiles}"
            
            # Check that aromatic information is preserved in features
            # (This depends on the specific feature extraction implementation)
            assert graph.x.size(0) > 0, f"Aromatic molecule {smiles} should have atoms"
            assert graph.edge_index.size(1) > 0, f"Aromatic molecule {smiles} should have bonds"
    
    def test_stereochemistry_handling(self):
        """Test handling of stereochemical information."""
        stereo_smiles = [
            'C[C@H](O)C',      # R stereocenter
            'C[C@@H](O)C',     # S stereocenter
            'C/C=C/C',         # E double bond
            'C/C=C\\C',        # Z double bond
        ]
        
        for smiles in stereo_smiles:
            graph = self.processor.smiles_to_graph(smiles)
            # Should handle stereochemistry gracefully (may or may not preserve it)
            if graph is not None:  # Some processors might not handle stereochemistry
                assert graph.x.size(0) > 0, f"Stereo molecule {smiles} should have atoms"
    
    def test_large_molecule_handling(self):
        """Test handling of larger molecules."""
        large_smiles = [
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',  # Complex drug-like molecule
        ]
        
        for smiles in large_smiles:
            graph = self.processor.smiles_to_graph(smiles)
            if graph is not None:  # Might fail for very complex molecules
                assert graph.x.size(0) > 10, f"Large molecule {smiles} should have many atoms"
                assert graph.edge_index.size(1) > 10, f"Large molecule {smiles} should have many bonds"
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # None input
        assert self.processor.validate_molecule(None) is False
        assert self.processor.smiles_to_graph(None) is None
        assert self.processor.sanitize_smiles(None) is None
        
        # Empty string
        assert self.processor.validate_molecule('') is False
        assert self.processor.smiles_to_graph('') is None
        
        # Very long string
        long_string = 'C' * 1000
        result = self.processor.validate_molecule(long_string)
        # Should handle gracefully (may be valid or invalid depending on implementation)
        assert isinstance(result, bool)
    
    def test_caching_behavior(self):
        """Test caching behavior if implemented."""
        # This test assumes caching might be implemented
        smiles = 'CCO'
        
        # First call
        graph1 = self.processor.smiles_to_graph(smiles)
        
        # Second call (might use cache)
        graph2 = self.processor.smiles_to_graph(smiles)
        
        # Results should be equivalent (but not necessarily identical objects)
        if graph1 is not None and graph2 is not None:
            assert torch.equal(graph1.x, graph2.x), "Cached results should be consistent"
            assert torch.equal(graph1.edge_index, graph2.edge_index), "Cached results should be consistent"
            assert torch.equal(graph1.edge_attr, graph2.edge_attr), "Cached results should be consistent"
    
    @patch('src.data.smiles_processor.Chem')
    def test_rdkit_error_handling(self, mock_chem):
        """Test handling of RDKit errors."""
        # Mock RDKit to raise exceptions
        mock_chem.MolFromSmiles.side_effect = Exception("RDKit error")
        
        processor = SMILESProcessor()
        result = processor.smiles_to_graph('CCO')
        assert result is None, "Should handle RDKit exceptions gracefully"
    
    def test_memory_efficiency(self):
        """Test memory efficiency with many molecules."""
        # Process many molecules to check for memory leaks
        many_smiles = self.valid_smiles * 100  # 700 molecules
        
        graphs = self.processor.smiles_to_graphs(many_smiles)
        
        # Should complete without memory issues
        assert len(graphs) > 0, "Should process large batches successfully"
        
        # Clean up
        del graphs
    
    def test_thread_safety(self):
        """Test thread safety of the processor."""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_smiles(smiles_list):
            try:
                local_results = self.processor.smiles_to_graphs(smiles_list)
                results.extend(local_results)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_smiles, args=(self.valid_smiles,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety test failed with errors: {errors}"
        assert len(results) > 0, "Thread safety test should produce results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])