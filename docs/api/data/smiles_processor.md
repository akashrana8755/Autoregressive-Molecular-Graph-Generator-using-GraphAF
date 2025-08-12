# SMILESProcessor

The `SMILESProcessor` class handles bidirectional conversion between SMILES strings and molecular graph representations compatible with PyTorch Geometric.

## Class Definition

```python
class SMILESProcessor:
    """
    Converts SMILES strings to molecular graphs and vice versa.
    
    This class handles the conversion between SMILES string representations
    and PyTorch Geometric Data objects representing molecular graphs.
    """
```

## Constructor

```python
def __init__(self, 
             add_self_loops: bool = False,
             explicit_hydrogens: bool = False,
             sanitize: bool = True)
```

**Parameters:**
- `add_self_loops` (bool): Whether to add self-loops to atoms in the graph
- `explicit_hydrogens` (bool): Whether to include explicit hydrogen atoms
- `sanitize` (bool): Whether to sanitize molecules during processing

## Methods

### smiles_to_graph

```python
def smiles_to_graph(self, smiles: str) -> Optional[Data]
```

Convert a SMILES string to a molecular graph.

**Parameters:**
- `smiles` (str): SMILES string representation of the molecule

**Returns:**
- `Optional[Data]`: PyTorch Geometric Data object or None if conversion fails

**Example:**
```python
processor = SMILESProcessor()
graph = processor.smiles_to_graph("CCO")  # Ethanol
if graph is not None:
    print(f"Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
```

### graph_to_smiles

```python
def graph_to_smiles(self, graph: Data) -> Optional[str]
```

Convert a molecular graph back to SMILES string.

**Parameters:**
- `graph` (Data): PyTorch Geometric Data object representing molecular graph

**Returns:**
- `Optional[str]`: SMILES string or None if conversion fails

**Example:**
```python
smiles = processor.graph_to_smiles(graph)
print(f"Reconstructed SMILES: {smiles}")
```

### validate_molecule

```python
def validate_molecule(self, smiles: str) -> bool
```

Validate if a SMILES string represents a valid molecule.

**Parameters:**
- `smiles` (str): SMILES string to validate

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
is_valid = processor.validate_molecule("CCO")  # True
is_invalid = processor.validate_molecule("C[C")  # False (invalid syntax)
```

### sanitize_smiles

```python
def sanitize_smiles(self, smiles: str) -> Optional[str]
```

Sanitize and canonicalize a SMILES string.

**Parameters:**
- `smiles` (str): Input SMILES string

**Returns:**
- `Optional[str]`: Canonicalized SMILES string or None if invalid

**Example:**
```python
canonical = processor.sanitize_smiles("CCO")
print(f"Canonical SMILES: {canonical}")
```

### batch_process_smiles

```python
def batch_process_smiles(self, smiles_list: List[str]) -> List[Optional[Data]]
```

Process a batch of SMILES strings to molecular graphs.

**Parameters:**
- `smiles_list` (List[str]): List of SMILES strings

**Returns:**
- `List[Optional[Data]]`: List of Data objects (None for failed conversions)

**Example:**
```python
smiles_list = ["CCO", "CC", "CCC"]
graphs = processor.batch_process_smiles(smiles_list)
valid_graphs = [g for g in graphs if g is not None]
print(f"Successfully converted {len(valid_graphs)}/{len(smiles_list)} molecules")
```

### get_molecule_info

```python
def get_molecule_info(self, smiles: str) -> Optional[Dict[str, Any]]
```

Get basic information about a molecule from its SMILES.

**Parameters:**
- `smiles` (str): SMILES string

**Returns:**
- `Optional[Dict[str, Any]]`: Dictionary with molecule information or None if invalid

**Example:**
```python
info = processor.get_molecule_info("CCO")
if info:
    print(f"Atoms: {info['num_atoms']}, MW: {info['molecular_weight']:.2f}")
```

## Usage Examples

### Basic Conversion

```python
from src.data.smiles_processor import SMILESProcessor

# Initialize processor
processor = SMILESProcessor(sanitize=True)

# Convert SMILES to graph
smiles = "CC(=O)O"  # Acetic acid
graph = processor.smiles_to_graph(smiles)

if graph is not None:
    print(f"Graph created with {graph.x.shape[0]} atoms")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge features shape: {graph.edge_attr.shape}")
    
    # Convert back to SMILES
    reconstructed = processor.graph_to_smiles(graph)
    print(f"Original: {smiles}")
    print(f"Reconstructed: {reconstructed}")
```

### Batch Processing

```python
# Process multiple molecules
drug_smiles = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CC1=CC=C(C=C1)C(=O)O",            # p-Toluic acid
    "C1=CC=C(C=C1)C(=O)O"              # Benzoic acid
]

graphs = processor.batch_process_smiles(drug_smiles)
valid_count = sum(1 for g in graphs if g is not None)
print(f"Successfully processed {valid_count}/{len(drug_smiles)} molecules")

# Get molecule information
for smiles in drug_smiles:
    info = processor.get_molecule_info(smiles)
    if info:
        print(f"{smiles}: {info['num_atoms']} atoms, MW={info['molecular_weight']:.1f}")
```

### Validation and Sanitization

```python
# Validate molecules before processing
test_smiles = ["CCO", "C[C", "CC(=O)O", "invalid"]

for smiles in test_smiles:
    if processor.validate_molecule(smiles):
        canonical = processor.sanitize_smiles(smiles)
        print(f"Valid: {smiles} -> {canonical}")
    else:
        print(f"Invalid: {smiles}")
```

## Error Handling

The SMILESProcessor handles various error conditions gracefully:

- **Invalid SMILES**: Returns None and logs warning
- **RDKit parsing failures**: Returns None with error logging
- **Sanitization errors**: Returns None for unsanitizable molecules
- **Graph conversion errors**: Returns None with detailed error information

## Performance Considerations

- Use `batch_process_smiles()` for processing multiple molecules efficiently
- Set `explicit_hydrogens=False` for faster processing if hydrogens aren't needed
- Enable `sanitize=True` for robust molecule validation
- Consider caching results for frequently processed molecules

## Dependencies

- RDKit: For molecular parsing and manipulation
- PyTorch Geometric: For graph data structures
- NumPy: For numerical operations