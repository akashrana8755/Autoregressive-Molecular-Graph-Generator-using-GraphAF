# FeatureExtractor

The `FeatureExtractor` class extracts comprehensive features from molecular structures for use in graph neural networks, including atom and bond features with chemical properties and structural information.

## Class Definition

```python
class FeatureExtractor:
    """
    Extracts atom and bond features from molecular structures.
    
    This class provides comprehensive feature extraction for atoms and bonds
    in molecular graphs, including chemical properties and structural information.
    """
```

## Constructor

```python
def __init__(self, 
             use_chirality: bool = True,
             use_partial_charge: bool = False,
             max_atomic_num: int = 100)
```

**Parameters:**
- `use_chirality` (bool): Whether to include chirality information
- `use_partial_charge` (bool): Whether to compute partial charges (slower)
- `max_atomic_num` (int): Maximum atomic number to consider

## Methods

### get_atom_features

```python
def get_atom_features(self, atom: Chem.Atom) -> List[float]
```

Extract comprehensive features for an atom.

**Parameters:**
- `atom` (Chem.Atom): RDKit Atom object

**Returns:**
- `List[float]`: List of atom features

**Features Extracted:**
- Atomic number (one-hot encoded)
- Degree (number of bonds)
- Formal charge
- Hybridization state
- Aromaticity
- Number of hydrogen atoms
- Chirality (if enabled)
- Ring membership
- Atomic mass
- Partial charge (if enabled)

**Example:**
```python
from rdkit import Chem
from src.data.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
mol = Chem.MolFromSmiles("CCO")
atom = mol.GetAtomByIdx(0)  # First carbon
features = extractor.get_atom_features(atom)
print(f"Atom features: {len(features)} dimensions")
```

### get_bond_features

```python
def get_bond_features(self, bond: Chem.Bond) -> List[float]
```

Extract comprehensive features for a bond.

**Parameters:**
- `bond` (Chem.Bond): RDKit Bond object

**Returns:**
- `List[float]`: List of bond features

**Features Extracted:**
- Bond type (single, double, triple, aromatic)
- Bond stereo configuration
- Conjugation
- Ring membership
- Aromaticity

**Example:**
```python
mol = Chem.MolFromSmiles("C=C")
bond = mol.GetBondBetweenAtoms(0, 1)  # Double bond
features = extractor.get_bond_features(bond)
print(f"Bond features: {len(features)} dimensions")
```

### get_graph_features

```python
def get_graph_features(self, mol: Chem.Mol) -> Dict[str, torch.Tensor]
```

Extract features for all atoms and bonds in a molecule.

**Parameters:**
- `mol` (Chem.Mol): RDKit Mol object

**Returns:**
- `Dict[str, torch.Tensor]`: Dictionary containing node and edge features as tensors

**Example:**
```python
mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
features = extractor.get_graph_features(mol)

print(f"Node features shape: {features['x'].shape}")
print(f"Edge features shape: {features['edge_attr'].shape}")
print(f"Edge indices shape: {features['edge_index'].shape}")
```

### get_atom_features_dim

```python
def get_atom_features_dim(self) -> int
```

Get the dimensionality of atom features.

**Returns:**
- `int`: Number of atom feature dimensions

### get_bond_features_dim

```python
def get_bond_features_dim(self) -> int
```

Get the dimensionality of bond features.

**Returns:**
- `int`: Number of bond feature dimensions

### get_feature_names

```python
def get_feature_names(self) -> Dict[str, List[str]]
```

Get names of all features for interpretability.

**Returns:**
- `Dict[str, List[str]]`: Dictionary with atom and bond feature names

**Example:**
```python
feature_names = extractor.get_feature_names()
print("Atom features:", feature_names['atom_features'][:5])  # First 5
print("Bond features:", feature_names['bond_features'])
```

### compute_molecular_descriptors

```python
def compute_molecular_descriptors(self, mol: Chem.Mol) -> Dict[str, float]
```

Compute additional molecular descriptors.

**Parameters:**
- `mol` (Chem.Mol): RDKit Mol object

**Returns:**
- `Dict[str, float]`: Dictionary of molecular descriptors

**Descriptors Computed:**
- Molecular weight
- LogP (partition coefficient)
- Number of H-bond donors/acceptors
- Topological polar surface area (TPSA)
- Number of rotatable bonds
- Ring counts
- Fraction of sp3 carbons
- Number of heteroatoms

**Example:**
```python
mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")  # Acetaminophen
descriptors = extractor.compute_molecular_descriptors(mol)
print(f"Molecular weight: {descriptors['molecular_weight']:.2f}")
print(f"LogP: {descriptors['logp']:.2f}")
print(f"H-bond donors: {descriptors['num_hbd']}")
```

### extract_pharmacophore_features

```python
def extract_pharmacophore_features(self, mol: Chem.Mol) -> Dict[str, Any]
```

Extract pharmacophore-relevant features.

**Parameters:**
- `mol` (Chem.Mol): RDKit Mol object

**Returns:**
- `Dict[str, Any]`: Dictionary of pharmacophore features

**Features Extracted:**
- Has aromatic ring
- Has basic nitrogen
- Has acidic group
- Has hydroxyl group
- Has carbonyl group

**Example:**
```python
mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")  # Acetaminophen
pharm_features = extractor.extract_pharmacophore_features(mol)
print(f"Has aromatic ring: {pharm_features['has_aromatic_ring']}")
print(f"Has hydroxyl: {pharm_features['has_hydroxyl']}")
```

## Usage Examples

### Basic Feature Extraction

```python
from rdkit import Chem
from src.data.feature_extractor import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor(use_chirality=True, use_partial_charge=False)

# Create molecule
mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid

# Extract graph features
features = extractor.get_graph_features(mol)
print(f"Extracted features for molecule with {mol.GetNumAtoms()} atoms")
print(f"Node feature dimensions: {features['x'].shape[1]}")
print(f"Edge feature dimensions: {features['edge_attr'].shape[1]}")
```

### Feature Dimensionality

```python
# Get feature dimensions
atom_dim = extractor.get_atom_features_dim()
bond_dim = extractor.get_bond_features_dim()

print(f"Atom features: {atom_dim} dimensions")
print(f"Bond features: {bond_dim} dimensions")

# Get feature names for interpretability
feature_names = extractor.get_feature_names()
print(f"Total atom features: {len(feature_names['atom_features'])}")
print(f"Total bond features: {len(feature_names['bond_features'])}")
```

### Molecular Descriptors

```python
# Compute comprehensive molecular descriptors
molecules = [
    "CC(=O)O",                    # Acetic acid
    "CC(=O)Nc1ccc(O)cc1",        # Acetaminophen
    "CC1=CC=C(C=C1)C(=O)O"       # p-Toluic acid
]

for smiles in molecules:
    mol = Chem.MolFromSmiles(smiles)
    descriptors = extractor.compute_molecular_descriptors(mol)
    
    print(f"\n{smiles}:")
    print(f"  MW: {descriptors['molecular_weight']:.2f}")
    print(f"  LogP: {descriptors['logp']:.2f}")
    print(f"  HBD: {descriptors['num_hbd']}")
    print(f"  HBA: {descriptors['num_hba']}")
    print(f"  TPSA: {descriptors['tpsa']:.2f}")
```

### Pharmacophore Analysis

```python
# Analyze pharmacophore features
drug_molecules = [
    "CC(=O)Nc1ccc(O)cc1",        # Acetaminophen (analgesic)
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", # Ibuprofen (NSAID)
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" # Caffeine (stimulant)
]

for smiles in drug_molecules:
    mol = Chem.MolFromSmiles(smiles)
    pharm_features = extractor.extract_pharmacophore_features(mol)
    
    print(f"\n{smiles}:")
    for feature, value in pharm_features.items():
        print(f"  {feature}: {value}")
```

### Custom Feature Configuration

```python
# Configure extractor for specific needs
extractor_minimal = FeatureExtractor(
    use_chirality=False,      # Skip chirality for faster processing
    use_partial_charge=False, # Skip partial charges
    max_atomic_num=50         # Limit to common elements
)

extractor_detailed = FeatureExtractor(
    use_chirality=True,       # Include chirality
    use_partial_charge=True,  # Include partial charges (slower)
    max_atomic_num=100        # Include more elements
)

# Compare feature dimensions
mol = Chem.MolFromSmiles("CC(=O)O")
features_minimal = extractor_minimal.get_graph_features(mol)
features_detailed = extractor_detailed.get_graph_features(mol)

print(f"Minimal features: {features_minimal['x'].shape[1]} atom dims")
print(f"Detailed features: {features_detailed['x'].shape[1]} atom dims")
```

## Feature Vocabularies

The extractor uses predefined vocabularies for categorical features:

### Atomic Numbers
- Supports elements 1-100 (configurable via `max_atomic_num`)
- One-hot encoded for neural network compatibility

### Hybridization States
- SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED
- Important for understanding molecular geometry

### Bond Types
- SINGLE, DOUBLE, TRIPLE, AROMATIC
- Fundamental for chemical bonding representation

### Formal Charges
- Range: -3 to +3
- Important for ionic and charged species

## Performance Considerations

- **Chirality**: Adds computational overhead but important for drug molecules
- **Partial charges**: Significantly slower but provides electronic information
- **Batch processing**: Process multiple molecules together when possible
- **Feature caching**: Consider caching features for frequently used molecules

## Dependencies

- RDKit: For molecular property calculation
- PyTorch: For tensor operations
- NumPy: For numerical computations