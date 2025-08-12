# MolecularGenerator

The `MolecularGenerator` class provides the main interface for generating molecular structures using trained generative models with constraint filtering and validation.

## Class Definition

```python
class MolecularGenerator:
    """
    Main interface for molecular generation with constraint filtering.
    
    This class provides batch generation capabilities with configurable parameters,
    graph-to-SMILES conversion with validation, and constraint-aware generation.
    """
```

## Constructor

```python
def __init__(self, 
             model: BaseGenerativeModel,
             smiles_processor: Optional[SMILESProcessor] = None,
             constraint_filter: Optional[ConstraintFilter] = None,
             device: Optional[torch.device] = None)
```

**Parameters:**
- `model` (BaseGenerativeModel): Trained generative model (GraphDiffusion or GraphAF)
- `smiles_processor` (Optional[SMILESProcessor]): SMILES processor for graph-to-SMILES conversion
- `constraint_filter` (Optional[ConstraintFilter]): Constraint filter for drug-likeness filtering
- `device` (Optional[torch.device]): Device to run generation on

## Class Methods

### from_checkpoint

```python
@classmethod
def from_checkpoint(cls, 
                   checkpoint_path: Union[str, Path],
                   model_class: Optional[type] = None,
                   device: Optional[torch.device] = None,
                   **kwargs) -> 'MolecularGenerator'
```

Create generator from a saved model checkpoint.

**Parameters:**
- `checkpoint_path` (Union[str, Path]): Path to the model checkpoint
- `model_class` (Optional[type]): Model class (GraphDiffusion or GraphAF)
- `device` (Optional[torch.device]): Device to load model on
- `**kwargs`: Additional arguments for generator initialization

**Returns:**
- `MolecularGenerator`: Initialized MolecularGenerator instance

**Example:**
```python
from src.generate.molecular_generator import MolecularGenerator

# Load from checkpoint
generator = MolecularGenerator.from_checkpoint(
    checkpoint_path="models/diffusion_model.pt",
    device=torch.device("cuda")
)
```

## Instance Methods

### generate

```python
def generate(self, 
             num_molecules: int,
             max_nodes: Optional[int] = None,
             temperature: float = 1.0,
             batch_size: int = 32,
             max_attempts: int = 10,
             return_graphs: bool = False,
             **kwargs) -> Union[List[str], Tuple[List[str], List[Data]]]
```

Generate molecular structures as SMILES strings.

**Parameters:**
- `num_molecules` (int): Number of molecules to generate
- `max_nodes` (Optional[int]): Maximum number of nodes per molecule
- `temperature` (float): Sampling temperature (higher = more diverse)
- `batch_size` (int): Batch size for generation
- `max_attempts` (int): Maximum attempts per molecule
- `return_graphs` (bool): Whether to return molecular graphs along with SMILES
- `**kwargs`: Additional sampling parameters

**Returns:**
- `Union[List[str], Tuple[List[str], List[Data]]]`: List of SMILES strings, optionally with molecular graphs

**Example:**
```python
# Basic generation
molecules = generator.generate(num_molecules=100, temperature=1.2)
print(f"Generated {len(molecules)} molecules")

# Generation with graphs
molecules, graphs = generator.generate(
    num_molecules=50, 
    return_graphs=True,
    max_nodes=30
)
```

### generate_with_constraints

```python
def generate_with_constraints(self,
                             num_molecules: int,
                             constraints: Optional[Dict[str, Any]] = None,
                             max_nodes: Optional[int] = None,
                             temperature: float = 1.0,
                             batch_size: int = 32,
                             max_attempts: int = 20,
                             iterative_filtering: bool = True,
                             return_all: bool = False,
                             **kwargs) -> Union[List[str], Dict[str, List[str]]]
```

Generate molecules with constraint filtering.

**Parameters:**
- `num_molecules` (int): Number of constraint-passing molecules to generate
- `constraints` (Optional[Dict[str, Any]]): Dictionary of constraints to apply
- `max_nodes` (Optional[int]): Maximum number of nodes per molecule
- `temperature` (float): Sampling temperature
- `batch_size` (int): Batch size for generation
- `max_attempts` (int): Maximum generation attempts
- `iterative_filtering` (bool): Whether to filter during generation or after
- `return_all` (bool): Whether to return all generated molecules (passed and failed)
- `**kwargs`: Additional generation parameters

**Returns:**
- `Union[List[str], Dict[str, List[str]]]`: List of constraint-passing SMILES, or dict with all results if return_all=True

**Constraint Options:**
- `lipinski` (bool): Apply Lipinski's Rule of Five
- `qed_threshold` (float): Minimum QED score
- `mw_range` (List[float]): Molecular weight range [min, max]
- `logp_range` (List[float]): LogP range [min, max]

**Example:**
```python
# Generate drug-like molecules
constraints = {
    'lipinski': True,
    'qed_threshold': 0.5,
    'mw_range': [200, 500],
    'logp_range': [0, 3]
}

drug_molecules = generator.generate_with_constraints(
    num_molecules=100,
    constraints=constraints,
    temperature=1.0
)
print(f"Generated {len(drug_molecules)} drug-like molecules")
```

### generate_with_properties

```python
def generate_with_properties(self,
                            target_properties: Dict[str, float],
                            num_molecules: int,
                            property_predictor: Optional[Any] = None,
                            tolerance: float = 0.1,
                            max_nodes: Optional[int] = None,
                            temperature: float = 1.0,
                            batch_size: int = 32,
                            max_attempts: int = 30,
                            **kwargs) -> List[str]
```

Generate molecules targeting specific properties.

**Parameters:**
- `target_properties` (Dict[str, float]): Dictionary of target property values
- `num_molecules` (int): Number of molecules to generate
- `property_predictor` (Optional[Any]): Property prediction model (optional)
- `tolerance` (float): Tolerance for property matching
- `max_nodes` (Optional[int]): Maximum number of nodes per molecule
- `temperature` (float): Sampling temperature
- `batch_size` (int): Batch size for generation
- `max_attempts` (int): Maximum generation attempts
- `**kwargs`: Additional generation parameters

**Returns:**
- `List[str]`: List of SMILES strings with properties close to targets

**Example:**
```python
# Target specific properties
target_props = {
    'molecular_weight': 300.0,
    'logp': 2.5,
    'qed': 0.7
}

targeted_molecules = generator.generate_with_properties(
    target_properties=target_props,
    num_molecules=50,
    tolerance=0.15
)
```

### iterative_constraint_generation

```python
def iterative_constraint_generation(self,
                                  num_molecules: int,
                                  constraints: Dict[str, Any],
                                  max_iterations: int = 10,
                                  batch_size: int = 32,
                                  temperature_schedule: Optional[List[float]] = None,
                                  **kwargs) -> List[str]
```

Generate molecules using iterative constraint satisfaction.

**Parameters:**
- `num_molecules` (int): Number of molecules to generate
- `constraints` (Dict[str, Any]): Dictionary of constraints to satisfy
- `max_iterations` (int): Maximum number of iterations
- `batch_size` (int): Batch size per iteration
- `temperature_schedule` (Optional[List[float]]): List of temperatures for each iteration
- `**kwargs`: Additional generation parameters

**Returns:**
- `List[str]`: List of constraint-satisfying SMILES strings

**Example:**
```python
# Iterative generation with cooling schedule
constraints = {'lipinski': True, 'qed_threshold': 0.6}
temperature_schedule = [2.0, 1.5, 1.2, 1.0, 0.8]  # Cooling schedule

molecules = generator.iterative_constraint_generation(
    num_molecules=100,
    constraints=constraints,
    max_iterations=5,
    temperature_schedule=temperature_schedule
)
```

### validate_molecules

```python
def validate_molecules(self, smiles_list: List[str]) -> Dict[str, Any]
```

Validate a list of molecules and return statistics.

**Parameters:**
- `smiles_list` (List[str]): List of SMILES strings to validate

**Returns:**
- `Dict[str, Any]`: Dictionary with validation statistics

**Example:**
```python
# Validate generated molecules
stats = generator.validate_molecules(molecules)
print(f"Validity rate: {stats['validity_rate']:.2%}")
print(f"Uniqueness rate: {stats['uniqueness_rate']:.2%}")
```

### get_generation_statistics

```python
def get_generation_statistics(self) -> Dict[str, Any]
```

Get current generation statistics.

**Returns:**
- `Dict[str, Any]`: Dictionary with generation statistics

**Example:**
```python
stats = generator.get_generation_statistics()
print(f"Total generated: {stats['total_generated']}")
print(f"Valid molecules: {stats['valid_molecules']}")
print(f"Validity rate: {stats['validity_rate']:.2%}")
```

## Usage Examples

### Basic Generation

```python
from src.generate.molecular_generator import MolecularGenerator
import torch

# Load trained model
generator = MolecularGenerator.from_checkpoint(
    "checkpoints/diffusion_model.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Generate molecules
molecules = generator.generate(
    num_molecules=1000,
    temperature=1.0,
    batch_size=64
)

print(f"Generated {len(molecules)} unique molecules")

# Validate results
stats = generator.validate_molecules(molecules)
print(f"Validity: {stats['validity_rate']:.1%}")
print(f"Uniqueness: {stats['uniqueness_rate']:.1%}")
```

### Drug-like Molecule Generation

```python
# Define drug-likeness constraints
drug_constraints = {
    'lipinski': True,           # Lipinski's Rule of Five
    'qed_threshold': 0.5,       # Minimum drug-likeness score
    'mw_range': [150, 500],     # Molecular weight range
    'logp_range': [-1, 4]       # LogP range for oral bioavailability
}

# Generate drug-like molecules
drug_molecules = generator.generate_with_constraints(
    num_molecules=500,
    constraints=drug_constraints,
    temperature=1.2,
    max_attempts=30,
    iterative_filtering=True
)

print(f"Generated {len(drug_molecules)} drug-like molecules")

# Get detailed results
all_results = generator.generate_with_constraints(
    num_molecules=100,
    constraints=drug_constraints,
    return_all=True
)

print(f"Passed constraints: {len(all_results['constraint_passed'])}")
print(f"Failed constraints: {len(all_results['constraint_failed'])}")
print(f"Total generated: {len(all_results['all_generated'])}")
```

### Property-Targeted Generation

```python
# Target specific molecular properties
target_properties = {
    'molecular_weight': 250.0,  # Target MW
    'logp': 2.0,               # Target LogP
    'qed': 0.8,                # Target drug-likeness
    'tpsa': 60.0               # Target polar surface area
}

# Generate molecules with target properties
targeted_molecules = generator.generate_with_properties(
    target_properties=target_properties,
    num_molecules=200,
    tolerance=0.2,  # 20% tolerance
    temperature=1.1
)

print(f"Generated {len(targeted_molecules)} property-matched molecules")
```

### Iterative Generation with Cooling

```python
# Define challenging constraints
strict_constraints = {
    'lipinski': True,
    'qed_threshold': 0.7,      # High drug-likeness
    'mw_range': [200, 350],    # Narrow MW range
    'logp_range': [1, 3]       # Optimal LogP range
}

# Use cooling schedule for better constraint satisfaction
temperature_schedule = [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6]

molecules = generator.iterative_constraint_generation(
    num_molecules=300,
    constraints=strict_constraints,
    max_iterations=len(temperature_schedule),
    temperature_schedule=temperature_schedule,
    batch_size=50
)

print(f"Generated {len(molecules)} molecules with strict constraints")
```

### Batch Generation and Analysis

```python
# Generate multiple batches for analysis
all_molecules = []
batch_stats = []

for i in range(5):  # 5 batches
    batch_molecules = generator.generate(
        num_molecules=200,
        temperature=1.0 + i * 0.1,  # Varying temperature
        batch_size=32
    )
    
    all_molecules.extend(batch_molecules)
    
    # Analyze each batch
    stats = generator.validate_molecules(batch_molecules)
    batch_stats.append(stats)
    
    print(f"Batch {i+1}: {len(batch_molecules)} molecules, "
          f"validity={stats['validity_rate']:.1%}")

# Overall statistics
overall_stats = generator.validate_molecules(all_molecules)
print(f"\nOverall: {len(all_molecules)} molecules")
print(f"Validity: {overall_stats['validity_rate']:.1%}")
print(f"Uniqueness: {overall_stats['uniqueness_rate']:.1%}")
```

## Performance Optimization

### Batch Size Tuning
```python
# Optimize batch size for your hardware
import time

batch_sizes = [16, 32, 64, 128]
for batch_size in batch_sizes:
    start_time = time.time()
    molecules = generator.generate(
        num_molecules=100,
        batch_size=batch_size
    )
    elapsed = time.time() - start_time
    print(f"Batch size {batch_size}: {elapsed:.2f}s for 100 molecules")
```

### Memory Management
```python
# For large-scale generation, use smaller batches
def generate_large_scale(generator, total_molecules, batch_size=32):
    all_molecules = []
    remaining = total_molecules
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch_molecules = generator.generate(
            num_molecules=current_batch,
            max_attempts=3  # Limit attempts to save time
        )
        all_molecules.extend(batch_molecules)
        remaining -= len(batch_molecules)
        
        if len(all_molecules) % 1000 == 0:
            print(f"Generated {len(all_molecules)}/{total_molecules} molecules")
    
    return all_molecules

# Generate 10,000 molecules efficiently
large_set = generate_large_scale(generator, 10000, batch_size=64)
```

## Error Handling

The MolecularGenerator handles various error conditions:

- **Model loading failures**: Detailed error messages for checkpoint issues
- **Invalid graph generation**: Skips invalid structures and continues
- **SMILES conversion errors**: Logs failures and returns valid molecules only
- **Constraint satisfaction failures**: Provides statistics on pass/fail rates
- **Device memory issues**: Automatic batch size reduction suggestions

## Dependencies

- PyTorch: For model inference and tensor operations
- PyTorch Geometric: For graph data handling
- RDKit: For molecular validation and property calculation
- NumPy: For numerical operations