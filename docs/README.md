# MolecuGen Documentation

MolecuGen is a diffusion-based graph generator for drug-likeness-constrained molecular generation. This documentation provides comprehensive API reference, usage examples, and best practices.

## Table of Contents

- [API Reference](api/README.md) - Complete API documentation
- [Configuration Guide](configuration.md) - Configuration reference and best practices
- [Examples](examples/README.md) - Usage examples and tutorials
- [Best Practices](best_practices.md) - Recommended practices and patterns

## Quick Start

```python
from src.generate.molecular_generator import MolecularGenerator
from src.models.graph_diffusion import GraphDiffusion

# Load trained model
generator = MolecularGenerator.from_checkpoint('path/to/checkpoint.pt')

# Generate molecules
molecules = generator.generate(num_molecules=100)
print(f"Generated {len(molecules)} molecules")
```

## Key Components

- **Data Processing**: SMILES to graph conversion and feature extraction
- **Models**: GraphDiffusion and GraphAF generative models
- **Generation**: Molecular generation with constraint filtering
- **Evaluation**: Comprehensive molecular evaluation metrics
- **Training**: Unified training framework with experiment management

## Installation Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- RDKit
- NumPy
- SciPy (optional, for advanced statistics)

See [requirements.txt](../requirements.txt) for complete dependencies.