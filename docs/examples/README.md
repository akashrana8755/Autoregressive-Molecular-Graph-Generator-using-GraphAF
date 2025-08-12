# Examples and Tutorials

This directory contains comprehensive examples and tutorials for using MolecuGen effectively.

## Quick Start Examples

### Basic Usage
```python
# Simple molecule generation
from src.generate.molecular_generator import MolecularGenerator

generator = MolecularGenerator.from_checkpoint('model.pt')
molecules = generator.generate(num_molecules=100)
print(f"Generated {len(molecules)} molecules")
```

### Drug-like Generation
```python
# Generate drug-like molecules with constraints
constraints = {
    'lipinski': True,
    'qed_threshold': 0.5
}

drug_molecules = generator.generate_with_constraints(
    num_molecules=100,
    constraints=constraints
)
```

## Detailed Examples

- [Basic Training](basic_training.md) - Train your first model
- [Advanced Generation](advanced_generation.md) - Sophisticated generation strategies
- [Constraint Optimization](constraint_optimization.md) - Optimize constraint satisfaction
- [Property Targeting](property_targeting.md) - Generate molecules with specific properties
- [Evaluation and Analysis](evaluation_analysis.md) - Comprehensive evaluation workflows
- [Production Deployment](production_deployment.md) - Deploy models in production

## Jupyter Notebooks

- [End-to-End Tutorial](notebooks/end_to_end_tutorial.ipynb) - Complete workflow
- [Visualization Examples](notebooks/visualization_examples.ipynb) - Molecule visualization
- [Performance Benchmarking](notebooks/performance_benchmarking.ipynb) - Model comparison

## Use Case Examples

- [Drug Discovery](use_cases/drug_discovery.md) - Pharmaceutical applications
- [Chemical Space Exploration](use_cases/chemical_space.md) - Explore chemical diversity
- [Lead Optimization](use_cases/lead_optimization.md) - Optimize existing compounds

## Configuration Examples

- [Development Config](configs/dev_config.yaml) - For development and testing
- [Production Config](configs/prod_config.yaml) - For production deployment
- [GPU Cluster Config](configs/cluster_config.yaml) - For distributed training

## API Integration Examples

- [REST API](api_examples/rest_api.py) - Flask-based API
- [Batch Processing](api_examples/batch_processing.py) - Large-scale generation
- [Real-time Generation](api_examples/realtime_generation.py) - Low-latency serving