# MolecuGen

Graph-based molecular generation system using diffusion models and autoregressive flows for drug-likeness-constrained molecular generation.

## Overview

MolecuGen learns the distribution of drug-like molecules from datasets like ZINC15 and QM9, then generates novel, chemically valid molecules that satisfy drug-likeness constraints such as Lipinski's Rule of Five, QED scores, and other chemical property constraints.

## Features

- **Graph-based Generation**: Uses molecular graph representations with atoms as nodes and bonds as edges
- **Multiple Model Types**: Supports GraphDiffusion (SDE-based) and GraphAF (autoregressive flows)
- **Drug-likeness Constraints**: Enforces Lipinski's Rule of Five and QED score filtering
- **Property Conditioning**: Generate molecules with specific target properties
- **Comprehensive Evaluation**: Validity, uniqueness, novelty, and drug-likeness metrics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd molecugen

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from src.config import ConfigManager
from src.models import GraphDiffusion
from src.generate import MolecularGenerator

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("graphdiff_default")

# Initialize model and generator
model = GraphDiffusion(config.model)
generator = MolecularGenerator(model)

# Generate molecules
molecules = generator.generate(num_molecules=100)
```

## Project Structure

```
molecugen/
├── src/
│   ├── data/          # Data processing and molecular representations
│   ├── models/        # Generative model implementations
│   ├── training/      # Training infrastructure
│   ├── generate/      # Molecular generation engine
│   ├── evaluate/      # Evaluation framework
│   └── config.py      # Configuration management
├── config/            # YAML configuration files
├── scripts/           # Command-line utilities
├── tests/             # Test suite
└── requirements.txt   # Dependencies
```

## Configuration

The system uses YAML configuration files for all parameters. See `config/` directory for examples:

- `graphdiff_default.yaml`: GraphDiffusion model configuration
- `graphaf_default.yaml`: GraphAF model configuration  
- `test_config.yaml`: Small-scale configuration for testing

## License

MIT License