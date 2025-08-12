# API Reference

This section provides comprehensive documentation for all MolecuGen classes and methods.

## Core Modules

### Data Processing
- [SMILESProcessor](data/smiles_processor.md) - SMILES to graph conversion
- [FeatureExtractor](data/feature_extractor.md) - Molecular feature extraction
- [MolecularDataset](data/molecular_dataset.md) - Dataset handling

### Models
- [BaseGenerativeModel](models/base_model.md) - Abstract base class for generative models
- [GraphDiffusion](models/graph_diffusion.md) - Diffusion-based molecular generation
- [GraphAF](models/graph_af.md) - Autoregressive flow model

### Generation
- [MolecularGenerator](generate/molecular_generator.md) - Main generation interface
- [ConstraintFilter](generate/constraint_filter.md) - Drug-likeness filtering

### Evaluation
- [MolecularEvaluator](evaluate/molecular_evaluator.md) - Comprehensive evaluation metrics

### Training
- [Trainer](training/trainer.md) - Main training orchestrator
- [ExperimentLogger](training/experiment_logger.md) - Experiment tracking

## Quick Reference

### Basic Usage Pattern

```python
# 1. Data Processing
from src.data.smiles_processor import SMILESProcessor
processor = SMILESProcessor()
graph = processor.smiles_to_graph("CCO")

# 2. Model Training
from src.training.trainer import Trainer
trainer = Trainer(config)
trainer.setup_data(dataset)
trainer.setup_model()
trainer.train()

# 3. Generation
from src.generate.molecular_generator import MolecularGenerator
generator = MolecularGenerator.from_checkpoint("model.pt")
molecules = generator.generate(100)

# 4. Evaluation
from src.evaluate.molecular_evaluator import MolecularEvaluator
evaluator = MolecularEvaluator()
metrics = evaluator.evaluate(molecules)
```

### Common Parameters

Most classes accept these common parameters:

- `device`: PyTorch device (cuda/cpu)
- `config`: Configuration dictionary
- `batch_size`: Batch size for processing
- `num_workers`: Number of data loading workers