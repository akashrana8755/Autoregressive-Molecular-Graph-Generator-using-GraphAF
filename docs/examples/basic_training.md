# Basic Training Example

This example demonstrates how to train a molecular generation model from scratch using MolecuGen.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample dataset (optional)
python scripts/download_zinc15_subset.py --size 10000 --output data/zinc15_10k.smi
```

## Step 1: Prepare Your Data

```python
from src.data.molecular_dataset import MolecularDataset
from src.data.smiles_processor import SMILESProcessor
from src.data.feature_extractor import FeatureExtractor

# Load SMILES data
smiles_file = "data/zinc15_10k.smi"
with open(smiles_file, 'r') as f:
    smiles_list = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(smiles_list)} SMILES strings")

# Initialize processors
smiles_processor = SMILESProcessor(sanitize=True)
feature_extractor = FeatureExtractor(use_chirality=True)

# Create dataset
dataset = MolecularDataset(
    smiles_list=smiles_list,
    smiles_processor=smiles_processor,
    feature_extractor=feature_extractor,
    max_nodes=50
)

print(f"Created dataset with {len(dataset)} valid molecules")
```

## Step 2: Configure Training

```python
# Create training configuration
config = {
    'name': 'my_first_model',
    'output_dir': 'experiments',
    
    'model': {
        'type': 'diffusion',
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'num_timesteps': 1000,
        'beta_schedule': 'cosine',
        'max_nodes': 50
    },
    
    'training': {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        'patience': 15,
        'save_every': 10,
        'validate_every': 1,
        
        'optimizer': {
            'type': 'adam',
            'betas': [0.9, 0.999]
        },
        
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        }
    },
    
    'data': {
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1
    },
    
    'logging': {
        'level': 'INFO',
        'use_tensorboard': True,
        'use_wandb': False  # Set to True if you have W&B account
    }
}
```

## Step 3: Initialize and Train

```python
from src.training.trainer import Trainer
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize trainer
trainer = Trainer(config=config, device=device)

# Setup data
trainer.setup_data(dataset)
print("Data setup completed")

# Setup model
trainer.setup_model()
print("Model setup completed")

# Setup optimizer
trainer.setup_optimizer()
print("Optimizer setup completed")

# Start training
print("Starting training...")
results = trainer.train()

print(f"Training completed!")
print(f"Best validation loss: {results['best_val_loss']:.6f}")
print(f"Total epochs: {results['total_epochs']}")
print(f"Training time: {results['total_time']:.2f} seconds")
```

## Step 4: Monitor Training Progress

```python
# View training history
import matplotlib.pyplot as plt

history = trainer.training_history

plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(history['epochs'], history['train_losses'], label='Train Loss')
plt.plot(history['epochs'], history['val_losses'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')

# Plot learning rate
plt.subplot(1, 2, 2)
plt.plot(history['epochs'], history['learning_rates'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()
```

## Step 5: Evaluate Trained Model

```python
# Evaluate on test set
test_results = trainer.evaluate()
print(f"Test loss: {test_results['test_loss']:.6f}")

# Generate sample molecules
from src.generate.molecular_generator import MolecularGenerator

# Load best model
best_checkpoint = trainer.output_dir / "checkpoints" / "best_model.pt"
generator = MolecularGenerator.from_checkpoint(best_checkpoint, device=device)

# Generate molecules
sample_molecules = generator.generate(num_molecules=100, temperature=1.0)
print(f"Generated {len(sample_molecules)} molecules")

# Display some examples
print("\nSample generated molecules:")
for i, smiles in enumerate(sample_molecules[:10]):
    print(f"{i+1:2d}: {smiles}")
```

## Step 6: Validate Generation Quality

```python
from src.evaluate.molecular_evaluator import MolecularEvaluator

# Create evaluator with training data as reference
train_smiles = [dataset[i].smiles for i in range(len(dataset)) if hasattr(dataset[i], 'smiles')]
evaluator = MolecularEvaluator(reference_molecules=train_smiles)

# Evaluate generated molecules
evaluation_results = evaluator.evaluate(sample_molecules)

print("\nGeneration Quality Metrics:")
print(f"Validity: {evaluation_results['validity']:.2%}")
print(f"Uniqueness: {evaluation_results['uniqueness']:.2%}")
print(f"Novelty: {evaluation_results['novelty']:.2%}")

# Drug-likeness evaluation
drug_metrics = evaluator.compute_drug_likeness_metrics(sample_molecules)
print(f"\nDrug-likeness Metrics:")
print(f"Mean QED: {drug_metrics['mean_qed']:.3f}")
print(f"Lipinski pass rate: {drug_metrics['lipinski_pass_rate']:.2%}")
```

## Complete Training Script

Here's a complete script that combines all the steps:

```python
#!/usr/bin/env python3
"""
Complete training script for MolecuGen.
Usage: python train_basic_model.py --data data/zinc15_10k.smi --output experiments/basic_model
"""

import argparse
import torch
from pathlib import Path

from src.data.molecular_dataset import MolecularDataset
from src.data.smiles_processor import SMILESProcessor
from src.data.feature_extractor import FeatureExtractor
from src.training.trainer import Trainer
from src.generate.molecular_generator import MolecularGenerator
from src.evaluate.molecular_evaluator import MolecularEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train a basic molecular generation model')
    parser.add_argument('--data', required=True, help='Path to SMILES file')
    parser.add_argument('--output', default='experiments/basic_model', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data}")
    with open(args.data, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(smiles_list)} SMILES strings")
    
    # Create dataset
    smiles_processor = SMILESProcessor(sanitize=True)
    feature_extractor = FeatureExtractor(use_chirality=True)
    
    dataset = MolecularDataset(
        smiles_list=smiles_list,
        smiles_processor=smiles_processor,
        feature_extractor=feature_extractor,
        max_nodes=50
    )
    
    print(f"Created dataset with {len(dataset)} valid molecules")
    
    # Training configuration
    config = {
        'name': 'basic_model',
        'output_dir': args.output,
        
        'model': {
            'type': 'diffusion',
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.1,
            'num_timesteps': 1000,
            'beta_schedule': 'cosine',
            'max_nodes': 50
        },
        
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'patience': 15,
            'save_every': 10,
            'validate_every': 1,
            
            'optimizer': {'type': 'adam', 'betas': [0.9, 0.999]},
            'scheduler': {'type': 'cosine', 'eta_min': 1e-6}
        },
        
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        
        'logging': {
            'level': 'INFO',
            'use_tensorboard': True,
            'use_wandb': False
        }
    }
    
    # Initialize and train
    trainer = Trainer(config=config, device=device)
    
    print("Setting up training...")
    trainer.setup_data(dataset)
    trainer.setup_model()
    trainer.setup_optimizer()
    
    print("Starting training...")
    results = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Training time: {results['total_time']:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_results = trainer.evaluate()
    print(f"Test loss: {test_results['test_loss']:.6f}")
    
    # Generate and evaluate sample molecules
    print("\nGenerating sample molecules...")
    best_checkpoint = Path(args.output) / "checkpoints" / "best_model.pt"
    generator = MolecularGenerator.from_checkpoint(best_checkpoint, device=device)
    
    sample_molecules = generator.generate(num_molecules=1000, temperature=1.0)
    print(f"Generated {len(sample_molecules)} molecules")
    
    # Evaluate generation quality
    train_smiles = [dataset[i].smiles for i in range(len(dataset)) if hasattr(dataset[i], 'smiles')]
    evaluator = MolecularEvaluator(reference_molecules=train_smiles)
    
    evaluation_results = evaluator.evaluate(sample_molecules)
    drug_metrics = evaluator.compute_drug_likeness_metrics(sample_molecules)
    
    print(f"\nGeneration Quality:")
    print(f"Validity: {evaluation_results['validity']:.2%}")
    print(f"Uniqueness: {evaluation_results['uniqueness']:.2%}")
    print(f"Novelty: {evaluation_results['novelty']:.2%}")
    print(f"Mean QED: {drug_metrics['mean_qed']:.3f}")
    print(f"Lipinski pass rate: {drug_metrics['lipinski_pass_rate']:.2%}")
    
    print(f"\nModel saved to: {args.output}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
```

## Running the Training

Save the complete script as `train_basic_model.py` and run:

```bash
# Basic training
python train_basic_model.py --data data/zinc15_10k.smi

# With custom parameters
python train_basic_model.py \
    --data data/zinc15_10k.smi \
    --output experiments/my_model \
    --epochs 200 \
    --batch-size 64 \
    --device cuda
```

## Monitoring Training

### TensorBoard
```bash
# Start TensorBoard to monitor training
tensorboard --logdir experiments/basic_model/runs
```

### Weights & Biases (Optional)
```python
# Enable W&B logging in config
config['logging']['use_wandb'] = True
config['logging']['wandb_project'] = 'molecugen'
```

## Next Steps

After completing basic training:

1. **Experiment with hyperparameters** - Try different model sizes, learning rates
2. **Advanced generation** - Explore constraint-based generation
3. **Model comparison** - Train multiple models and compare performance
4. **Production deployment** - Set up model serving for applications

See [Advanced Generation](advanced_generation.md) for more sophisticated generation strategies.