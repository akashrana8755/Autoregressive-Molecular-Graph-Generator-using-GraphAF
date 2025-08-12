# Best Practices Guide

This guide provides recommended practices and patterns for using MolecuGen effectively, covering model training, generation, evaluation, and deployment considerations.

## Model Training Best Practices

### Data Preparation

#### Dataset Quality
```python
# Always validate your dataset before training
from src.data.data_validator import DataValidator

validator = DataValidator()
validation_report = validator.validate_dataset("data/molecules.smi")

print(f"Valid molecules: {validation_report['valid_count']}")
print(f"Invalid molecules: {validation_report['invalid_count']}")
print(f"Duplicate molecules: {validation_report['duplicate_count']}")

# Remove problematic molecules
clean_molecules = validator.clean_dataset(
    "data/molecules.smi",
    remove_invalid=True,
    remove_duplicates=True,
    min_atoms=3,
    max_atoms=50
)
```

#### Data Splits
```python
# Use stratified splits for property-balanced datasets
from src.data.molecular_dataset import create_stratified_splits

train_data, val_data, test_data = create_stratified_splits(
    dataset,
    property_column='logp',  # Stratify by LogP
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
)
```

### Model Architecture Selection

#### Choose Model Type Based on Use Case
```python
# For unconditional generation with high diversity
config = {
    'model': {
        'type': 'diffusion',
        'num_timesteps': 1000,
        'beta_schedule': 'cosine'
    }
}

# For fast generation with good quality
config = {
    'model': {
        'type': 'autoregressive_flow',
        'num_flow_layers': 4,
        'coupling_type': 'affine'
    }
}
```

#### Model Size Guidelines
```python
# Small model (fast training, lower quality)
small_config = {
    'hidden_dim': 128,
    'num_layers': 3,
    'batch_size': 64
}

# Medium model (balanced)
medium_config = {
    'hidden_dim': 256,
    'num_layers': 6,
    'batch_size': 32
}

# Large model (high quality, slower training)
large_config = {
    'hidden_dim': 512,
    'num_layers': 8,
    'batch_size': 16
}
```

### Training Configuration

#### Learning Rate Scheduling
```python
# Recommended learning rate schedule
training_config = {
    'learning_rate': 1e-4,
    'scheduler': {
        'type': 'cosine',
        'eta_min': 1e-6,
        'warmup_epochs': 5
    },
    'gradient_clip': 1.0
}
```

#### Early Stopping and Checkpointing
```python
# Robust training with early stopping
training_config = {
    'patience': 20,           # Stop if no improvement for 20 epochs
    'min_delta': 1e-5,        # Minimum improvement threshold
    'save_every': 10,         # Save checkpoint every 10 epochs
    'keep_best': True,        # Always keep best model
    'keep_last': 3            # Keep 3 most recent checkpoints
}
```

### Monitoring Training

#### Essential Metrics to Track
```python
# Monitor these metrics during training
metrics_to_track = [
    'train_loss',
    'val_loss',
    'learning_rate',
    'gradient_norm',
    'generation_validity',    # Periodically generate and validate
    'generation_uniqueness'
]
```

#### Validation During Training
```python
# Periodically generate molecules during training for validation
class TrainingCallback:
    def __init__(self, generator, evaluator):
        self.generator = generator
        self.evaluator = evaluator
    
    def on_epoch_end(self, epoch, model):
        if epoch % 10 == 0:  # Every 10 epochs
            # Generate sample molecules
            model.eval()
            with torch.no_grad():
                samples = model.sample(num_samples=100)
            
            # Convert to SMILES and evaluate
            smiles = [self.generator.graph_to_smiles(g) for g in samples]
            valid_smiles = [s for s in smiles if s is not None]
            
            if valid_smiles:
                metrics = self.evaluator.evaluate(valid_smiles)
                print(f"Epoch {epoch}: Validity={metrics['validity']:.2%}, "
                      f"Uniqueness={metrics['uniqueness']:.2%}")
```

## Generation Best Practices

### Temperature and Sampling

#### Temperature Selection
```python
# Temperature guidelines
temperatures = {
    'conservative': 0.7,    # High quality, low diversity
    'balanced': 1.0,        # Good balance
    'diverse': 1.5,         # High diversity, lower quality
    'exploratory': 2.0      # Maximum diversity
}

# Adaptive temperature based on constraints
def adaptive_temperature(constraints):
    if constraints.get('strict_lipinski', False):
        return 0.8  # Lower temperature for strict constraints
    elif constraints.get('high_diversity', False):
        return 1.5  # Higher temperature for diversity
    else:
        return 1.0  # Default
```

#### Sampling Strategies
```python
# Progressive sampling with temperature cooling
def progressive_generation(generator, num_molecules, cooling_steps=5):
    molecules = []
    temp_schedule = np.linspace(2.0, 0.8, cooling_steps)
    
    for temp in temp_schedule:
        batch_molecules = generator.generate(
            num_molecules=num_molecules // cooling_steps,
            temperature=temp
        )
        molecules.extend(batch_molecules)
    
    return molecules
```

### Constraint Application

#### Hierarchical Filtering
```python
# Apply constraints in order of computational cost
def hierarchical_filtering(molecules):
    # 1. Fast structural filters first
    molecules = filter_by_size(molecules, min_atoms=5, max_atoms=50)
    molecules = filter_by_rings(molecules, max_rings=6)
    
    # 2. Medium-cost property filters
    molecules = filter_by_lipinski(molecules)
    molecules = filter_by_logp(molecules, min_logp=-2, max_logp=5)
    
    # 3. Expensive ML-based filters last
    molecules = filter_by_toxicity_prediction(molecules)
    molecules = filter_by_synthetic_accessibility(molecules)
    
    return molecules
```

#### Constraint Optimization
```python
# Iterative constraint satisfaction
def iterative_constraint_generation(generator, target_count, constraints):
    molecules = []
    max_iterations = 10
    batch_size = target_count * 2  # Generate extra for filtering
    
    for iteration in range(max_iterations):
        if len(molecules) >= target_count:
            break
            
        # Adjust temperature based on success rate
        success_rate = len(molecules) / ((iteration + 1) * batch_size)
        temperature = 1.0 + (1.0 - success_rate)  # Higher temp if low success
        
        batch = generator.generate_with_constraints(
            num_molecules=batch_size,
            constraints=constraints,
            temperature=temperature
        )
        
        molecules.extend(batch)
        molecules = remove_duplicates(molecules)
    
    return molecules[:target_count]
```

### Batch Processing

#### Memory-Efficient Generation
```python
def memory_efficient_generation(generator, total_molecules, max_batch_size=32):
    """Generate large numbers of molecules without memory issues."""
    molecules = []
    remaining = total_molecules
    
    while remaining > 0:
        current_batch = min(max_batch_size, remaining)
        
        try:
            batch_molecules = generator.generate(
                num_molecules=current_batch,
                max_attempts=3  # Limit attempts to prevent hanging
            )
            molecules.extend(batch_molecules)
            remaining -= len(batch_molecules)
            
            # Periodic cleanup
            if len(molecules) % 1000 == 0:
                torch.cuda.empty_cache()  # Clear GPU memory
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                max_batch_size = max(1, max_batch_size // 2)
                print(f"Reducing batch size to {max_batch_size}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    return molecules
```

## Evaluation Best Practices

### Comprehensive Evaluation

#### Multi-Metric Assessment
```python
def comprehensive_evaluation(generated_molecules, reference_molecules=None):
    """Perform comprehensive evaluation of generated molecules."""
    from src.evaluate.molecular_evaluator import MolecularEvaluator
    from src.generate.constraint_filter import ConstraintFilter
    
    evaluator = MolecularEvaluator(reference_molecules)
    filter_obj = ConstraintFilter()
    
    # Basic metrics
    basic_metrics = evaluator.evaluate(generated_molecules)
    
    # Drug-likeness metrics
    drug_metrics = evaluator.compute_drug_likeness_metrics(generated_molecules)
    
    # Property distributions
    prop_distributions = evaluator.compute_property_distributions(generated_molecules)
    
    # Constraint compliance
    constraint_stats = filter_obj.get_comprehensive_statistics(generated_molecules)
    
    # Diversity metrics
    diversity_metrics = evaluator.compute_diversity_metrics(generated_molecules)
    
    return {
        'basic': basic_metrics,
        'drug_likeness': drug_metrics,
        'properties': prop_distributions,
        'constraints': constraint_stats,
        'diversity': diversity_metrics
    }
```

#### Benchmark Comparisons
```python
def benchmark_against_datasets(generated_molecules):
    """Compare generated molecules against known datasets."""
    benchmarks = {
        'zinc15_drugs': load_zinc15_drug_subset(),
        'chembl_drugs': load_chembl_approved_drugs(),
        'natural_products': load_natural_products()
    }
    
    results = {}
    for name, reference in benchmarks.items():
        evaluator = MolecularEvaluator(reference)
        comparison = evaluator.compare_property_distributions(
            generated_molecules, reference
        )
        results[name] = comparison
    
    return results
```

### Statistical Validation

#### Significance Testing
```python
def statistical_validation(generated_set1, generated_set2, reference_set):
    """Perform statistical tests to validate generation quality."""
    from scipy import stats
    
    # Property distributions
    props1 = compute_properties(generated_set1)
    props2 = compute_properties(generated_set2)
    props_ref = compute_properties(reference_set)
    
    results = {}
    for prop in ['molecular_weight', 'logp', 'tpsa']:
        # Kolmogorov-Smirnov test
        ks_stat1, p_val1 = stats.ks_2samp(props1[prop], props_ref[prop])
        ks_stat2, p_val2 = stats.ks_2samp(props2[prop], props_ref[prop])
        
        results[prop] = {
            'set1_ks_pvalue': p_val1,
            'set2_ks_pvalue': p_val2,
            'better_set': 'set1' if p_val1 > p_val2 else 'set2'
        }
    
    return results
```

## Performance Optimization

### GPU Memory Management

#### Memory-Efficient Training
```python
# Gradient accumulation for large effective batch sizes
def train_with_gradient_accumulation(model, dataloader, optimizer, 
                                   effective_batch_size=128, actual_batch_size=32):
    accumulation_steps = effective_batch_size // actual_batch_size
    
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        loss = model.training_step(batch) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

#### Memory Monitoring
```python
def monitor_gpu_memory():
    """Monitor GPU memory usage during training/generation."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        if allocated > 10:  # Warning if using >10GB
            print("Warning: High GPU memory usage detected")
```

### Computational Efficiency

#### Batch Size Optimization
```python
def find_optimal_batch_size(model, sample_batch, max_batch_size=128):
    """Find the largest batch size that fits in memory."""
    batch_size = 1
    
    while batch_size <= max_batch_size:
        try:
            # Create batch of current size
            test_batch = create_batch_of_size(sample_batch, batch_size)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(test_batch)
            
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_batch_size = batch_size // 2
                torch.cuda.empty_cache()
                return optimal_batch_size
            else:
                raise e
    
    return max_batch_size
```

## Deployment Considerations

### Model Serving

#### Production-Ready Generator
```python
class ProductionMolecularGenerator:
    """Production-ready molecular generator with error handling and monitoring."""
    
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = MolecularGenerator.from_checkpoint(checkpoint_path, device=self.device)
        self.stats = {'total_requests': 0, 'successful_generations': 0}
    
    def generate_safe(self, num_molecules, **kwargs):
        """Generate molecules with comprehensive error handling."""
        self.stats['total_requests'] += 1
        
        try:
            # Input validation
            if num_molecules <= 0 or num_molecules > 10000:
                raise ValueError("num_molecules must be between 1 and 10000")
            
            # Generation with timeout
            molecules = self.generator.generate(
                num_molecules=num_molecules,
                max_attempts=5,  # Limit attempts
                **kwargs
            )
            
            # Validation
            if len(molecules) < num_molecules * 0.5:  # Less than 50% success
                logger.warning(f"Low generation success rate: {len(molecules)}/{num_molecules}")
            
            self.stats['successful_generations'] += len(molecules)
            return molecules
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return []
    
    def get_health_status(self):
        """Return health status for monitoring."""
        success_rate = (self.stats['successful_generations'] / 
                       max(self.stats['total_requests'], 1))
        
        return {
            'status': 'healthy' if success_rate > 0.8 else 'degraded',
            'success_rate': success_rate,
            'total_requests': self.stats['total_requests']
        }
```

### API Design

#### RESTful API Example
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
generator = ProductionMolecularGenerator('models/production_model.pt')

@app.route('/generate', methods=['POST'])
def generate_molecules():
    try:
        data = request.json
        num_molecules = data.get('num_molecules', 100)
        constraints = data.get('constraints', {})
        temperature = data.get('temperature', 1.0)
        
        molecules = generator.generate_safe(
            num_molecules=num_molecules,
            constraints=constraints,
            temperature=temperature
        )
        
        return jsonify({
            'success': True,
            'molecules': molecules,
            'count': len(molecules)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(generator.get_health_status())
```

## Troubleshooting Common Issues

### Training Issues

#### Loss Not Decreasing
```python
# Debugging checklist for training issues
def debug_training_issues(model, dataloader):
    # 1. Check data loading
    batch = next(iter(dataloader))
    print(f"Batch size: {batch.batch.max().item() + 1}")
    print(f"Node features shape: {batch.x.shape}")
    print(f"Edge features shape: {batch.edge_attr.shape}")
    
    # 2. Check model forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)
        print(f"Model output keys: {output.keys()}")
    
    # 3. Check loss computation
    model.train()
    loss = model.training_step(batch)
    print(f"Training loss: {loss.item()}")
    
    # 4. Check gradients
    loss.backward()
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")
```

#### Memory Issues
```python
# Memory optimization strategies
def optimize_memory_usage():
    # 1. Reduce batch size
    # 2. Use gradient accumulation
    # 3. Enable mixed precision training
    # 4. Clear cache regularly
    
    torch.cuda.empty_cache()
    
    # Monitor memory usage
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Generation Issues

#### Low Validity Rate
```python
def improve_validity_rate(generator):
    # 1. Lower temperature
    molecules = generator.generate(num_molecules=100, temperature=0.8)
    
    # 2. Increase max_attempts
    molecules = generator.generate(num_molecules=100, max_attempts=20)
    
    # 3. Post-process with validation
    valid_molecules = [m for m in molecules if validate_molecule(m)]
    
    return valid_molecules
```

#### Poor Constraint Satisfaction
```python
def improve_constraint_satisfaction(generator, constraints):
    # 1. Use iterative generation
    molecules = generator.iterative_constraint_generation(
        num_molecules=100,
        constraints=constraints,
        max_iterations=10
    )
    
    # 2. Adjust constraint thresholds
    relaxed_constraints = constraints.copy()
    relaxed_constraints['qed_threshold'] *= 0.9  # Relax by 10%
    
    return molecules
```

## Code Quality and Testing

### Unit Testing
```python
# Example unit tests for molecular generation
import unittest

class TestMolecularGeneration(unittest.TestCase):
    def setUp(self):
        self.generator = MolecularGenerator.from_checkpoint('test_model.pt')
    
    def test_basic_generation(self):
        molecules = self.generator.generate(num_molecules=10)
        self.assertGreater(len(molecules), 0)
        self.assertLessEqual(len(molecules), 10)
    
    def test_constraint_generation(self):
        constraints = {'lipinski': True}
        molecules = self.generator.generate_with_constraints(
            num_molecules=10, constraints=constraints
        )
        # Verify all molecules pass Lipinski rules
        for mol in molecules:
            self.assertTrue(passes_lipinski(mol))
```

### Integration Testing
```python
def test_end_to_end_pipeline():
    """Test complete pipeline from training to generation."""
    # 1. Train small model
    config = create_test_config()
    trainer = Trainer(config)
    trainer.setup_data(test_dataset)
    trainer.setup_model()
    results = trainer.train()
    
    # 2. Generate molecules
    generator = MolecularGenerator.from_checkpoint(trainer.best_checkpoint)
    molecules = generator.generate(num_molecules=50)
    
    # 3. Evaluate results
    evaluator = MolecularEvaluator()
    metrics = evaluator.evaluate(molecules)
    
    # Assert minimum quality thresholds
    assert metrics['validity'] > 0.5
    assert metrics['uniqueness'] > 0.8
```

This comprehensive best practices guide should help users effectively utilize MolecuGen for their molecular generation tasks while avoiding common pitfalls and optimizing performance.