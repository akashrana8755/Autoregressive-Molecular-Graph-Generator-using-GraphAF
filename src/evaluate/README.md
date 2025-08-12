# Molecular Evaluation Framework

This module provides comprehensive evaluation metrics for generated molecules, including validity, uniqueness, novelty, and drug-likeness assessments.

## Features

### Core Metrics
- **Validity**: Percentage of chemically valid molecules (RDKit validation)
- **Uniqueness**: Percentage of unique molecules in the generated set
- **Novelty**: Percentage of molecules not present in reference dataset

### Drug-likeness Assessment
- **QED Scores**: Quantitative Estimate of Drug-likeness (0.0 to 1.0)
- **Lipinski Rule of Five**: Compliance with drug-likeness criteria
  - Molecular weight ≤ 500 Da
  - LogP ≤ 5
  - Hydrogen bond donors ≤ 5
  - Hydrogen bond acceptors ≤ 10

### Property Analysis
- **Property Distributions**: Molecular weight, LogP, atom count, etc.
- **Statistical Comparisons**: KS-test and Wasserstein distance vs. reference
- **Diversity Metrics**: Structural diversity assessment

## Usage

### Basic Evaluation

```python
from src.evaluate.molecular_evaluator import MolecularEvaluator

# Initialize evaluator
evaluator = MolecularEvaluator()

# Evaluate generated molecules
generated_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
metrics = evaluator.evaluate(generated_smiles)

print(f"Validity: {metrics['validity']:.3f}")
print(f"Uniqueness: {metrics['uniqueness']:.3f}")
```

### Drug-likeness Assessment

```python
# Compute drug-likeness metrics
drug_metrics = evaluator.compute_drug_likeness_metrics(generated_smiles)

print(f"Mean QED: {drug_metrics['mean_qed']:.3f}")
print(f"Lipinski pass rate: {drug_metrics['lipinski_pass_rate']:.3f}")

# Filter drug-like molecules
drug_like = evaluator.filter_drug_like_molecules(
    generated_smiles, 
    qed_threshold=0.5, 
    require_lipinski=True
)
```

### Comprehensive Analysis

```python
# Include reference molecules for novelty computation
reference_smiles = ['CCO', 'CC(=O)O']
evaluator = MolecularEvaluator(reference_molecules=reference_smiles)

# Generate comprehensive report
report = evaluator.generate_comprehensive_report(
    generated_smiles, 
    reference_smiles
)

# Access different sections
print("Basic metrics:", report['basic_metrics'])
print("Drug-likeness:", report['drug_likeness'])
print("Property comparison:", report['property_comparison'])
```

## Requirements

- **RDKit**: Required for molecular validation and property calculation
- **NumPy**: For numerical computations
- **SciPy**: Optional, for advanced statistical tests

Install dependencies:
```bash
pip install rdkit-pypi numpy scipy
```

## API Reference

### MolecularEvaluator

Main evaluation class providing comprehensive molecular assessment.

#### Methods

- `evaluate(molecules)`: Basic validity, uniqueness, and novelty metrics
- `compute_qed_scores(molecules)`: QED drug-likeness scores
- `compute_lipinski_compliance(molecules)`: Lipinski Rule of Five compliance
- `compute_drug_likeness_metrics(molecules)`: Comprehensive drug-likeness assessment
- `compute_property_distributions(molecules)`: Molecular property distributions
- `compare_property_distributions(generated, reference)`: Statistical comparison
- `filter_drug_like_molecules(molecules, qed_threshold, require_lipinski)`: Filter by drug-likeness
- `generate_comprehensive_report(generated, reference)`: Full evaluation report

## Examples

See `examples/evaluation_example.py` for a complete usage example.

## Testing

Run tests with:
```bash
python -m pytest tests/test_molecular_evaluator.py -v
```

Note: Tests use mocked RDKit functions and can run without RDKit installation.