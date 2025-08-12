#!/usr/bin/env python3
"""
Example usage of the MolecularEvaluator for comprehensive molecule evaluation.

This script demonstrates how to use the evaluation framework to assess
generated molecules for validity, uniqueness, novelty, and drug-likeness.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate.molecular_evaluator import MolecularEvaluator


def main():
    """Demonstrate molecular evaluation functionality."""
    
    # Example molecules (some valid, some invalid, some duplicates)
    generated_molecules = [
        'CCO',                    # ethanol - valid
        'CC(=O)O',               # acetic acid - valid
        'c1ccccc1',              # benzene - valid
        'CCO',                   # duplicate ethanol
        'invalid_smiles',        # invalid
        'CC(C)O',                # isopropanol - valid
        'CCN(CC)CC',             # triethylamine - valid
        'OCC',                   # ethanol (different representation)
    ]
    
    # Reference molecules for novelty computation
    reference_molecules = [
        'CCO',                   # ethanol (known)
        'CC(=O)O',              # acetic acid (known)
        'CCCCCCCCCC',           # decane (known)
    ]
    
    print("=== Molecular Evaluation Example ===\n")
    
    try:
        # Initialize evaluator with reference molecules
        evaluator = MolecularEvaluator(reference_molecules=reference_molecules)
        
        print("Generated molecules:")
        for i, smiles in enumerate(generated_molecules, 1):
            print(f"  {i}. {smiles}")
        
        print(f"\nReference molecules: {len(reference_molecules)} molecules")
        print()
        
        # Basic evaluation
        print("=== Basic Evaluation Metrics ===")
        basic_metrics = evaluator.evaluate(generated_molecules)
        
        for metric, value in basic_metrics.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"{metric}: {value:.3f}")
                else:
                    print(f"{metric}: {value}")
            else:
                print(f"{metric}: N/A")
        
        print()
        
        # Drug-likeness evaluation
        print("=== Drug-likeness Metrics ===")
        drug_metrics = evaluator.compute_drug_likeness_metrics(generated_molecules)
        
        for metric, value in drug_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        print()
        
        # Filter drug-like molecules
        print("=== Drug-like Molecule Filtering ===")
        drug_like = evaluator.filter_drug_like_molecules(
            generated_molecules, 
            qed_threshold=0.5, 
            require_lipinski=True
        )
        
        print(f"Drug-like molecules (QED >= 0.5, Lipinski compliant): {len(drug_like)}")
        for smiles in drug_like:
            print(f"  - {smiles}")
        
        print()
        
        # Comprehensive report
        print("=== Comprehensive Report ===")
        report = evaluator.generate_comprehensive_report(
            generated_molecules, 
            reference_molecules
        )
        
        print("Report sections:")
        for section in report.keys():
            print(f"  - {section}")
        
        print(f"\nSummary:")
        summary = report['summary']
        print(f"  Total generated: {summary['total_generated']}")
        print(f"  Valid molecules: {summary['valid_count']}")
        print(f"  Unique molecules: {summary['unique_count']}")
        print(f"  Validity rate: {summary['validity_rate']:.3f}")
        print(f"  Uniqueness rate: {summary['uniqueness_rate']:.3f}")
        
        print("\n=== Evaluation Complete ===")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nNote: This example requires RDKit to be installed.")
        print("Install with: pip install rdkit-pypi")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()