# 🧪 MolecuGen: Diffusion-Based Graph Generator for Drug-Likeness-Constrained Molecular Generation

**MolecuGen** is a generative modeling framework for synthesizing novel, chemically valid, and drug-like molecules using graph-based diffusion models and autoregressive flows. The model operates on molecular graph representations and is trained to satisfy drug-likeness criteria such as **Lipinski’s Rule of Five**, QED score, and other chemical property constraints.

---

## 🎯 Project Objective

- Learn the distribution of drug-like molecules from datasets such as **ZINC15** and **QM9**.
- Generate molecular graphs that:
  - Are chemically valid and novel.
  - Satisfy drug-likeness constraints (Lipinski, QED, logP).
  - Can optionally optimize properties like toxicity or solubility.

---

## 📂 Directory Structure

molecugen/
├── data/
│   ├── raw/         # Original SMILES or molecular data
│   └── processed/   # Featurized molecular graph data
├── src/
│   ├── models/      # GraphDiffusion, GraphAF implementations
│   ├── utils/       # RDKit featurization, graph helpers
│   ├── training/    # Training scripts and experiment configs
│   └── evaluate/    # Evaluation metrics and postprocessing
├── notebooks/
│   └── eda.ipynb    # Dataset analysis & visualization
├── results/
│   └── molecules/   # Generated molecular graphs
├── requirements.txt
└── README.md

---

## 🔧 Key Features

- ✅ Molecular graph generation using GraphDiffusion or GraphAF
- ✅ Drug-likeness filtering using RDKit and Lipinski’s Rule of Five
- ✅ Property prediction and multi-objective conditioning (logP, QED, toxicity)
- ✅ Evaluation of validity, uniqueness, novelty, and property distributions
- ✅ Modular codebase and PyTorch Geometric support

---

## 📚 Datasets

| Dataset | Description | Source |
|--------|-------------|--------|
| ZINC15 | Drug-like molecules in SMILES format | [ZINC](https://zinc.docking.org/) |
| QM9    | Organic molecules with 19 quantum properties | [PyG Docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html) |

---

## 🧬 Lipinski’s Rule of Five

Molecules are filtered based on the following:

- Molecular weight < 500 Da  
- logP < 5  
- ≤ 5 hydrogen bond donors  
- ≤ 10 hydrogen bond acceptors  

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validity | % of chemically valid molecules |
| Uniqueness | % of unique molecules in generated set |
| Novelty | % of generated molecules not in training set |
| QED Score | Quantitative drug-likeness |
| Lipinski Pass Rate | % satisfying all 4 rules |
| Property Distribution | Histogram matching (e.g., logP, MW) |

---

## 🔍 Getting Started

### 🧱 Installation

bash
git clone https://github.com/yourusername/molecugen.git
cd molecugen
conda create -n molecugen python=3.10
conda activate molecugen
pip install -r requirements.txt

💾 Dataset Setup
	•	Download SMILES from ZINC15 or use torch_geometric.datasets.QM9
	•	Convert SMILES to graph using RDKit + custom featurizer in src/utils/rdkit_featurizer.py

🚀 Training

python src/training/train_graphdiff.py --config configs/graphdiff.yaml

🧪 Generation

python src/generate/generate_graphs.py --model-checkpoint path/to/model.ckpt


⸻

🧠 Extensions
	•	🎯 Multi-objective generation with QED + solubility + toxicity
	•	🔐 Privacy-preserving training with differential privacy
	•	🔍 Explainability with fragment attribution for generated molecules

⸻

📘 References
	1.	Graph Diffusion Models – Jo et al., 2022
	2.	GraphAF – Shi et al., 2020
	3.	RDKit Documentation
	4.	MolBERT for Molecular Property Prediction
	5.	ZINC: Irwin et al., 2012

⸻

👨‍🔬 Author

Akash
Master’s in Computer Science (AI), Case Western Reserve University


⸻

🧪 License

MIT License. See LICENSE file for details.

---

Let me know:
- If you want this auto-filled into your GitHub repository.
- Or a version with badges (e.g., Build, Python version, RDKit).
- Or to link to a Gradio app or colab demo for molecular generation.
