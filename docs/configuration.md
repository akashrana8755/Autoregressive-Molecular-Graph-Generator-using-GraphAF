# Configuration Guide

MolecuGen uses YAML configuration files to manage model parameters, training settings, and generation options. This guide provides comprehensive documentation of all configuration options and best practices.

## Configuration Structure

```yaml
# config/example_config.yaml
name: "molecugen_experiment"
output_dir: "experiments"

model:
  type: "diffusion"  # or "autoregressive_flow"
  # Model-specific parameters...

training:
  batch_size: 32
  num_epochs: 100
  # Training-specific parameters...

data:
  dataset: "zinc15"
  # Data-specific parameters...

generation:
  num_samples: 1000
  # Generation-specific parameters...

constraints:
  lipinski: true
  # Constraint-specific parameters...

logging:
  level: "INFO"
  # Logging-specific parameters...
```

## Model Configuration

### GraphDiffusion Model

```yaml
model:
  type: "diffusion"
  
  # Architecture parameters
  node_dim: 128          # Node feature dimension (auto-detected from data)
  edge_dim: 64           # Edge feature dimension (auto-detected from data)
  hidden_dim: 256        # Hidden layer dimension
  num_layers: 6          # Number of GNN layers
  dropout: 0.1           # Dropout rate
  
  # Diffusion-specific parameters
  num_timesteps: 1000    # Number of diffusion timesteps
  beta_schedule: "cosine" # Noise schedule: "linear", "cosine", "sigmoid"
  beta_start: 0.0001     # Starting noise level
  beta_end: 0.02         # Ending noise level
  
  # Graph constraints
  max_nodes: 50          # Maximum nodes per molecule
  max_edges: 100         # Maximum edges per molecule
  
  # Advanced options
  self_condition: false  # Self-conditioning for improved sampling
  learned_variance: false # Learn variance in addition to mean
  clip_denoised: true    # Clip denoised values to valid range
```

### GraphAF Model

```yaml
model:
  type: "autoregressive_flow"
  
  # Architecture parameters
  node_dim: 128          # Node feature dimension
  edge_dim: 64           # Edge feature dimension
  hidden_dim: 256        # Hidden layer dimension
  num_layers: 4          # Number of GNN layers
  dropout: 0.1           # Dropout rate
  
  # Flow-specific parameters
  num_flow_layers: 4     # Number of normalizing flow layers
  coupling_type: "affine" # Coupling layer type: "additive", "affine"
  mask_type: "checkerboard" # Masking pattern for coupling layers
  
  # Autoregressive parameters
  node_order: "random"   # Node ordering: "random", "canonical", "bfs"
  edge_order: "random"   # Edge ordering strategy
  
  # Graph constraints
  max_nodes: 50          # Maximum nodes per molecule
  
  # Advanced options
  use_edge_features: true # Whether to use edge features in flows
  temperature_annealing: false # Anneal temperature during training
```

## Training Configuration

```yaml
training:
  # Basic training parameters
  batch_size: 32         # Training batch size
  num_epochs: 100        # Number of training epochs
  learning_rate: 1e-4    # Initial learning rate
  weight_decay: 1e-5     # L2 regularization weight
  
  # Optimization settings
  optimizer:
    type: "adam"         # Optimizer: "adam", "adamw", "sgd", "rmsprop"
    betas: [0.9, 0.999]  # Adam beta parameters
    eps: 1e-8            # Adam epsilon
    amsgrad: false       # Use AMSGrad variant
  
  scheduler:
    type: "cosine"       # Scheduler: "cosine", "step", "exponential", "plateau"
    eta_min: 1e-6        # Minimum learning rate for cosine
    step_size: 30        # Step size for step scheduler
    gamma: 0.1           # Decay factor for step/exponential
    patience: 10         # Patience for plateau scheduler
    factor: 0.5          # Factor for plateau scheduler
  
  # Regularization
  gradient_clip: 1.0     # Gradient clipping norm
  dropout: 0.1           # Dropout rate (if not specified in model)
  
  # Training dynamics
  warmup_epochs: 5       # Learning rate warmup epochs
  patience: 15           # Early stopping patience
  min_delta: 1e-6        # Minimum improvement for early stopping
  
  # Validation and checkpointing
  validate_every: 1      # Validation frequency (epochs)
  save_every: 10         # Checkpoint saving frequency (epochs)
  keep_best: true        # Keep best model checkpoint
  keep_last: 3           # Number of recent checkpoints to keep
  
  # Loss function parameters (model-specific)
  loss_type: "mse"       # Loss type for diffusion: "mse", "l1", "huber"
  node_weight: 1.0       # Weight for node prediction loss
  edge_weight: 1.0       # Weight for edge prediction loss
  timestep_weighting: "uniform" # Timestep weighting: "uniform", "snr"
  
  # Reproducibility
  seed: 42               # Random seed
  deterministic: false   # Use deterministic algorithms (slower)
  
  # Hardware settings
  num_workers: 4         # Number of data loading workers
  pin_memory: true       # Pin memory for faster GPU transfer
  mixed_precision: false # Use automatic mixed precision
```

## Data Configuration

```yaml
data:
  # Dataset selection
  dataset: "zinc15"      # Dataset: "zinc15", "qm9", "custom"
  data_path: "data/"     # Path to dataset files
  
  # Data splits
  train_split: 0.8       # Training set fraction
  val_split: 0.1         # Validation set fraction
  test_split: 0.1        # Test set fraction
  
  # Preprocessing options
  max_nodes: 50          # Maximum nodes per molecule
  min_nodes: 3           # Minimum nodes per molecule
  remove_invalid: true   # Remove invalid molecules
  canonicalize: true     # Canonicalize SMILES strings
  
  # Feature extraction
  use_chirality: true    # Include chirality features
  use_partial_charge: false # Include partial charge features (slower)
  explicit_hydrogens: false # Include explicit hydrogens
  
  # Data augmentation
  augment_data: false    # Enable data augmentation
  augmentation_factor: 2 # Augmentation multiplier
  rotation_prob: 0.5     # Probability of random rotation
  
  # Caching and performance
  cache_processed: true  # Cache processed molecules
  cache_dir: "cache/"    # Cache directory
  preload_data: false    # Preload all data into memory
  
  # Custom dataset options (if dataset: "custom")
  smiles_file: "molecules.smi" # SMILES file path
  property_file: null    # Optional property file
  delimiter: "\t"        # File delimiter
  smiles_column: 0       # SMILES column index
  header: true           # File has header row
```

## Generation Configuration

```yaml
generation:
  # Basic generation parameters
  num_samples: 1000      # Number of molecules to generate
  batch_size: 32         # Generation batch size
  temperature: 1.0       # Sampling temperature
  
  # Sampling parameters
  max_nodes: 50          # Maximum nodes per molecule
  min_nodes: 3           # Minimum nodes per molecule
  max_attempts: 10       # Maximum generation attempts per molecule
  
  # Diffusion-specific sampling
  num_inference_steps: 1000 # Number of denoising steps
  eta: 0.0               # DDIM eta parameter (0 = DDIM, 1 = DDPM)
  guidance_scale: 1.0    # Classifier-free guidance scale
  
  # Flow-specific sampling
  temperature_schedule: null # Temperature annealing schedule
  top_k: null            # Top-k sampling (if specified)
  top_p: null            # Nucleus sampling threshold
  
  # Post-processing
  remove_duplicates: true # Remove duplicate molecules
  canonicalize_output: true # Canonicalize output SMILES
  validate_output: true  # Validate generated molecules
  
  # Output options
  output_format: "smi"   # Output format: "smi", "sdf", "json"
  save_graphs: false     # Save molecular graphs
  save_properties: true  # Save computed properties
```

## Constraint Configuration

```yaml
constraints:
  # Lipinski's Rule of Five
  lipinski: true         # Apply Lipinski rules
  mw_threshold: 500.0    # Molecular weight threshold (Da)
  logp_threshold: 5.0    # LogP threshold
  hbd_threshold: 5       # H-bond donor threshold
  hba_threshold: 10      # H-bond acceptor threshold
  
  # Drug-likeness scores
  qed_threshold: 0.5     # Minimum QED score
  sa_threshold: null     # Synthetic accessibility threshold
  
  # Property ranges
  mw_range: [100, 600]   # Molecular weight range
  logp_range: [-2, 6]    # LogP range
  tpsa_range: [0, 200]   # TPSA range
  rotatable_bonds_max: 10 # Maximum rotatable bonds
  
  # Structural constraints
  max_rings: 6           # Maximum number of rings
  max_aromatic_rings: 4  # Maximum aromatic rings
  allow_charged: true    # Allow charged molecules
  allow_radicals: false  # Allow radical species
  
  # Custom filters
  substructure_filters:  # Unwanted substructures (SMARTS)
    - "[N+](=O)[O-]"     # Nitro groups
    - "S(=O)(=O)"        # Sulfonyl groups
  
  required_substructures: # Required substructures
    - "c1ccccc1"         # Benzene ring (example)
  
  # PAINS filters
  apply_pains_filter: true # Remove PAINS compounds
  pains_severity: "A"    # PAINS severity: "A", "B", "C"
```

## Logging Configuration

```yaml
logging:
  # Basic logging
  level: "INFO"          # Log level: "DEBUG", "INFO", "WARNING", "ERROR"
  log_file: "training.log" # Log file path
  console_output: true   # Enable console logging
  
  # Experiment tracking
  use_wandb: false       # Use Weights & Biases
  wandb_project: "molecugen" # W&B project name
  wandb_entity: null     # W&B entity (username/team)
  wandb_tags: []         # W&B tags
  
  use_tensorboard: true  # Use TensorBoard
  tensorboard_dir: "runs/" # TensorBoard log directory
  
  # Metrics logging
  log_frequency: 100     # Log frequency (steps)
  save_samples: true     # Save generated samples
  sample_frequency: 1000 # Sample saving frequency (steps)
  num_samples_log: 10    # Number of samples to log
  
  # Model monitoring
  log_gradients: false   # Log gradient statistics
  log_weights: false     # Log weight statistics
  log_activations: false # Log activation statistics
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# config/dev_config.yaml
name: "dev_experiment"

model:
  type: "diffusion"
  hidden_dim: 128        # Smaller model for faster iteration
  num_layers: 3

training:
  batch_size: 16         # Smaller batch for limited GPU memory
  num_epochs: 10         # Fewer epochs for quick testing
  validate_every: 1
  save_every: 5

data:
  train_split: 0.1       # Use subset of data for development
  val_split: 0.05
  test_split: 0.05

logging:
  level: "DEBUG"         # Verbose logging for debugging
  use_wandb: false       # Disable external logging
```

### Production Configuration

```yaml
# config/prod_config.yaml
name: "production_model"

model:
  type: "diffusion"
  hidden_dim: 512        # Larger model for better performance
  num_layers: 8
  num_timesteps: 2000    # More timesteps for quality

training:
  batch_size: 64         # Larger batch for stability
  num_epochs: 200        # More epochs for convergence
  learning_rate: 5e-5    # Lower learning rate for stability
  patience: 30           # More patience for convergence

data:
  train_split: 0.9       # Use most data for training
  val_split: 0.05
  test_split: 0.05
  cache_processed: true  # Cache for faster loading

generation:
  num_samples: 10000     # Generate large sets
  batch_size: 128        # Larger batch for efficiency

logging:
  use_wandb: true        # Enable experiment tracking
  wandb_project: "molecugen_production"
```

## Best Practices

### Model Selection

```yaml
# For fast prototyping
model:
  type: "diffusion"
  hidden_dim: 128
  num_layers: 3
  num_timesteps: 100

# For high-quality generation
model:
  type: "diffusion"
  hidden_dim: 512
  num_layers: 8
  num_timesteps: 1000
  beta_schedule: "cosine"
```

### Training Optimization

```yaml
training:
  # Use learning rate scheduling
  scheduler:
    type: "cosine"
    eta_min: 1e-6
  
  # Enable gradient clipping
  gradient_clip: 1.0
  
  # Use early stopping
  patience: 20
  min_delta: 1e-5
  
  # Optimize batch size for your hardware
  batch_size: 32  # Adjust based on GPU memory
```

### Data Preprocessing

```yaml
data:
  # Remove problematic molecules
  remove_invalid: true
  min_nodes: 3
  max_nodes: 50
  
  # Canonicalize for consistency
  canonicalize: true
  
  # Cache processed data
  cache_processed: true
```

### Generation Quality

```yaml
generation:
  # Use appropriate temperature
  temperature: 1.0  # 1.0 for diversity, <1.0 for quality
  
  # Enable post-processing
  remove_duplicates: true
  validate_output: true
  
  # Apply constraints
constraints:
  lipinski: true
  qed_threshold: 0.5
```

## Configuration Validation

MolecuGen automatically validates configuration files and provides helpful error messages:

```python
from src.training.config_manager import ConfigManager

# Load and validate configuration
config_manager = ConfigManager()
config = config_manager.load_config("config/my_config.yaml")

# Validation errors will be reported with suggestions
```

## Environment Variables

Some settings can be overridden with environment variables:

```bash
# Override output directory
export MOLECUGEN_OUTPUT_DIR="/path/to/experiments"

# Override device
export MOLECUGEN_DEVICE="cuda:1"

# Override logging level
export MOLECUGEN_LOG_LEVEL="DEBUG"

# Override W&B settings
export WANDB_PROJECT="my_project"
export WANDB_ENTITY="my_team"
```

## Configuration Templates

Use these templates as starting points:

- `config/diffusion_default.yaml` - Default diffusion model configuration
- `config/graphaf_default.yaml` - Default GraphAF configuration
- `config/generation_default.yaml` - Default generation configuration
- `config/test_config.yaml` - Configuration for testing and development