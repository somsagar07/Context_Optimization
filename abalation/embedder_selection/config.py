"""
Configuration for embedding ablation experiments.
"""

# Embedders to test
EMBEDDERS_TO_TEST = [

    # MetaCLIP
    "metaclip-2-worldwide-l14",
    "metaclip-h14",

    # Multimodal embedders
    "jina-clip-v2",  # Jina CLIP v2 - multilingual multimodal (89 languages, 8192 tokens)
    "flava-full",    # FLAVA - Facebook unified vision-language model
    
    # SigLIP variants 
    "siglip-base",
    "siglip-large",

    #  CLIP variants 
    "clip-base",
    "clip-large",
    "clip-base-patch16",
    
    
    # Sentence-transformers 
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "sentence-t5-base",
    "e5-base",
]

# Projection settings
# Set to True to project all embeddings to same dimension (fair comparison, RL-ready)
# Set to False to use native dimensions (no information loss, test intrinsic quality)
PROJECT_TO_SAME_DIM = True  # Will test both True and False modes
TARGET_EMBEDDING_DIM = 512  # Only used when PROJECT_TO_SAME_DIM = True

# Datasets to use for experiments
DATASETS = ["gsm8k", "hotpotqa", "aime25", "medqa"]

# Number of samples to use per dataset (None = use all)
MAX_SAMPLES_PER_DATASET = 500  # Reduce for faster experiments

RANDOM_STATE = 789

# Experiment settings
EXPERIMENT_SETTINGS = {
    "clustering": {
        "n_clusters": 10,  # Number of clusters for K-means
        "random_state": RANDOM_STATE,
    },
    "classification": {
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "cv_folds": 5,  # Cross-validation folds
    },
    "complexity": {
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
    },
    "decision_prediction": {
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "hidden_dim": 256,
        "epochs": 50,
    },
}

# Output directory for results
RESULTS_DIR = "embedder_selection_results"

