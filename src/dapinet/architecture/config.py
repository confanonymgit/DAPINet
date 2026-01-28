import torch


class Config:
    # --- Data Constraints ---
    MAX_ROWS = 2500
    MAX_COLS = 200
    NUM_CLASSES = 12

    # --- Model Selection ---
    # Options: "default", "dace_simple", "pool_mean", "no_col_attn", "positional"
    MODEL_TYPE = "default"

    # --- Model Architecture ---
    D_MODEL = 32
    N_HEAD = 4
    N_LAYERS = 4
    NUM_BINS = 10  # Number of quantile bins for DACE column identification
    DROPOUT = 0.3

    # --- Training ---
    BATCH_SIZE = 8  # Small batch size due to A100 memory constraints with large attention matrices
    ACCUM_STEPS = 4  # Effective batch size = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    EPOCHS = 10
    K_FOLDS = 5
    PATIENCE = 5
    NUM_WORKERS = 4

    # --- System ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "table_classifier_checkpoint.pth"

    @classmethod
    def update_from_args(cls, args):
        """Update config from parsed args."""

        if hasattr(args, "d_model"):
            cls.D_MODEL = args.d_model
        if hasattr(args, "n_head"):
            cls.N_HEAD = args.n_head
        if hasattr(args, "n_layers"):
            cls.N_LAYERS = args.n_layers
        if hasattr(args, "dropout"):
            cls.DROPOUT = args.dropout
        if hasattr(args, "lr"):
            cls.LEARNING_RATE = args.lr
        if hasattr(args, "weight_decay"):
            cls.WEIGHT_DECAY = args.weight_decay
        if hasattr(args, "batch_size"):
            cls.BATCH_SIZE = args.batch_size
        if hasattr(args, "accum_steps"):
            cls.ACCUM_STEPS = args.accum_steps
        if hasattr(args, "epochs"):
            cls.EPOCHS = args.epochs
        if hasattr(args, "k_folds"):
            cls.K_FOLDS = args.k_folds
        if hasattr(args, "patience"):
            cls.PATIENCE = args.patience
        if hasattr(args, "num_workers"):
            cls.NUM_WORKERS = args.num_workers
        if hasattr(args, "num_bins"):
            cls.NUM_BINS = args.num_bins
        if hasattr(args, "model_type"):
            cls.MODEL_TYPE = args.model_type

    @classmethod
    def to_dict(cls):
        """Return config as a dict."""
        return {k: v for k, v in cls.__dict__.items() if k.isupper() and not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict):
        """Update config from a dict."""
        for k, v in config_dict.items():
            if hasattr(cls, k.upper()):
                setattr(cls, k.upper(), v)

    def __str__(self):
        return str(self.to_dict())
