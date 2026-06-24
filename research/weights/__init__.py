"""research.weights -- AFML Ch.4 sample weights (concurrency, attribution, decay, bootstrap)."""
from research.weights.attribution import normalize_weights, return_attribution
from research.weights.bootstrap import (
    get_ind_matrix,
    ind_matrix_avg_uniqueness,
    seq_bootstrap,
)
from research.weights.concurrency import avg_uniqueness, num_co_events
from research.weights.config import WeightConfig
from research.weights.decay import time_decay
from research.weights.pipeline import build_sample_weights
from research.weights.storage import list_weight_configs, load_weights, store_weights

__all__ = [
    "WeightConfig",
    "build_sample_weights",
    "store_weights",
    "load_weights",
    "list_weight_configs",
    "num_co_events",
    "avg_uniqueness",
    "return_attribution",
    "normalize_weights",
    "time_decay",
    "get_ind_matrix",
    "ind_matrix_avg_uniqueness",
    "seq_bootstrap",
]