"""research.labels — CUSUM event filter + triple-barrier labeling (AFML Ch.2-3)."""
from research.labels.barriers import Barrier
from research.labels.filters import cusum_filter

__all__ = [
    "Barrier",
    "cusum_filter",
    # Added as later tasks land: get_events, get_bins,
    # LabelConfig, store_labels, load_labels, list_label_configs
]
