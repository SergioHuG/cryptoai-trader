"""research.labels — CUSUM event filter + triple-barrier labeling (AFML Ch.2-3)."""
from research.labels.barriers import Barrier, get_bins, get_events
from research.labels.config import LabelConfig
from research.labels.filters import cusum_filter
from research.labels.pipeline import build_triple_barrier_labels
from research.labels.storage import list_label_configs, load_labels, store_labels

__all__ = [
    "Barrier",
    "cusum_filter",
    "get_events",
    "get_bins",
    "LabelConfig",
    "build_triple_barrier_labels",
    "store_labels",
    "load_labels",
    "list_label_configs",
]