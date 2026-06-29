"""MyPipeline + cv_score -- the cross-validation orchestrator (AFML Snippet
7.4 / Ch.7, Step 5).

The convergence point of research/validation: this module is the first
thing in the sub-package that actually CONSUMES config, purge, and both
splitters together. It contains no purge/embargo/splitting logic of its
own -- it only routes sample_weight through fitting (MyPipeline) and
drives the fit-predict-score loop over whatever splitter it's given
(cv_score).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from research.validation.splitters import PurgedKFold

__all__ = ["MyPipeline", "cv_score"]

_VALID_SCORING = {"neg_log_loss", "accuracy"}


class MyPipeline(Pipeline):
    """A Pipeline whose fit() accepts sample_weight directly (AFML 7.4).

    sklearn's stock ``Pipeline.fit`` only accepts ``sample_weight`` via
    the verbose ``{final_step_name}__sample_weight`` fit-param key, which
    requires the caller to know the pipeline's final step name. This
    class exists purely to remove that requirement: ``fit`` is the ONLY
    method overridden -- ``predict``, ``score``, ``fit_transform``, and
    ``__init__`` all inherit ``Pipeline``'s behavior unchanged.

    When ``sample_weight`` is ``None`` or absent, this is a strict
    superset of ``Pipeline.fit`` -- it delegates straight through with no
    key-rewrite, so it is safe to use as the default pipeline class even
    when weights aren't supplied.
    """

    def fit(self, X, y=None, sample_weight=None, **fit_params):
        """Fit the pipeline, routing sample_weight to the final step.

        If ``sample_weight`` is not ``None``, it is rewritten to
        ``{final_step_name}__sample_weight`` in ``fit_params`` before
        delegating to ``Pipeline.fit`` -- sklearn's own fit-param
        namespacing then ensures it reaches only the final step, never
        any earlier transformer.
        """
        if sample_weight is not None:
            final_step_name = self.steps[-1][0]
            fit_params[f"{final_step_name}__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


def cv_score(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    cv=None,
    scoring: str = "neg_log_loss",
    n_splits: int = 3,
    embargo_pct: float = 0.0,
) -> np.ndarray:
    """Cross-validation score, leak-proof by construction (AFML Ch.7).

    Pure compute, no I/O, no aggregation -- returns the raw per-fold score
    array; the caller decides how (or whether) to aggregate it. This
    matters most for CombinatorialPurgedKFold, where naively averaging
    across combinatorial paths is the WRONG thing to do (Phase-2
    path-reconstruction territory) -- aggregating here would bake in an
    assumption that doesn't hold for every splitter this function can
    drive.

    Parameters
    ----------
    clf:
        An estimator or ``Pipeline``. If ``clf`` is not already a
        ``Pipeline`` (bare estimator), it is wrapped in a one-step
        :class:`MyPipeline` so ``sample_weight`` can be routed uniformly
        regardless of what was passed in. An estimator already wrapped
        in a ``Pipeline`` (including a plain sklearn ``Pipeline``, not
        only ``MyPipeline``) is used as-is.
    X:
        Feature frame.
    y:
        Labels, aligned with ``X``.
    sample_weight:
        Per-observation weights, aligned with ``X``/``y`` -- sliced into
        ``sample_weight[train]``/``sample_weight[test]`` per fold and
        threaded into BOTH fitting and scoring. Dropping the test-side
        weighting would silently ignore sample uniqueness on the
        evaluation side, defeating half the point of the AFML weights
        layer this sub-package consumes.
    t1:
        Event end timestamps, aligned with ``X``/``y`` -- used to build
        the default splitter when ``cv`` is not supplied.
    cv:
        An optional pre-constructed splitter (e.g. :class:`PurgedKFold`
        or :class:`research.validation.cpcv.CombinatorialPurgedKFold`).
        If ``None``, a :class:`PurgedKFold` is built from ``n_splits``,
        ``t1``, and ``embargo_pct``. This is what lets the SAME
        ``cv_score`` drive either splitter interchangeably.
    scoring:
        ``"neg_log_loss"`` (default) or ``"accuracy"``.
    n_splits, embargo_pct:
        Only used to build the default :class:`PurgedKFold` when ``cv``
        is ``None``; ignored if ``cv`` is supplied.

    Returns
    -------
    np.ndarray
        One score per fold, in the order the splitter yields them.

    Raises
    ------
    ValueError
        If ``scoring`` is not one of ``{"neg_log_loss", "accuracy"}``.
    AttributeError
        If ``scoring == "neg_log_loss"`` and the (possibly-wrapped)
        estimator has no ``predict_proba`` -- fails loud before any
        fold is fit, rather than silently falling back to accuracy.
    """
    if scoring not in _VALID_SCORING:
        raise ValueError(
            f"cv_score scoring must be one of {_VALID_SCORING}, got "
            f"{scoring!r}."
        )

    if not isinstance(clf, Pipeline):
        clf = MyPipeline([("clf", clf)])

    if scoring == "neg_log_loss" and not hasattr(clf, "predict_proba"):
        raise AttributeError(
            "cv_score scoring='neg_log_loss' requires the estimator to "
            "implement predict_proba; this estimator does not."
        )

    splitter = (
        cv if cv is not None else PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
    )

    scores = []
    for train_pos, test_pos in splitter.split(X):
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
        w_train, w_test = sample_weight.iloc[train_pos], sample_weight.iloc[test_pos]

        clf.fit(X_train, y_train, sample_weight=w_train)

        if scoring == "neg_log_loss":
            prob = clf.predict_proba(X_test)
            # labels=clf.classes_ explicitly, NOT log_loss's default
            # inference from y_test alone -- a test fold missing a class
            # the classifier trained on would otherwise mismatch the
            # predict_proba column count and raise.
            score = -log_loss(
                y_test, prob, sample_weight=w_test, labels=clf.classes_
            )
        else:
            pred = clf.predict(X_test)
            score = accuracy_score(y_test, pred, sample_weight=w_test)

        scores.append(score)

    return np.array(scores)
