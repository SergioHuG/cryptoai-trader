"""Acceptance tests for research/validation/cv.py -- MyPipeline + cv_score
(AFML Snippet 7.4 / Ch.7, Step 5).

This is the convergence point: cv_score is the first thing in this
sub-package that actually CONSUMES config, purge, and both splitters
together.

5a -- MyPipeline(Pipeline) (Q7): fit-only override routing sample_weight
to the final estimator via the `{step}__sample_weight` key-rewrite,
None-passthrough making it a strict Pipeline superset.

  Override-scope testing note: sklearn's available_if-decorated methods
  (predict, score, ...) are re-wrapped by a descriptor on EVERY subclass
  access -- `SomeSubclass.predict is Pipeline.predict` is False even for
  a subclass that overrides nothing at all (verified empirically before
  writing these tests). Identity comparison against the parent class is
  therefore NOT a valid way to prove "we didn't override this" -- the
  tests below inspect `MyPipeline.__dict__` directly instead, which
  reflects only what MyPipeline itself defines.

5b -- cv_score (Q8): injected-or-defaulted splitter, sample_weight
threaded into both fit and score, neg_log_loss default with fail-loud on
missing predict_proba, explicit clf.classes_ to log_loss, raw per-fold
ndarray, no aggregation, no I/O. Bare (non-Pipeline) estimators are
wrapped in a one-step MyPipeline (Q7d).

Every cv_score test below compares against an INDEPENDENTLY computed
oracle (calling sklearn's own log_loss/accuracy_score directly on
manually-sliced folds) rather than a hand-derived number -- this proves
cv_score's wiring (right slices, right weights, right labels, right
sign) without needing to trust arithmetic done by hand. The
fold-missing-a-class scenario was verified empirically to actually crash
sklearn's log_loss without an explicit `labels=` argument, before being
locked in as a test.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from research.validation.cpcv import CombinatorialPurgedKFold
from research.validation.cv import MyPipeline, cv_score
from research.validation.splitters import PurgedKFold

_BARS = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(len(_BARS))}, index=_BARS)


def _t1() -> pd.Series:
    return pd.Series({b: b for b in _BARS})


def _weights() -> pd.Series:
    return pd.Series(np.linspace(0.5, 1.5, 12), index=_BARS)


# ── MyPipeline fixtures (Step 5a) ───────────────────────────────────────────


class _SpyEstimator(BaseEstimator):
    """Records every fit() call's sample_weight; predicts a constant."""

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y=None, sample_weight=None):
        self.fit_calls.append({"sample_weight": sample_weight})
        return self

    def predict(self, X):
        return np.zeros(len(X))


class _IdentityTransformer(BaseEstimator, TransformerMixin):
    """Records its own fit() calls -- used to prove sample_weight is NOT
    routed to non-final pipeline steps."""

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y=None, sample_weight=None):
        self.fit_calls.append({"sample_weight": sample_weight})
        return self

    def transform(self, X):
        return X


class TestMyPipelineSampleWeightRouting:
    def test_sample_weight_reaches_the_final_estimator(self):
        spy = _SpyEstimator()
        pipe = MyPipeline([("clf", spy)])
        X, y, w = _feature_frame(), pd.Series(np.arange(12) % 2), _weights()

        pipe.fit(X, y, sample_weight=w)

        assert spy.fit_calls[-1]["sample_weight"] is w

    def test_non_final_steps_never_receive_sample_weight(self):
        identity = _IdentityTransformer()
        spy = _SpyEstimator()
        pipe = MyPipeline([("identity", identity), ("clf", spy)])
        X, y, w = _feature_frame(), pd.Series(np.arange(12) % 2), _weights()

        pipe.fit(X, y, sample_weight=w)

        assert identity.fit_calls[-1]["sample_weight"] is None
        assert spy.fit_calls[-1]["sample_weight"] is w

    def test_absent_sample_weight_behaves_as_plain_pipeline_fit(self):
        spy = _SpyEstimator()
        pipe = MyPipeline([("clf", spy)])
        X, y = _feature_frame(), pd.Series(np.arange(12) % 2)

        pipe.fit(X, y)  # no sample_weight kwarg at all

        assert spy.fit_calls[-1]["sample_weight"] is None

    def test_explicit_none_sample_weight_behaves_the_same_as_absent(self):
        spy = _SpyEstimator()
        pipe = MyPipeline([("clf", spy)])
        X, y = _feature_frame(), pd.Series(np.arange(12) % 2)

        pipe.fit(X, y, sample_weight=None)

        assert spy.fit_calls[-1]["sample_weight"] is None

    def test_fit_returns_self_for_chaining(self):
        pipe = MyPipeline([("clf", _SpyEstimator())])
        X, y = _feature_frame(), pd.Series(np.arange(12) % 2)

        result = pipe.fit(X, y)

        assert result is pipe


class TestMyPipelineOverrideScope:
    """fit-only override (Q7b) -- verified via __dict__ inspection, not
    identity comparison (see module docstring)."""

    def test_fit_is_defined_on_mypipeline_itself(self):
        assert "fit" in MyPipeline.__dict__

    def test_predict_is_not_overridden(self):
        assert "predict" not in MyPipeline.__dict__

    def test_score_is_not_overridden(self):
        assert "score" not in MyPipeline.__dict__

    def test_fit_transform_is_not_overridden(self):
        assert "fit_transform" not in MyPipeline.__dict__

    def test_init_is_not_overridden(self):
        assert "__init__" not in MyPipeline.__dict__

    def test_is_a_pipeline_subclass(self):
        assert issubclass(MyPipeline, Pipeline)


# ── cv_score fixtures (Step 5b) ─────────────────────────────────────────────


class _DeterministicClf(BaseEstimator, ClassifierMixin):
    """predict_proba/predict depend only on X's 'x' column value, never on
    training data -- enabling exact oracle comparison without a real
    fitted model. Also records every fit() call for spying."""

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y, sample_weight=None):
        self.fit_calls.append(
            {
                "sample_weight": None if sample_weight is None else sample_weight.copy(),
                "index": list(X.index),
            }
        )
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_proba(self, X):
        p1 = np.where(X["x"].values % 3 == 0, 0.9, 0.2)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        p1 = np.where(X["x"].values % 3 == 0, 0.9, 0.2)
        return (p1 >= 0.5).astype(int)


class _NoProbaClf(BaseEstimator, ClassifierMixin):
    """A classifier with NO predict_proba at all -- for the fail-loud
    guard tests."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _mixed_y() -> pd.Series:
    """Alternating-ish labels with enough variety that every fold's train
    subset sees both classes under the default 3-fold split."""
    return pd.Series([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], index=_BARS)


class TestCvScoreScoringValidation:
    def test_invalid_scoring_raises(self):
        with pytest.raises(ValueError):
            cv_score(
                _DeterministicClf(),
                _feature_frame(),
                _mixed_y(),
                _weights(),
                _t1(),
                n_splits=3,
                scoring="bogus",
            )


class TestCvScorePredictProbaGuard:
    def test_neg_log_loss_on_no_proba_estimator_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            cv_score(
                _NoProbaClf(),
                _feature_frame(),
                _mixed_y(),
                _weights(),
                _t1(),
                n_splits=3,
                scoring="neg_log_loss",
            )

    def test_accuracy_on_the_same_no_proba_estimator_works_fine(self):
        """Proves the guard is scoring-specific, not a blanket
        compatibility check on the estimator."""
        scores = cv_score(
            _NoProbaClf(),
            _feature_frame(),
            _mixed_y(),
            _weights(),
            _t1(),
            n_splits=3,
            scoring="accuracy",
        )
        assert len(scores) == 3


class TestCvScoreOracleEquivalence:
    """Every assertion compares cv_score's output against an
    independently computed oracle (same deterministic splitter + same
    deterministic classifier + sklearn's own metric functions) rather
    than a hand-derived number."""

    def _oracle(self, scoring, X, y, w, splitter):
        scores = []
        for train_pos, test_pos in splitter.split(X):
            clf = _DeterministicClf()
            clf.fit(X.iloc[train_pos], y.iloc[train_pos], sample_weight=w.iloc[train_pos])
            if scoring == "neg_log_loss":
                prob = clf.predict_proba(X.iloc[test_pos])
                scores.append(
                    -log_loss(
                        y.iloc[test_pos],
                        prob,
                        sample_weight=w.iloc[test_pos],
                        labels=clf.classes_,
                    )
                )
            else:
                pred = clf.predict(X.iloc[test_pos])
                scores.append(
                    accuracy_score(y.iloc[test_pos], pred, sample_weight=w.iloc[test_pos])
                )
        return np.array(scores)

    def test_neg_log_loss_matches_independent_oracle(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        actual = cv_score(
            _DeterministicClf(), X, y, w, t1, n_splits=3, embargo_pct=0.0, scoring="neg_log_loss"
        )
        expected = self._oracle(
            "neg_log_loss", X, y, w, PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)
        )
        np.testing.assert_allclose(actual, expected)

    def test_accuracy_matches_independent_oracle(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        actual = cv_score(
            _DeterministicClf(), X, y, w, t1, n_splits=3, embargo_pct=0.0, scoring="accuracy"
        )
        expected = self._oracle(
            "accuracy", X, y, w, PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)
        )
        np.testing.assert_allclose(actual, expected)

    def test_nonuniform_weights_actually_change_the_score(self):
        """Guards against an accidentally-uniform fixture that would let
        a no-op weighting bug pass silently."""
        X, y, t1 = _feature_frame(), _mixed_y(), _t1()
        weighted = cv_score(
            _DeterministicClf(), X, y, _weights(), t1, n_splits=3, scoring="neg_log_loss"
        )
        unweighted = cv_score(
            _DeterministicClf(),
            X,
            y,
            pd.Series(1.0, index=_BARS),
            t1,
            n_splits=3,
            scoring="neg_log_loss",
        )
        assert not np.allclose(weighted, unweighted)


class TestCvScoreSampleWeightThreading:
    def test_fit_receives_the_correct_per_fold_weight_slice(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        clf = _DeterministicClf()
        pkf = PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)

        cv_score(clf, X, y, w, t1, n_splits=3, embargo_pct=0.0, scoring="neg_log_loss")

        for i, (train_pos, _) in enumerate(pkf.split(X)):
            expected = w.iloc[train_pos]
            actual = clf.fit_calls[i]["sample_weight"]
            assert actual.equals(expected)


class TestCvScoreExplicitClassesLabels:
    def test_fold_with_a_test_set_missing_a_class_does_not_crash(self):
        """Test fold positions [4,5,6,7] are all y=0 -- TRAIN still sees
        both classes (clf.classes_=[0,1]), but TEST has only class 0.
        Verified empirically that sklearn's log_loss raises ValueError on
        exactly this fold without an explicit labels= argument; cv_score
        must pass clf.classes_ to avoid it."""
        y = pd.Series([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1], index=_BARS)
        X, w, t1 = _feature_frame(), _weights(), _t1()

        scores = cv_score(
            _DeterministicClf(), X, y, w, t1, n_splits=3, embargo_pct=0.0, scoring="neg_log_loss"
        )

        assert len(scores) == 3
        assert np.all(np.isfinite(scores))


class TestCvScoreBareEstimatorWrapping:
    def test_bare_estimator_matches_manually_wrapped_mypipeline(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()

        bare = cv_score(
            _DeterministicClf(), X, y, w, t1, n_splits=3, embargo_pct=0.0, scoring="neg_log_loss"
        )
        manually_wrapped = cv_score(
            MyPipeline([("clf", _DeterministicClf())]),
            X,
            y,
            w,
            t1,
            n_splits=3,
            embargo_pct=0.0,
            scoring="neg_log_loss",
        )

        np.testing.assert_allclose(bare, manually_wrapped)


class TestCvScoreSplitterInjection:
    """Q8: injected-or-defaulted splitter -- the SAME cv_score drives
    both PurgedKFold (default) and an injected CombinatorialPurgedKFold."""

    def test_default_path_uses_n_splits_to_build_purgedkfold(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        scores = cv_score(_DeterministicClf(), X, y, w, t1, n_splits=3, scoring="neg_log_loss")
        assert len(scores) == 3

    def test_injected_cpcv_splitter_is_honored_not_ignored(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        cpkf = CombinatorialPurgedKFold(n_groups=4, n_test_groups=2, t1=t1, embargo_pct=0.0)

        scores = cv_score(
            _DeterministicClf(), X, y, w, t1, cv=cpkf, scoring="neg_log_loss"
        )

        assert len(scores) == 6  # C(4,2) -- distinct from the n_splits=3 default


class TestCvScoreReturnShape:
    def test_returns_a_plain_ndarray(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        scores = cv_score(_DeterministicClf(), X, y, w, t1, n_splits=3, scoring="neg_log_loss")
        assert isinstance(scores, np.ndarray)

    def test_no_aggregation_one_score_per_fold(self):
        X, y, w, t1 = _feature_frame(), _mixed_y(), _weights(), _t1()
        scores = cv_score(_DeterministicClf(), X, y, w, t1, n_splits=3, scoring="neg_log_loss")
        assert scores.shape == (3,)
