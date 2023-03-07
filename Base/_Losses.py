
import numpy as np
from sklearn.tree import _tree
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import _gb_losses

from scipy.special import logsumexp
from sklearn.utils.multiclass import type_of_target

TREE_LEAF = _tree.TREE_LEAF
DTYPE = _tree.DTYPE


class CondensedDeviance(_gb_losses.ClassificationLossFunction):
    def __init__(self, n_classes_):
        self.K = 1
        self.n_classes_ = n_classes_
        self.is_multi_class = False

    def init_estimator(self):
        return DummyClassifier(strategy="prior")

    def __call__(self, y, raw_predictions, sample_weight=None):

        return np.average(-1 * (y * raw_predictions).sum(axis=1) +
                          logsumexp(raw_predictions, axis=1),
                          weights=sample_weight)

    def update_terminal_regions(self,
                                tree,
                                X,
                                y,
                                residual,
                                raw_predictions,
                                sample_weight,
                                sample_mask,
                                learning_rate,
                                k=0):

        terminal_regions = tree.apply(X)

        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions, leaf,
                                         X, y, residual, raw_predictions,
                                         sample_weight)

        raw_predictions[:, :] += \
            (learning_rate * tree.value[:, :, 0]
             ).take(terminal_regions, axis=0)

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):

        return y - np.nan_to_num(
            np.exp(raw_predictions -
                   logsumexp(raw_predictions, axis=1, keepdims=True)))

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        n_classes = self.n_classes_
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        sample_weight = sample_weight[:, np.newaxis]
        numerator = np.sum(sample_weight * residual, axis=0)
        numerator *= (n_classes - 1) / n_classes
        denominator = np.sum(sample_weight * (y - residual) *
                             (1 - y + residual),
                             axis=0)
        tree.value[leaf, :, 0] = np.where(
            abs(denominator) < 1e-150, 0.0, numerator / denominator)

    def _raw_prediction_to_proba(self, raw_predictions):
        return np.nan_to_num(
            np.exp(raw_predictions -
                   (logsumexp(raw_predictions, axis=1)[:, np.newaxis])))

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions

