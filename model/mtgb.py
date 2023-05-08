# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from Base._Base import MTCondensedGradientBoosting
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils.validation import check_array
import numpy as np

class MTGBClassifier(GradientBoostingClassifier, MTCondensedGradientBoosting):

    def __init__(self,
                 *,
                 loss='log_loss',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='squared_error',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0,
                 n_common_estimators=0):

        super().__init__(loss=loss,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_depth=max_depth,
                         init=init,
                         subsample=subsample,
                         max_features=max_features,
                         random_state=random_state,
                         verbose=verbose,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         warm_start=warm_start,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         tol=tol,
                         ccp_alpha=ccp_alpha)
        self.n_common_estimators = n_common_estimators

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        encoded_labels = \
            self._loss._raw_prediction_to_decision(raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = \
                self._loss._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    def decision_function(self, X, task_info=-1):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : ndarray of shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            :term:`classes_`. Regression and binary classification produce an
            array of shape (n_samples,).
        """
        X_full = X # not_necessary?
        X, t = self._split_task(X_full)

        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        raw_predictions = self._raw_predict(X_full)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions


class MTGBRegressor(GradientBoostingRegressor, MTCondensedGradientBoosting):

    def __init__(self,
                 *,
                 loss='ls',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='squared_error',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0,
                 metric='rmse'):

        super().__init__(loss=loss,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_depth=max_depth,
                         init=init,
                         subsample=subsample,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         random_state=random_state,
                         verbose=verbose,
                         alpha=alpha,
                         max_leaf_nodes=max_leaf_nodes,
                         warm_start=warm_start,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         tol=tol,
                         ccp_alpha=ccp_alpha)

    def predict(self, X):
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        return self._raw_predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        output_errors = np.mean((y - pred) ** 2, axis=0)
        return np.mean(np.sqrt(output_errors))
