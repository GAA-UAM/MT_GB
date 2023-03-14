from numbers import Integral
import numpy as np
from sklearn.tree import _tree
from ._Losses import CondensedDeviance, MultiOutputLeastSquaresError

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble._gb import BaseGradientBoosting, VerboseReporter
from sklearn.ensemble import _gradient_boosting

from scipy.sparse.base import issparse
from sklearn.utils.multiclass import type_of_target
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection._split import train_test_split
from sklearn.utils.validation import check_array, check_random_state, column_or_1d, _check_sample_weight

from icecream import ic

DTYPE = _tree.DTYPE

            
class MTCondensedGradientBoosting(BaseGradientBoosting):

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 loss="log_loss",
                 criterion="squared_error",
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 subsample=1.0,
                 max_features=None,
                 max_depth=5,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=0.0001,
                 init=None,
                 random_state=None,
                 n_common_estimators=0):

        super().__init__(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         loss=loss,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         subsample=subsample,
                         max_features=max_features,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         ccp_alpha=ccp_alpha,
                         alpha=alpha,
                         verbose=verbose,
                         max_leaf_nodes=max_leaf_nodes,
                         warm_start=warm_start,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         tol=tol,
                         init=init,
                         random_state=random_state)
        self.n_common_estimators = n_common_estimators
        
    def _init_state(self):
        """Initialize model state and allocate model state data structures."""

        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self._loss.init_estimator()

        self.estimators_ = np.empty((self.n_estimators, self.T, self._loss.K), dtype=object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators), dtype=np.float64)

    def _fit_stage(self,
                   i,
                   r, # task
                   X,
                   y,
                   raw_predictions,
                   sample_weight,
                   sample_mask,
                   random_state,
                   X_csc=None,
                   X_csr=None):

        assert sample_mask.dtype == bool
        loss = self._loss

        original_y = y

        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y,
                                              raw_predictions_copy,
                                              k=k,
                                              sample_weight=sample_weight)

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha)

            if self.subsample < 1.0:

                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X,
                     residual,
                     sample_weight=sample_weight,
                     check_input=False)

            loss.update_terminal_regions(tree.tree_,
                                         X,
                                         y,
                                         residual,
                                         raw_predictions,
                                         sample_weight,
                                         sample_mask,
                                         learning_rate=self.learning_rate,
                                         k=k)

            if i >= self.n_common_estimators:
                self.estimators_[i, r, k] = tree
            else:
                self.estimators_[i, 0, k] = tree # We use the 0-th estimator for all tasks

        return raw_predictions

    def _fit_stages(self,
                    X,
                    t,
                    y,
                    raw_predictions,
                    sample_weight,
                    random_state,
                    X_val,
                    y_val,
                    sample_weight_val,
                    begin_at_stage=0,
                    monitor=None):
        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self._loss

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # Transform the y shape, if the problem is classification
        if type_of_target(y) == 'multiclass':
            y = LabelBinarizer().fit_transform(y)
        elif type_of_target(y) == 'binary':
            Y = np.zeros((y.shape[0], 2), dtype=np.float64)
            for k in range(2):
                Y[:, k] = y == k
            y = Y

        # perform boosting iterations
        i = begin_at_stage
        _random_sample_mask = _gradient_boosting._random_sample_mask
        for i in range(begin_at_stage, self.n_estimators):
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, # n_samples_r instead of n_samples
                                                random_state)
            if i >= self.n_common_estimators:
                for r_label, r in self.tasks_dic.items():
                    idx_r = (t == r_label)
                    X_r = X[idx_r]
                    X_csc_r = X_csc[idx_r] if issparse(X) else None
                    X_csr_r = X_csr[idx_r] if issparse(X) else None
                    y_r = y[idx_r]
                    raw_predictions_r = raw_predictions[idx_r]
                    sample_weight_r = sample_weight[idx_r]
                    sample_mask_r = sample_mask[idx_r]
                    # subsampling
                    if do_oob:                        
                        # OOB score before adding this stage
                        old_oob_score = loss_(y_r[~sample_mask_r],
                                            raw_predictions_r[~sample_mask_r],
                                            sample_weight_r[~sample_mask_r])

                    # fit next stage of trees
                    raw_predictions[idx_r] = self._fit_stage(i, r, X_r, y_r, raw_predictions_r,
                                                    sample_weight_r, sample_mask_r,
                                                    random_state, X_csc_r, X_csr_r)
            else:
                if do_oob:                        
                    # OOB score before adding this stage
                    old_oob_score = loss_(y[~sample_mask],
                                        raw_predictions[~sample_mask],
                                        sample_weight[~sample_mask])

                    # fit next stage of trees
                raw_predictions = self._fit_stage(i, None, X, y, raw_predictions,
                                                sample_weight, sample_mask,
                                                random_state, X_csc, X_csr)
            

            # track loss
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                            raw_predictions[sample_mask],
                                            sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score -
                    loss_(y[~sample_mask], raw_predictions[~sample_mask],
                        sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1

    def _check_params(self):

        if self.loss == 'log_loss':
            loss_class = CondensedDeviance
        else:
            loss_class = MultiOutputLeastSquaresError

        if self.loss == 'log_loss':
            self._loss = loss_class(self.n_classes_)
        elif self.loss in ("huber", "quantile"):
            self._loss = loss_class(self.alpha)
        else:
            self._loss = loss_class()

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classifier(self):
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            else:  # self.max_features == "log2"
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:  # float
            max_features = max(1, int(self.max_features * self.n_features_in_))

        self.max_features_ = max_features


    def _split_task(self, X):
        X_task = X[:, self.task_info]
        X_data = np.delete(X, self.task_info, axis=1).astype(float)
        return X_data, X_task

    def fit(self, X, y, sample_weight=None, monitor=None, task_info=-1):

        if not self.warm_start:
            self._clear_state()
        
        # Separate the task information from the data
        if not isinstance(task_info, int):
            raise ArithmeticError('task_info must be an integer')
        self.task_info = task_info
        
        X_full = X # not_necessary?
        X, t = self._split_task(X_full)

        unique = np.unique(t)
        self.T = len(unique)
        self.tasks_dic = dict(zip(unique, range(self.T)))
        
        # validate data
        X, y = self._validate_data(X,
                                   y,
                                   accept_sparse=['csr', 'csc', 'coo'],
                                   dtype=DTYPE,
                                   multi_output=True)

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_sample_weight(sample_weight, X)

        if is_classifier(self):
            y = column_or_1d(y, warn=True)
            y = self._validate_y(y, sample_weight)
        else:
            y = self._validate_y(y)

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X,
                                 y,
                                 sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction,
                                 stratify=stratify))
            if is_classifier(self):
                if self.n_classes_ != np.unique(y).shape[0]:
                    raise ValueError(
                        'The training data after the early stopping split '
                        'is missing some classes. Try using another random '
                        'seed.')
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if not self._is_initialized():

            self._init_state()

            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self._loss.K),
                                           dtype=np.float64)
            else:
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError:  # regular estimator without SW support
                        raise ValueError(msg)
                    except ValueError as e:
                        if "pass parameters to specific steps of "\
                            "your pipeline using the "\
                                "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = \
                    self._loss.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:

            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    'n_estimators=%d must be larger or equal to '
                    'estimators_.shape[0]=%d when '
                    'warm_start==True' %
                    (self.n_estimators, self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            # ic(raw_predictions)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(X, t, y, raw_predictions, sample_weight,
                                    self._rng, X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor)

        # changne shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()

        # separate tasks and data
        X_full = X
        X_data, t = self._split_task(X_full)
        unique = np.unique(t)
        
        if self.n_common_estimators > 0:
            X_data = self.estimators_[0, 0, 0]._validate_X_predict(X_data, check_input=True) # We use the 0-th index (common estim) for the validation
        else:
            for r_label in unique:
                idx_r = (t == r_label)
                X_data[idx_r] = self.estimators_[0, self.tasks_dic[r_label], 0]._validate_X_predict(X_data[idx_r], check_input=True)
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self._loss.K),
                                       dtype=np.float64)
        else:
            raw_predictions = self._loss.get_init_raw_predictions(
                X_data, self.init_).astype(np.float64)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X) # init is common for all tasks

        # separate tasks and data
        X_full = X
        X, t = self._split_task(X_full)
        unique = np.unique(t)

        if raw_predictions.shape[1] == 1:
            raw_predictions = np.squeeze(raw_predictions)        
        for i in range(self.n_estimators):
            if i >= self.n_common_estimators:
                for r_label in unique:
                    if r_label not in self.tasks_dic:
                        raise ValueError("The task {} was not present in the training set".format(r_label))
                    tree = self.estimators_[i, self.tasks_dic[r_label], 0]
                    idx_r = (t == r_label)
                    raw_predictions[idx_r] += (self.learning_rate * tree.predict(X[idx_r]))
            else:
                tree = self.estimators_[i, 0, 0] # The common estimator is in the 0-th index
                raw_predictions += (self.learning_rate * tree.predict(X))

        return raw_predictions

    def _staged_raw_predict(self, X):
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict_init(X) # init is common for all tasks

        # separate tasks and data
        X_full = X
        X, t = self._split_task(X_full)
        unique = np.unique(t)

        if raw_predictions.shape[1] == 1:
            raw_predictions = np.squeeze(raw_predictions)
        for i in range(self.n_estimators):
            if i >= self.n_common_estimators:
                for r_label in unique:
                    if r_label not in self.tasks_dic:
                        raise ValueError("The task {} was not present in the training set".format(r_label))
                    tree = self.estimators_[i, self.tasks_dic[r_label], 0]
                    idx_r = (t == r_label)
                    raw_predictions[idx_r] += (self.learning_rate * tree.predict(X[idx_r]))
            else:
                tree = self.estimators_[i, 0, 0] # The common estimator is in the 0-th index
                raw_predictions += (self.learning_rate * tree.predict(X))

        yield raw_predictions.copy()
