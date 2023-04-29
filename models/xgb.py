import sys
import numpy as np
from Base._BaseXGB import XGBoost


class MTXGBoostClassifier(XGBoost):
    def __init__(self, loss_function="squared_error",
                 n_estimators=5,
                 min_samples_split=3,
                 min_samples_leaf=2,
                 max_depth=6,
                 max_samples_linear_model=sys.maxsize,
                 subsample=1,
                 learning_rate=0.3,
                 min_split_loss=0,
                 gamma=0,
                 lbda=0,
                 prune=True,
                 random_state=None,
                 verbose=0,
                 nthread=1):
        
        super().__init__(loss_function,
                         n_estimators,
                         min_samples_split,
                         min_samples_leaf,
                         max_depth,
                         max_samples_linear_model,
                         subsample,
                         learning_rate,
                         min_split_loss,
                         gamma,
                         lbda,
                         prune,
                         random_state,
                         verbose,
                         nthread)


class MTXGBoostRegreesor(XGBoost):

    def __init__(self, loss_function="squared_error",
                 n_estimators=5,
                 min_samples_split=3,
                 min_samples_leaf=2,
                 max_depth=6,
                 max_samples_linear_model=sys.maxsize,
                 subsample=1,
                 learning_rate=0.3,
                 min_split_loss=0,
                 gamma=0,
                 lbda=0,
                 prune=True,
                 random_state=None,
                 verbose=0,
                 nthread=1):
        super().__init__(loss_function,
                         n_estimators,
                         min_samples_split,
                         min_samples_leaf,
                         max_depth,
                         max_samples_linear_model,
                         subsample,
                         learning_rate,
                         min_split_loss,
                         gamma,
                         lbda,
                         prune,
                         random_state,
                         verbose,
                         nthread)

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=float)
        if not self.trees:
            return pred
        for t in range(len(self.trees)-1):
            pred += self.learning_rate*self.trees[t].predict(X)
        pred += self.trees[-1].predict(X)

        return pred
