from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
import numpy as np
import pandas as pd


def gridsearch(X, 
               y, 
               estimator, 
               classification, 
               param_grid, 
               random_state):

    kfold = KFold(n_splits=3, 
                  shuffle=True, 
                  random_state=random_state)

    if classification:
        scoring = 'accuracy'
    else:
        scoring = 'neg_root_mean_squared_error'

    pred = np.zeros_like(y)
    bestparams = []
    cv_results = []

    for i, (train_index, test_index) in tqdm(enumerate(kfold.split(X, y))):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid = GridSearchCV(estimator=estimator,
                            param_grid=param_grid,
                            scoring=scoring,
                            n_jobs=-1,
                            refit=True,
                            cv=3)

        grid.fit(x_train, y_train)
        pred[test_index] = grid.predict(x_test)

        bestparams.append(grid.best_params_)

        grid.cv_results_['final_test_error'] = grid.score(x_test, y_test)

        cv_results.append(grid.cv_results_)

    results = {}
    results['Metric'] = [scoring]
    results['test_score'] = grid.cv_results_[
        'mean_test_score'][grid.best_index_]
    results['Mean_generalization_score'] = grid.cv_results_[
        'final_test_error']

    # pd.DataFrame(results).to_csv('_Summary.csv')
    # pd.DataFrame(cv_results).to_csv('_CV_results.csv')
    # pd.DataFrame(bestparams).to_csv('_Best_Parameters.csv')

    return pred
