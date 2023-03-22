import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from gridsearch import gridsearch


data = pd.read_csv('landmine_data.csv')
target = pd.read_csv('landmine_target.csv')

df = pd.merge(data, target)
df = df.drop(columns=df.columns[0])




tasks = np.unique(df.task)


param_grid = {'learning_rate': [0.025, 0.05, 0.1, 0.5, 1],
              'max_depth': [2, 5, 10, 20],
              'subsample': [0.5, 0.75, 1]}

score = []
pred = []



for task in tasks:
    data = df[df['task'] == task]
    X = (data.drop(columns=['task', 'label'])).values
    y = (data.label).values
    pred_ = np.zeros_like(y)


    model = GradientBoostingClassifier(learning_rate=0.3,
                                        max_depth=6,
                                        subsample=1,
                                        n_estimators=100)

    pred_ = gridsearch(X=X, y=y, 
                    estimator=model, 
                    classification=False, 
                    param_grid=param_grid, 
                    random_state=1)


    score.append(accuracy_score(y, pred_))
    pred.append(pred_)

with open('GBM_landmine_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for a in pred:
        writer.writerow(a)

np.savetxt('GBM_landmine_score.txt', score)
