import csv
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from gridsearch import gridsearch


data = pd.read_csv('school_data.csv')
target = pd.read_csv('school_target.csv')

df = pd.merge(data, target)
df = df.drop(columns=df.columns[0])


scl = LabelBinarizer()

tasks = np.unique(df.task)


score = []
pred = []

param_grid = {'eta': [0.025, 0.05, 0.1, 0.5, 1],
              'max_depth': [2, 5, 10, 20],
              'subsample': [0.5, 0.75, 1]}


for task in tasks:
    data = df[df['task'] == task]
    X = (data.drop(columns=['task', '0'])).values
    y = (data['0']).values

    model = XGBRegressor(eta=0.3,
                            max_depth=6,
                            subsample=1,
                            n_estimators=100)

    pred_ = gridsearch(X=X, y=y, 
                       estimator=model, 
                       classification=True, 
                       param_grid=param_grid, 
                       random_state=1)

    score.append(mean_squared_error(y, pred_, squared=False))
    pred.append(pred_)

with open('xgboost_school_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for a in pred:
        writer.writerow(a)

np.savetxt('xgboost_school_score.txt', score)
