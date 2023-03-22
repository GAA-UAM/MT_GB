import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelBinarizer

data = pd.read_csv('school_data.csv')
target = pd.read_csv('school_target.csv')

df = pd.merge(data, target)
df = df.drop(columns=df.columns[0])


scl = LabelBinarizer()
df['0'] = scl.fit_transform(df['0'])

tasks = np.unique(df.task)


score = []
pred = []

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
param_grid = {'eta': [0.025, 0.05, 0.1, 0.5, 1],
              'max_depth': [2, 5, 10, 20],
              'subsample': [0.5, 0.75, 1]}




for task in tasks:
    data = df[df['task'] == task]
    X = (data.drop(columns=['task', '0'])).values
    y = (data['0']).values
    pred_ = np.zeros_like(y)
    for i, (train_index, test_index) in tqdm(enumerate(kfold.split(X, y))):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = XGBClassifier(eta=0.3,
                              max_depth=6,
                              subsample=1,
                              n_estimators=5)

        grid = GridSearchCV(estimator=model, param_grid=param_grid,
                            scoring='accuracy', n_jobs=-1, refit=True)

        grid.fit(x_train, y_train)
        pred_[test_index] = grid.predict(x_test)
    score.append(accuracy_score(y, pred_))
    pred.append(pred_)

with open('xgboost_school_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for a in pred:
        writer.writerow(a)

np.savetxt('xgboost_school_score.txt', score)
