import unittest
from sklearn.metrics import accuracy_score, f1_score
from models.cgb import MTcgb_clf
from data.DataLoader import DataLoader
import numpy as np
from sklearn.utils.validation import check_is_fitted
from icecream import ic
import random

class TestMILSVC(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(data_dir='_data')
        X, y = self.data_loader.load_dataset('landmine')
        self.X = X
        self.y = y.flatten()
        ic(y)
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        
    # def test_fit(self):
    #     arr = np.arange(self.X.shape[0])
    #     np.random.shuffle(arr)
    #     rnd_idx = arr[:]
    #     model = MTcgb_clf()
    #     model.fit(self.X[rnd_idx], self.y[rnd_idx])
    #     check_is_fitted(model)
        
    def test_score_train(self):
        arr = np.arange(self.X.shape[0])
        np.random.shuffle(arr)
        rnd_idx = arr[:]
        model = MTcgb_clf(n_common_estimators=3)
        model.fit(self.X[rnd_idx], self.y[rnd_idx])
        pred = model.predict(self.X)
        score_train = f1_score(self.y, pred)
        ic(score_train)
        self.assertGreaterEqual(score_train, 0.8)


if __name__ == '__main__':
    unittest.main()