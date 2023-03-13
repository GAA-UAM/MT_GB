import unittest
from data.DataLoader import DataLoader
import numpy as np

from icecream import ic

class TestDataLoader(unittest.TestCase):

    def test_school_dataset(self):
        dataloader = DataLoader('_data')
        df_data, df_target = dataloader.load_dataset(dataset_name='school')
        ic(df_data)
        self.assertEqual(df_data.shape, (15362, 28))
        self.assertEqual(df_target.shape, (15362,1))
        uniq = np.unique(df_data[:, -1])
        ic(len(uniq))
        self.assertEqual(len(uniq), 139)


    def test_landmine_dataset(self):
        dataloader = DataLoader('_data')
        df_data, df_target = dataloader.load_dataset(dataset_name='landmine')
        ic(df_data)
        self.assertEqual(df_data.shape, (14820, 10))
        self.assertEqual(df_target.shape, (14820,1))
        uniq = np.unique(df_data[:, -1])
        ic(len(uniq))
        self.assertEqual(len(uniq), 29)


        
    # def test_invalid_dataset(self):
    #     with self.assertRaises(AttributeError):
    #         dataloader = DataLoader('my_data')
    #         df_data, df_target, _, _ = dataloader.load_dataset(dataset_name='invalid')


if __name__ == '__main__':
    unittest.main()