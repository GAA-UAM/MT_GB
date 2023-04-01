import numpy as np
import pandas as pd
from scipy.io import loadmat

from icecream import ic

def _load_dataset_school(data_dir):
        """Load dataframe from csv for school."""

        path = '{}/school/'.format(data_dir)        
        df_data = pd.read_csv('{}/school_data.csv'.format(path), index_col=0)
        df_target = pd.read_csv('{}/school_target.csv'.format(path), index_col=0)

        return df_data, df_target

def _load_dataset_landmine(data_dir):
    """Load dataframe from csv for landmine."""

    path = '{}/landmine/'.format(data_dir)

    df_data = pd.read_csv('{}/landmine_data.csv'.format(path), index_col=0)
    df_target = pd.read_csv('{}/landmine_target.csv'.format(path), index_col=0)

    return df_data, df_target

class DataLoader:
    """
    A class for loading popular datasets from scikit-learn library.
    
    Parameters
    ----------
    dataset_name : str, default: 'iris'
        The name of the dataset to load. Can be 'iris', 'digits'
    
    Attributes
    ----------
    data : array-like
        The data of the dataset
    target : array-like
        The target values of the dataset
    feature_names : array-like
        The feature names of the dataset
    target_names : array-like
        The target names of the dataset
    DESCR : str
        The description of the dataset
        
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    
    
    def load_dataset_school(self):
        ic(self.data_dir)
        df_data, df_target = _load_dataset_school(data_dir=self.data_dir)
        ic(df_data)
        return df_data.values, df_target.values
    
    def load_dataset_landmine(self):
        ic(self.data_dir)
        df_data, df_target = _load_dataset_landmine(data_dir=self.data_dir)
        ic(df_target)
        ic(np.unique(df_target, return_counts=True))
        # ic(df_data)
        return df_data.values, df_target.values
        
        
    def load_dataset(self, dataset_name, **kwargs):
        try: 
            load_function = getattr(self, 'load_dataset_{}'.format(dataset_name))
            df_data, df_target = load_function(**kwargs)
        except AttributeError as e:
            print(e)
            # print(f"Invalid dataset_name: {dataset_name}.")
            raise(AttributeError(f"Invalid dataset_name: {dataset_name}."))
        return df_data, df_target
