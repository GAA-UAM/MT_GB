
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

    def _load_dataset_school():
        """Load dataframe from csv for mnist variations."""

        data_dir = '../data/school/'
        
        data = loadmat('{}/school_b.mat'.format(data_dir))
        print(data.keys())

        y = data['y']
        x = data['x'].T[:, :-1] # remove the bias column
        task_indexes = data['task_indexes']

        print(x.shape)
        print(y.shape)

        t_col = np.zeros(y.shape)

        for i, (t1, t2) in enumerate(zip(task_indexes[:-1], task_indexes[1:])):
            task1, task2 = t1[0]-1, t2[0]-1
            t_col[task1:task2] = i

        t_col[task2:] = i+1  
        print(t_col)

        # for n in range(1, 10):
        #      data = loadmat('{}/school_{}_indexes.mat'.format(data_dir, n))
        #      print(data.keys())

        dic_feat = {}
        dic_feat['year'] = 3
        # dic_feat['exam_score'] = 2
        dic_feat['per_fsm'] = 1
        dic_feat['per_vr1band'] = 1
        dic_feat['gender'] = 2
        dic_feat['vrband'] = 3
        dic_feat['ethnic_group'] = 11
        dic_feat['school_gender'] = 3
        dic_feat['school_denomination'] = 3

        columns = []
        for colname, n in dic_feat.items():
            for i in range(n):
                columns.append('{}-{}'.format(colname, i))
        print(columns)
        print(len(columns))
        # columns = [''] * x.shape[1]
        # columns[0] = 'year'
        # columns[1], columns[2], columns[3] = 'school1', 'school2', 'school3'
        # columns[4], columns[5] = 'exam_score1', 'exam_score2'
        # columns[6], columns[7] = 'per_fsm1', 'per_fsm2'
        # columns[8], columns[9] = 'per_vrband1', 'per_vrband2'
        # columns[10] = 'gender'
        # columns[11] = 'vrband'
        # columns[12], columns[13] = 'ethnic_group1', 'ethnic_group2'

        df_data = pd.DataFrame(x, columns=columns)

        print(df_data)
        df_data['task'] = t_col
        df_target = pd.Series(y.flatten())

        df_data.to_csv('school_data.csv')
        df_target.to_csv('school_target.csv')

        return df_data, df_target
    
    def load_dataset_school():
        pass
        
        
    def load_dataset(self, dataset_name, **kwargs):
        try: 
            load_function = getattr(self, 'load_dataset_{}'.format(dataset_name))
            df, target, outer_cv, inner_cv, task_info = load_function(data_dir=self.data_dir, **kwargs)
        except AttributeError as e:
            print(e)
            # print(f"Invalid dataset_name: {dataset_name}.")
            raise(AttributeError(f"Invalid dataset_name: {dataset_name}."))
        return df, target, outer_cv, inner_cv, task_info
