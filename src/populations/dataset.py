from config import Config
import pandas as pd
import numpy as np


KEY_COLUMNS = ('population', 'region', 'nationality', 'id', 'source')


def filter_loci(df):
    return [x for x in df.columns if x not in KEY_COLUMNS]


def filter_keys(df):
    return [x for x in df.columns if x in KEY_COLUMNS]
    

class Dataset:
    def __init__(self, path=None, df=None):
        if path:
            self._path = path
            self._df = pd.read_csv(self._path)
        else:
            self._df = df
            
        self._init_dataset()
        
    def _init_dataset(self):
        self._keys = filter_keys(self._df)
        self._loci = filter_loci(self._df)
        
        self._patch_alleles()
        self._drop_nan_records()
      
    @property
    def loci(self):
        return self._loci
   
    @property
    def df(self):
        return self._df
    
    @property
    def features(self):
        return self._df[self._loci].values
     
    def _patch_alleles(self):
        def __parse_value(x):
            if x == '?':
                return 0
            x = str(x).replace('-', '.')
            if x[-1] == '.':
                x = x[:-1]
            return float(x)

        for col in self._loci:
            self._df[col] = self._df[col].apply(lambda x: __parse_value(x))
    
    def _drop_nan_records(self):
        self._df = self._df.fillna(0)
        corrupted_index = self._df.loc[((self._df[self._loci] == 0).sum(axis=1) > 0)].index
        self._df = self._df.drop(corrupted_index, axis=0).reset_index(drop=True)
        
    def describe(self):
        print(self._df.shape)
        display(self._df.head())

        
def get_default_cfg_path():
    return 'cfg/pop_datasets_meta.yml'


class DatasetsHandler:
    def __init__(self, cfg_path=None):
        self._cfg = Config(cfg_path or get_default_cfg_path())
        self._init_datasets()
        
    def _init_datasets(self):
        self._datasets = dict()
        for ds_cfg in self._cfg['datasets']:
            self._datasets[ds_cfg['name']] = Dataset(path=ds_cfg['path'])
            
    def describe(self):
        for k, v in self._datasets.items():
            print(k)
            v.describe()
    
    def available_datasets(self):
        for k, v in self._datasets.items():
            print(k)
            
    def _loci_intersection(self, dataset_ids):
        loci_sets = [set(v.loci)
                     for k, v in self._datasets.items()
                     if k in dataset_ids]
        result = loci_sets[0]
        for x in loci_sets[1:]:
            result = result.intersection(x)
        return sorted(list(result))

    def join_datasets(self, new_id, dataset_ids):
        result_df = pd.concat([v.df.assign(source=k)
                               for k, v in self._datasets.items()
                               if k in dataset_ids],
                              axis=0).reset_index(drop=True)
        loci_set = self._loci_intersection(dataset_ids)
        result_df = result_df[filter_keys(result_df) + loci_set]
       
        self._datasets[new_id] = Dataset(df=result_df)
    
    def get(self, k):
        return self._datasets[k]
                        
            
            
            
            
            
            