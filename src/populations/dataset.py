import numpy as np
import pandas as pd

from config import Config

STR_KEY_COLUMNS = ('population', 'region', 'nationality', 'id', 'source')
SNP_KEY_COLUMNS = ('s_id', 'db_id', 'birth_region_id',
                   'ethnographic_group_id', 'appearance_type')


def filter_non_keys(df, keys):
    return [x for x in df.columns if x not in keys]


def filter_keys(df, keys):
    return [x for x in df.columns if x in keys]


class STRDataset:
    def __init__(self, path=None, df=None):
        if path:
            self._path = path
            self._df = pd.read_csv(self._path)
        else:
            self._df = df

        self._init_dataset()

    def _init_dataset(self):
        self._keys = filter_keys(self._df, STR_KEY_COLUMNS)
        self._loci = filter_non_keys(self._df, STR_KEY_COLUMNS)
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

    def patch_ids(self, prefix):
        if 'id' in self._df.columns:
            self._df.id = prefix + self._df.id.astype('str')

    def patch_meta_data(self, survey, cols):
        self._df.id = self._df.id.astype('str')
        self._df = pd.merge(
            survey[survey.valid][cols + ['region', 'lat', 'long']],
            self._df,
            on=cols,
            how='left'
        )
        print(
            f'Records patched: {(self._df.region_x != self._df.region_y).sum()}')
        self._df = (
            self._df[~self._df.region_y.isna()]
            .rename(columns={'region_x': 'region'})
            .drop('region_y', axis=1)
        )

    def _drop_nan_records(self):
        self._df = self._df.fillna(-999)
        corrupted_index = self._df.loc[(
            (self._df[self._loci] == -999).sum(axis=1) > 0)].index
        self._df = self._df.drop(
            corrupted_index, axis=0).reset_index(drop=True)

    def describe(self):
        print(self._df.shape)
        display(self._df.head())


def get_default_str_cfg_path():
    return 'cfg/pop_datasets_meta.yml'


class STRDatasetsHandler:
    def __init__(self, cfg_path=None):
        self._cfg = Config(cfg_path or get_default_str_cfg_path())
        self._init_datasets()

    def _init_datasets(self):
        self._datasets = dict()
        for ds_cfg in self._cfg['datasets']:
            self._datasets[ds_cfg['name']] = STRDataset(path=ds_cfg['path'])

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
        result_df = result_df[filter_keys(
            result_df, STR_KEY_COLUMNS) + loci_set]

        self._datasets[new_id] = STRDataset(df=result_df)

    def one_hot_encode(self, old_id, new_id):
        def __one_hot_encode_dataset(ds):
            from sklearn.preprocessing import OneHotEncoder

            df = ds.df
            f = ds.loci
            result = []
            for i in range(0, len(f), 2):
                base_f = f[i].split('-')[0]
                values = pd.concat([df[f[i]].rename({f[i]: base_f}),
                                    df[f[i + 1]].rename({f[i + 1]: base_f})],
                                   axis=0)
                oh = OneHotEncoder().fit(pd.DataFrame(values))
                res1 = oh.transform(df[[f[i]]]).todense()
                res2 = oh.transform(df[[f[i + 1]]]).todense()
                result.append(pd.DataFrame(res1 + res2,
                                           columns=list(map(lambda x: f'{base_f}, {x}',
                                                            oh.categories_[0].tolist()))))
            result = pd.concat(
                result + [df[[x for x in df.columns if x not in f]]], axis=1)
            return STRDataset(df=result)

        self._datasets[new_id] = __one_hot_encode_dataset(
            self._datasets[old_id])

    def get(self, k):
        return self._datasets[k]


class SNPBatch:
    FULL_VARIANT_COLUMNS = ['#CHROM', 'POS', 'ANNOTATION', 'REF', 'ALT']
    KEY_SURVEY_COLUMNS = ['ethnographic_group_id', 'appearance_type']

    def __init__(self, path=None):
        self._excel_file = pd.ExcelFile(path)
        # Variants
        assert 'variants.snps.ann' in self._excel_file.sheet_names[0]
        self._variants_df = self._excel_file.parse(
            self._excel_file.sheet_names[0])
        # Mapping
        self._mapping = self._excel_file.parse('mapping')
        # Survey
        self._survey = self._excel_file.parse('info2')

        self.__preprocess_data()

    @property
    def features(self):
        return self._full_variants

    @property
    def df(self):
        return self._batch

    def __variants_preprocess(self):
        # Filter out low quality variants
        self._variants_df = (
            self._variants_df
            .query('FILTER == "PASS"')
            .fillna('NA')
        )
        self._samples_columns = [x
                                 for x in self._variants_df.columns
                                 if x.lower().startswith('s')]
        # No duplicates
        assert self._variants_df.shape[0] == self._variants_df.drop_duplicates(
            self.FULL_VARIANT_COLUMNS).shape[0]
        # No NaNs
        assert self._variants_df[self.FULL_VARIANT_COLUMNS].isna(
        ).sum().sum() == 0

        self._full_variants = (
            self
            ._variants_df[self.FULL_VARIANT_COLUMNS]
            .agg(lambda x: '_'.join(map(str, x)), axis=1)
        )

        self._variants_df = (
            self._variants_df
            .set_index(self._full_variants)
            [self._samples_columns].T
            .reset_index().rename(columns={'index': 's_id'})
        )
        self._variants_df[self._full_variants] = self._variants_df[self._full_variants].astype(
            'category')

    def __mapping_preprocess(self):
        # Filter out columns
        cols_to_keep = ['Sample', self._mapping.columns[2], 'birth_region_id']
        self._mapping = (
            self._mapping[cols_to_keep]
            .rename(columns=dict(zip(cols_to_keep[:2],
                                     ['s_id', 'db_id'])))
        )

    def __survey_preprocess(self):
        self._survey = self._survey[~self._survey.sample_code.isna()].reset_index(
            drop=True)
        self._survey.sample_code = self._survey.sample_code.astype('int64')

    def __preprocess_data(self):
        self.__variants_preprocess()
        self.__mapping_preprocess()
        self.__survey_preprocess()

        self._batch = pd.merge(
            self._variants_df,
            self._mapping,
            on='s_id',
            how='left'
        ).merge(
            self._survey[['sample_code'] + self.KEY_SURVEY_COLUMNS],
            left_on='db_id', right_on='sample_code',
        ).drop('sample_code', axis=1)

    def __str__(self):
        return f'\n{self._batch.shape[0]} examples, {self._full_variants.size} variants\n'


class SNPDataset:
    def __init__(self, df, full_variants):
        self._df = df
        self._full_variants = full_variants

    @property
    def features(self):
        return self._full_variants

    @property
    def df(self):
        return self._df

    def __str__(self):
        return f'\n{self._df.shape[0]} examples, {len(self._full_variants)} variants\n'


def get_default_snp_cfg_path():
    return 'cfg/snp_batches_meta.yml'


class SNPDatasetsHandler:
    def __init__(self, cfg_path=None):
        self._cfg = Config(cfg_path or get_default_snp_cfg_path())
        self._init_datasets()

    def _init_datasets(self):
        self._datasets = dict()
        for b_cfg in self._cfg['batches']:
            self._datasets[b_cfg['name']] = SNPBatch(path=b_cfg['path'])

    def describe(self):
        for k, v in self._datasets.items():
            print(k, v)

    def available_batches(self):
        for k, v in self._datasets.items():
            print(k)

    def _variants_intersection(self, dataset_ids):
        variants_sets = [set(v.features)
                         for k, v in self._datasets.items()
                         if k in dataset_ids]
        result = variants_sets[0]
        for x in variants_sets[1:]:
            result = result.intersection(x)
        return sorted(list(result))

    def join_datasets(self, new_id, dataset_ids):
        result_df = pd.concat([v.df.assign(source=k)
                               for k, v in self._datasets.items()
                               if k in dataset_ids],
                              axis=0).reset_index(drop=True)
        variants_set = self._variants_intersection(dataset_ids)
        result_df = result_df[filter_keys(
            result_df, SNP_KEY_COLUMNS) + variants_set]

        self._datasets[new_id] = SNPDataset(result_df, variants_set)

    def get(self, k):
        return self._datasets[k]
