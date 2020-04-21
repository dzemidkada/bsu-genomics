from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from populations.viz_utils import plot_roc_aucs, lgbm_fe
import pandas as pd
import numpy as np


def one_vs_one_clfs(df, features, target, default_model_factory=None, explainability=False):
    groups = df[target].unique()

    for i, group0 in enumerate(groups):
        for j, group1 in enumerate(groups):
            if group0 != group1 and i < j:
                title = f'{group0} vs {group1}'
                print(title)
                cur_groups = [group0, group1]
                data = df.query(f'{target} in @cur_groups')
                display(data[target].value_counts())

                X_train, X_test, y_train, y_test = train_test_split(
                    data[features], data[target], stratify = data[target], shuffle=True, test_size=0.2)
                y_train = (y_train == group1).astype('int')
                y_test = (y_test == group1).astype('int')

                model = default_model_factory()
                model.fit(X_train, y_train)

                train_preds = model.predict_proba(X_train)[:, 1]
                test_preds = model.predict_proba(X_test)[:, 1]
                plot_roc_aucs([
                    [y_train, train_preds, f'{title} Train AUC'],
                    [y_test, np.ones(y_test.shape[0]) * (np.mean(y_train) < 0.5), f'{title} Test AUC (majority)'],
                    [y_test, test_preds, f'{title} Test AUC']
                ])
                
                lgbm_fe(model)
                

def one_vs_all_clfs(df, features, target, default_model_factory=None, explainability=False):
    groups = df[target].unique()

    for group in groups:
        title = f'{group} vs ALL'
        print(title)
        data = df.copy()

        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data[target], stratify = data[target], shuffle=True, test_size=0.2)
        y_train = (y_train == group).astype('int')
        y_test = (y_test == group).astype('int')

        model = default_model_factory()
        model.fit(X_train, y_train)

        train_preds = model.predict_proba(X_train)[:, 1]
        test_preds = model.predict_proba(X_test)[:, 1]
        plot_roc_aucs([
            [y_train, train_preds, f'{title} Train AUC'],
            [y_test, np.ones(y_test.shape[0]) * (np.mean(y_train) < 0.5), f'{title} Test AUC (majority)'],
            [y_test, test_preds, f'{title} Test AUC']
        ])

        lgbm_fe(model)
           
           