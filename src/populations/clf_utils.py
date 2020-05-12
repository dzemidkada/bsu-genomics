from lightgbm import LGBMClassifier, Dataset, train
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from populations.viz_utils import plot_roc_aucs, lgbm_fe
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import numpy as np
from collections import defaultdict
from catboost import CatBoostClassifier
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from IPython.core.display import HTML
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def display_correlation_with_target(x, y, features):
    display(HTML('<h3>Correlation with target</h3>'))
    data = pd.DataFrame(x, columns=features).assign(target=y)
    corrs = data.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
    corrs = corrs[corrs['level_0'] != corrs['level_1']]
    corrs = corrs[corrs['level_0'] == 'target']
    display(corrs.T)
    
    
def modeling_step(X, y, X_test, model, model_type, folds, n_fold=10):  
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    train_prediction = np.zeros(len(X))
    test_prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model_type == 'cat':
            cat_params = {'learning_rate': 0.02,
              'depth': 5,
              'l2_leaf_reg': 10,
              'bootstrap_type': 'Bernoulli',
              #'metric_period': 500,
              'od_type': 'Iter',
              'od_wait': 50,
              'random_seed': 11,
              'allow_writing_files': False}
            model = CatBoostClassifier(iterations=1000,  eval_metric='AUC', **cat_params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X)
            y_pred_valid = model.predict(X_valid)
            
            
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = roc_auc_score(y_valid, y_pred_valid)
            
            t = (lambda x: model.predict_proba(x)[:, 1]) \
                if model.__module__ != 'sklearn.svm._classes' else lambda x: model.decision_function(x)
            y_pred = t(X_test)
            y_pred_train = t(X)
            y_pred_valid = t(X_valid)
            
        if model_type == 'glm':
            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()
            model_results.predict(X_test)
            y_pred_valid = model_results.predict(X_valid).reshape(-1,)
            score = roc_auc_score(y_valid, y_pred_valid)
            
            y_pred = model_results.predict(X_test)
            y_pred_train = model_results.predict(X)
            y_pred_valid = model_results.predict(X_valid)
            
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        train_prediction += y_pred_train
        test_prediction += y_pred
        
    train_prediction /= n_fold
    test_prediction /= n_fold    
    return (train_prediction, test_prediction), scores


def model_selection(X_train, y_train, X_test, y_test, folds, title):
    results = defaultdict(list)
    best_preds, best_cv_mean = None, 0
    for model_type in ['sklearn', 'glm', 'cat']:
        if model_type == 'sklearn':
            for name, model in [
                    ('rf', RandomForestClassifier(n_estimators=100, max_leaf_nodes=5)),
                    ('lr c1e-2', linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=1e-2, solver='liblinear')),
                    ('svc', SVC())
            ]:
                results['Model'].append(f'{model_type} {name}')
                predictions, scores = modeling_step(
                    X_train.values, y_train.values, X_test.values, model, model_type=model_type, folds=folds)
                results['CV AUC mean'].append(np.mean(scores))
                results['CV AUC std'].append(np.std(scores))

                if np.mean(scores) > best_cv_mean:
                    best_cv_mean = np.mean(scores)
                    best_preds = predictions
        else:
            results['Model'].append(f'{model_type}')
            predictions, scores = modeling_step(
                X_train.values, y_train.values, X_test.values, model, model_type=model_type, folds=folds)
            results['CV AUC mean'].append(np.mean(scores))
            results['CV AUC std'].append(np.std(scores))

            if np.mean(scores) > best_cv_mean:
                best_cv_mean = np.mean(scores)
                best_preds = predictions

    display(HTML(f'<h3>CV results</h3>'))

    display(pd.DataFrame(results))
    
    plot_roc_aucs([
        [y_train, best_preds[0], f'{title} Train AUC'],
        [y_test, np.ones(y_test.shape[0]) * (np.mean(y_train) < 0.5), f'{title} Test AUC (majority)'],
        [y_test, best_preds[1], f'{title} Test AUC']
    ])

def one_vs_one_clfs(df, features, target, folds):
    groups = df[target].value_counts().index
    for i, group0 in enumerate(groups):
        for j, group1 in enumerate(groups):
            if group0 != group1 and i < j:
                title = f'{group0} vs {group1}'
                display(HTML(f'<h2>{title}</h2>'))

                cur_groups = [group0, group1]
                data = df.query(f'{target} in @cur_groups')
                display(data[target].value_counts())

                X_train, X_test, y_train, y_test = train_test_split(
                    data[features], data[target], stratify = data[target], shuffle=True, test_size=0.2, random_state=42)
                y_train = (y_train == group1).astype('int')
                y_test = (y_test == group1).astype('int')

                # Correlations
                display_correlation_with_target(X_train, y_train, features)
                
                model_selection(X_train, y_train, X_test, y_test, folds, title)


def one_vs_all_clfs(df, features, target, folds):
    groups = df[target].unique()

    for group in groups:
        title = f'{group} vs ALL'
        display(HTML(f'<h2>{title}</h2>'))
        data = df.copy()

        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data[target], stratify = data[target], shuffle=True, test_size=0.2, random_state=42)
        y_train = (y_train == group).astype('int')
        y_test = (y_test == group).astype('int')

        # Correlations
        display_correlation_with_target(X_train, y_train, features)

        model_selection(X_train, y_train, X_test, y_test, folds, title)
           
           