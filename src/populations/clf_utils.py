from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from IPython.core.display import HTML
from lightgbm import Dataset, LGBMClassifier, train
from scipy import stats
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from populations.viz_utils import lgbm_fe, plot_roc_aucs
from sklearn.metrics import classification_report, confusion_matrix


def display_correlation_with_target(x, y, features):
    display(HTML('<h3>Correlation with target</h3>'))
    data = pd.DataFrame(x, columns=features).assign(target=y)
    corrs = data.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
    corrs = corrs[corrs['level_0'] != corrs['level_1']]
    corrs = corrs[corrs['level_0'] == 'target']
    display(corrs.T)



PARAMS = {
    'cat_def': {
        'iterations': 500,
        'eval_metric': 'AUC',
        'random_seed': 11,
        'allow_writing_files': False,
        'loss_function': 'Logloss'
    },
    'cat_cv': {
        'depth': [4],
        'learning_rate': [1e-2],
        'l2_leaf_reg': [10]
    },
    'rf_cv': {
        'n_estimators': [10, 30, 50],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [2, 3],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    },
    'lr_cv': {
        'penalty': ['l1', 'l2'],
        'C': [1e-5, 1e-3, 1e-1, 10, 100],
        'class_weight': ['balanced'],
        'solver': ['lbfgs', 'liblinear'],
     },
    'svm_cv': {
        'C': [1e-5, 1e-3, 1e-1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3],
     }
}


def cvgrid_search(X, y, X_test, model_type):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, shuffle=True, test_size=0.2, random_state=42)

    if model_type == 'cat':
        model = CatBoostClassifier(**PARAMS['cat_def'])
        grid_search_result = model.grid_search(PARAMS['cat_cv'], X=X_train, y=y_train, plot=False, cv=5, stratified=True, verbose=0)
        best_model = CatBoostClassifier(**PARAMS['cat_def'], **grid_search_result['params'])

        best_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        get_preds = lambda model, x: model.predict(x, prediction_type='Probability')[:, 1]

    if model_type == 'rf':
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf,
                                       param_distributions=PARAMS['rf_cv'], n_iter=50, cv=5,
                                       verbose=0, random_state=42, n_jobs = -1)
        # Tune hyperparams
        rf_random.fit(X_train, y_train)
        best_model = rf_random.best_estimator_
        # Train set only
        best_model.fit(X_train, y_train)
        get_preds = lambda model, x: model.predict_proba(x)[:, 1]


    if model_type == 'lr':
        lr = linear_model.LogisticRegression()
        lr_random = RandomizedSearchCV(estimator=lr,
                                       param_distributions=PARAMS['lr_cv'], n_iter=50, cv=5,
                                       verbose=0, random_state=42, n_jobs = -1)
        # Tune hyperparams
        lr_random.fit(X_train, y_train)
        best_model = lr_random.best_estimator_
        # Train set only
        best_model.fit(X_train, y_train)
        get_preds = lambda model, x: model.predict_proba(x)[:, 1]

    if model_type == 'svm':
        svc = SVC()
        svc_random = RandomizedSearchCV(estimator=svc,
                                       param_distributions=PARAMS['svm_cv'], n_iter=50, cv=5,
                                       verbose=0, random_state=42, n_jobs = -1)
        # Tune hyperparams
        svc_random.fit(X_train, y_train)
        best_model = svc_random.best_estimator_
        # Train set only
        best_model.fit(X_train, y_train)
        get_preds = lambda model, x: model.decision_function(x)

    y_pred_train = get_preds(best_model, X_train)
    y_pred_val = get_preds(best_model, X_val)
    train_auc = roc_auc_score(y_train, y_pred_train)
    val_auc = roc_auc_score(y_val, y_pred_val)
    train_ap = average_precision_score(y_train, y_pred_train)
    val_ap = average_precision_score(y_val, y_pred_val)

    # Final fit
    best_model.fit(X, y)

    return (get_preds(best_model, X), get_preds(best_model, X_test)), (train_auc, val_auc), (train_ap, val_ap)


def modeling_step(X, y, X_test, model_type):
    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return cvgrid_search(X, y, X_test, model_type)

def plot_pr_curves(y_train, y_test, preds, title):
    train_p, train_r, _ = precision_recall_curve(y_train, preds[0])
    test_p, test_r, _ = precision_recall_curve(y_test, preds[1])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(train_r, train_p, linestyle='--', label=f'{title} Train PR')
    ax.plot(test_r, test_p, linestyle='--', label=f'{title} Test PR')
    no_skill = len(y_test[y_test==1]) / len(y_test)

    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'{title} Dummy clf PR')

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    plt.show()

    return auc(test_r, test_p)

def model_selection(X_train, y_train, X_test, y_test, title, c1='0', c2='1'):
    results = defaultdict(list)
    best_preds, best_val_mean = None, 0
    for model_type in ['cat', 'rf', 'lr', 'svm']:
        results['Model'].append(f'{model_type}')
        predictions, trainval_auc_scores, trainval_pr_scores = modeling_step(
            X_train.values, y_train.values, X_test.values, model_type=model_type)
        results['Train AUC'].append(trainval_auc_scores[0])
        results['Val AUC'].append(trainval_auc_scores[1])
        results['Train AP'].append(trainval_pr_scores[0])
        results['Val AP'].append(trainval_pr_scores[1])
        if trainval_pr_scores[1] > best_val_mean:
            best_val_mean = trainval_pr_scores[1]
            best_preds = predictions

    display(HTML(f'<h3>CV results</h3>'))

    display(pd.DataFrame(results))

    plot_roc_aucs([
        [y_train, best_preds[0], f'{title} Train AUC'],
        [y_test, np.ones(y_test.shape[0]) * (np.mean(y_train)
                                             < 0.5), f'{title} Dummy clf AUC'],
        [y_test, best_preds[1], f'{title} Test AUC']
    ])

    test_pr_auc = plot_pr_curves(y_train, y_test, best_preds, title)

    return test_pr_auc


def display_ks_test(x, y, features):
    display(HTML('<h3>KS test</h3>'))
    results = {'Feature': [], 'KS value': [], 'KS p-value': []}
    for c in features:
        v, p = stats.ks_2samp(x[c].values[y == 0], x[c].values[y == 1])
        results['Feature'].append(c)
        results['KS value'].append(v)
        results['KS p-value'].append(p)

    display(pd.DataFrame(results).sort_values(['KS p-value']).T)


def one_vs_one_clfs(ds, target):
    df = ds.df
    features = ds.loci
    groups = df[target].value_counts().index
    results = []
    for i, group0 in enumerate(groups):
        for j, group1 in enumerate(groups):
            if group0 != group1 and i < j:
                title = f'{group0} vs {group1}'
                display(HTML(f'<h2>{title}</h2>'))

                cur_groups = [group0, group1]
                data = df.query(f'{target} in @cur_groups')
                display(data[target].value_counts())

                X_train, X_test, y_train, y_test = train_test_split(
                    data[features], data[target], stratify=data[target], shuffle=True, test_size=0.2, random_state=42)
                y_train = (y_train == group1).astype('int')
                y_test = (y_test == group1).astype('int')

                # Correlations
                display_correlation_with_target(X_train, y_train, features)

                display_ks_test(X_train, y_train, features)

                roc_auc_test = model_selection(
                    X_train, y_train, X_test, y_test, title, c1=group0, c2=group1)
                results.append([group0, group1, roc_auc_test])
    return pd.DataFrame(results, columns=['Group1', 'Group2', 'Test ROC AUC'])


def one_vs_all_clfs(ds, target):
    df = ds.df
    features = ds.loci
    groups = df[target].unique()

    for group in groups:
        title = f'{group} vs ALL'
        display(HTML(f'<h2>{title}</h2>'))
        data = df.copy()

        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data[target], stratify=data[target], shuffle=True, test_size=0.2, random_state=42)
        y_train = (y_train == group).astype('int')
        y_test = (y_test == group).astype('int')

        # Correlations
        display_correlation_with_target(X_train, y_train, features)

        model_selection(X_train, y_train, X_test, y_test, title)
