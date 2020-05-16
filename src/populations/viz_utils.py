import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from lightgbm import plot_importance
from matplotlib import pyplot as plt
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set()
init_notebook_mode(connected=True)


def plot_scatter_plotly(embed_df, ref_df, col):
    tmp_df = pd.DataFrame({"x": embed_df[:, 0],
                           "y": embed_df[:, 1],
                           "z": embed_df[:, 2],
                           col: ref_df[col],
                           'id': ref_df['id'],
                           'size': 5})
    fig = px.scatter_3d(tmp_df, x='x', y='y', z='z', color=col)
    fig.show()


def plot_heatmap(df, title):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    pivot_df = df.pivot("Group1", "Group2", "Test ROC AUC")

    sns.heatmap(pivot_df, ax=ax, annot=True, cmap="YlGnBu")
    ax.set_xlabel("Group1")
    ax.set_ylabel("Group2")
    ax.set_title(title)
    plt.show()


def plot_scatter(embed_df, ref_df, col):
    tmp_df = pd.DataFrame(
        {"x": embed_df[:, 0], "y": embed_df[:, 1], "z": embed_df[:, 2], col: ref_df[col]})
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tmp_df['x'], tmp_df['y'], tmp_df['z'],
               c=pd.Categorical(ref_df[col]).codes, cmap="Set2_r", s=10)

    # make simple, bare axis lines through space:
    xAxisLine = ((min(tmp_df['x']), max(tmp_df['x'])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(tmp_df['y']), max(tmp_df['y'])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(tmp_df['z']), max(tmp_df['z'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA result")
    plt.show()


def one_hot_encode(x):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(x)
    return enc.transform(x).toarray()


def vizualize_pca(ds, hue, one_hot=False, outliers_mult=3):
    X = ds.features
    X = one_hot_encode(X) if one_hot else X
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled_pca = PCA(n_components=3).fit_transform(X_scaled)
    # Remove outliers
    X_pca_mean = X_scaled_pca.mean(axis=0)
    X_pca_std = X_scaled_pca.std(axis=0)
    good_index = (
        (X_scaled_pca > X_pca_mean - outliers_mult * X_pca_std).all(axis=1)) & (
        (X_scaled_pca < X_pca_mean + outliers_mult * X_pca_std).all(axis=1)
    )
    plot_scatter_plotly(X_scaled_pca[good_index], ds.df[good_index], hue)


def plot_roc_aucs(tps):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for tl, pp, title in tps:
        fpr, tpr, _ = roc_curve(tl, pp)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                lw=2, label='ROC curve (area = %0.2f) %s' % (roc_auc, title))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def lgbm_fe(model):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plot_importance(model, ax=ax)
    plt.show()
