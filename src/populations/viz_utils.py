import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from lightgbm import plot_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)


def plot_scatter_plotly(embed_df, ref_df, col):
    tmp_df = pd.DataFrame({"x": embed_df[:,0], "y": embed_df[:,1], "z": embed_df[:,2], col: ref_df[col], 'size': 10})
    fig = px.scatter_3d(tmp_df, x='x', y='y', z='z', color=col)
    fig.show()
    

def plot_scatter(embed_df, ref_df, col):
    tmp_df = pd.DataFrame({"x": embed_df[:,0], "y": embed_df[:,1], "z": embed_df[:,2], col: ref_df[col]})
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')
  
    ax.scatter(tmp_df['x'], tmp_df['y'], tmp_df['z'],
               c=pd.Categorical(ref_df[col]).codes, cmap="Set2_r", s=60)
  
    # make simple, bare axis lines through space:
    xAxisLine = ((min(tmp_df['x']), max(tmp_df['x'])), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(tmp_df['y']), max(tmp_df['y'])), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(tmp_df['z']), max(tmp_df['z'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA result")
    plt.show()

    
def vizualize_pca(ds, hue):
    X_scaled = StandardScaler().fit_transform(ds.features)
    X_scaled_pca = PCA(n_components=3).fit_transform(X_scaled)
    plot_scatter_plotly(X_scaled_pca, ds.df, hue)


def plot_roc_aucs(tps):
    fig = plt.figure(figsize=(12,12))
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
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plot_importance(model, ax=ax)
    plt.show()