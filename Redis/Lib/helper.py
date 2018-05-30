from flask import  make_response
import numpy as np
import random
import io
import base64
import pickle
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.cm import coolwarm

from redis import Redis
redis_connection = Redis('this_redis')

def matrix_from_redis(redis_client, key, dtype=int):
    matrix_bytestring = redis_client.get(key)
    matrix_raveled = np.fromstring(matrix_bytestring, dtype=dtype)
    n = int(redis_client.get(key+'_n').decode())
    m = int(redis_client.get(key+'_m').decode())
    matrix = matrix_raveled.reshape(n, m)
    return matrix

def get_iris_data():

    iris_X         = redis_connection.get('iris_X')
    iris_y         = redis_connection.get('iris_y')
    sample_indices = redis_connection.get('sample_indices')

    while iris_X is None:
        load_iris_data()
    iris_X = pickle.loads(iris_X)
    iris_y = pickle.loads(iris_y)

    if sample_indices:
        sample_indices = pickle.loads(sample_indices)

    return iris_X, iris_y, sample_indices

def load_iris_data():
    X, y = load_iris(return_X_y=True)
    pca = PCA(2)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_pca_pkl = pickle.dumps(X_pca)
    y_pkl = pickle.dumps(y)

    redis_connection.set('iris_X', X_pca_pkl)
    redis_connection.set('iris_y', y_pkl)

def reset():
    redis_connection.delete('sample_indices')
    redis_connection.delete('logistic_regression')

def sample():
    p =0.33
    sample_indices = np.random.choice(a=[False, True], size=150, p=[p, 1-p])
    sample_indices_pkl = pickle.dumps(sample_indices)

    redis_connection.set('sample_indices', sample_indices_pkl)

def train(iris_X, iris_y, sample_indices):

    while sample_indices is None:
        sample()
    iris_X_not_sample = iris_X[~sample_indices]
    iris_y_not_sample = iris_y[~sample_indices]
    lr = LogisticRegression()
    lr.fit(iris_X_not_sample, iris_y_not_sample)
    lr_pkl = pickle.dumps(lr)
    redis_connection.set('logistic_regression', lr_pkl)

def fig_to_png_string(fig):
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)  # rewind to beginning of file

    return base64.b64encode(output.getvalue())

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot(iris_X, iris_y, sample_indices):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    if sample_indices is not None:
        iris_X_sample = iris_X[sample_indices]
        iris_y_sample = iris_y[sample_indices]
        iris_X_not_sample = iris_X[~sample_indices]
        iris_y_not_sample = iris_y[~sample_indices]
    lr = redis_connection.get('logistic_regression')

    if lr:
        lr = pickle.loads(lr)
        X0, X1 = iris_X[:, 0], iris_X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(axis, lr, xx, yy, cmap=coolwarm, alpha=0.8)

    if sample_indices is not None:
        axis.scatter(iris_X_sample[:, 0], iris_X_sample[:, 1], marker='.', s=100, label='sample', alpha=0.6)
        axis.scatter(iris_X_not_sample[:, 0], iris_X_not_sample[:, 1], marker='s', label='not sample', alpha=0.6)
    else:
        axis.scatter(iris_X[:, 0], iris_X[:, 1], c=iris_y, marker='.', s=100, alpha=0.6)
    axis.legend()

    figdata_png = fig_to_png_string(fig)

    return figdata_png.decode()
