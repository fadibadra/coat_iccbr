from itertools import product

import numpy as np
from sklearn.decomposition import PCA

from . import TRANSFORMATIONS_DIR


def to_2D(d):
    """ Convert data to 2D (by PCA). """
    x = d.train[d.features]
    y = d.train[d.tgt_att]

    if len(x.columns) > 2:
        x = PCA(n_components=2).fit(x).transform(x)
    else:
        x = x.values
    return (x, y)


def save(L, name):
    np.save(TRANSFORMATIONS_DIR+name, L)


def load(name):
    return np.load(TRANSFORMATIONS_DIR+name)


def transform(d,L=None):
    """ Adds a transformation to the dataset.
    If a learned transformation is found in a file, 
    adds this transformation L as an attribute of the dataset object. 
    Otherwise adds the identity matrix.
    """
    if L is None:
        try:
            L = load(d.name)
        except IOError:
            L = np.identity(len(d.features))
    d.L = L
    # print(L)
    train_points = d.train[d.features].values
    d.transformed_train = np.matmul(train_points, L)
    if hasattr(d, 'test') and (id(d.test) != id(d.train)):
        d.transformed_test = np.matmul(d.test[d.features].values, L)
    d.transform_point = lambda p: np.matmul(np.array([p]), L)
    return d

def identity(d):
    return np.identity(len(d.features))

def random_transformation(d):
    L = np.zeros(shape=(len(d.features), len(d.features)))
    for (i, j) in product(range(L.shape[0]), range(L.shape[1])):
        L[i, j] = np.random.uniform(low=-1., high=1.)
    return L

