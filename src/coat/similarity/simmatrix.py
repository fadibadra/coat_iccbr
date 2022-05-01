from itertools import permutations

import numpy as np
import pandas as pd

from coat.similarity.variation import Variation


def asndarray(df):
    if type(df) is pd.DataFrame or type(df) is pd.Series:
        return df.values
    else:
        return df


def simmatrix(df, scale):
    ndarray = asndarray(df)
    n = ndarray.shape[0]
    m = SimMatrix(ndarray, scale, shape=(n, n))
    return m


class SimMatrix(np.ndarray):
    # https://numpy.org/doc/stable/user/basics.subclassing.html
    def __new__(subtype, ndarray, scale, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, info=None):
        obj = super(SimMatrix, subtype).__new__(
            subtype, shape, dtype, buffer, offset, strides, order)
        obj.df = ndarray
        obj.scale = scale
        return obj

    def __init__(self, ndarray, scale, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, info=None):
        n = shape[0]
        for (i, j) in permutations(range(n), 2):
            self[i, j] = scale.apply(ndarray[i], ndarray[j])
        # for i in range(n):
        #     for j in range(i):
        #         self[i][j]=scale.apply(ndarray[i],ndarray[j])
        for i in range(n):
            self[i, i] = 1.0
        # for i in range(n):
        #     for j in range(i+1,n):
        #         self[i][j]=self[j][i]

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.df = getattr(obj, 'df', None)
        self.scale = getattr(obj, 'scale', None)

    def fill_row(self, i):
        """
        Fills ith row of the matrix.
        """
        for j in range(self.shape[0]):
            self[i, j] = self.scale.apply(self.df[i], self.df[j])

    def fill_column(self, j):
        for i in range(self.shape[0]):
            self[i, j] = self.scale.apply(self.df[i], self.df[j])

    def add(self, array):
        """
        Adds a line to the original array,
        and update sim matrix accordingly.
        """
        n = self.shape[0]
        a = np.array(array)
        if len(a.shape) == 0:
            a = np.array([array])
        if len(self.df.shape) > 1 and len(a.shape) == 1:
            a = a[np.newaxis, :]
        df = np.concatenate((self.df, a))
        new_m = SimMatrix(df, self.scale, shape=(n+1, n+1), buffer=np.concatenate(
            ((np.concatenate((self, np.empty((1, n))), axis=0)), np.empty((n+1, 1))), axis=1))
        new_m.fill_row(n)
        new_m.fill_column(n)
        return new_m

    def delete(self, k):
        """
        Returns similarity matrix for all but kth element of the original array.
        """
        m = np.delete(np.delete(self, k, 0), k, 1)
        m.df = np.delete(self.df, k, 0)
        return m
