from statistics import mean, stdev

import numpy as np
import pandas as pd


def collect_all_instances(d):
    """ Collects all instances available (test and training set) for a dataset. """
    return pd.concat([d.train, d.test]) if id(d.test) != id(d.train) else d.train

def shuffle(df):
    return df.sample(frac=1).reset_index(drop=True)

def nfolds(df, d, folds=10, size=None):
    """ Construct n folds with homogeneous class distribution of outcomes in each fold. """
    fold = [[] for k in range(folds)]
    if size is None:
        size = len(df)/folds
    outcomes = np.unique(df[d.tgt_att], return_counts=True)[0]
    split_by_outcome = []
    for k in range(len(outcomes)):
        split_by_outcome.append(df[df[d.tgt_att] == outcomes[k]])
    i = 0
    stop = False
    while not stop:
        hasMore = False
        for k in range(len(fold)):
            for j in range(len(split_by_outcome)):
                if i < len(split_by_outcome[j]):
                    if len(fold[k]) < size:
                        fold[k].append(split_by_outcome[j].iloc[i].name)
                        hasMore = True
            i = i+1
        stop = not hasMore
    return [df.loc[fold[k]] for k in range(len(fold))]


def sample(df, d, n): return nfolds(df, d, 1, size=n)[0]


def split(folds):
    for k in range(len(folds)):
        yield (pd.concat([folds[i] for i in range(len(folds)) if i != k]), folds[k])

def pearson(x, y):
    """ Computes Pearson's coefficient. """
    mx = mean(x)
    my = mean(y)
    sx = stdev(x)
    sy = stdev(y)
    # print(f'x {mx}+-{sx} \ny {my}+-{sy}')
    cov = 0.
    n = len(x)
    # print(f'n={n}')
    for i in range(n):
        cov += (x[i]-mx)*(y[i]-my)
    cov /= len(x)
    # print(f'cov {cov}, sx*sy {sx*sy}')
    return cov / (sx*sy)

class MultiClassConfusionMatrix(object):
    def __init__(self, classes):
        super(MultiClassConfusionMatrix, self).__init__()
        self.n = len(classes)
        self.classes = classes
        self.m = np.zeros((self.n, self.n))
        self.nb_predictions = 0

    def add_prediction(self, predicted_class, real_class):
        i = self.classes.index(predicted_class)
        j = self.classes.index(real_class)
        self.m[i][j] += 1
        self.nb_predictions += 1

    def accuracy(self):
        acc = 0.0
        for k in range(self.n):
            acc += self.m[k][k]
        return acc/self.nb_predictions
