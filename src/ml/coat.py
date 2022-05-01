import time

import numpy as np

import coat as ct
from coat.similarity import ES
from data.transform import identity, random_transformation, transform
from ml.evaluation import collect_all_instances, split, nfolds, MultiClassConfusionMatrix

def upper_bound(n):
    """ The upper bound Emax for E on a set of n instances drawn from the dataset d. """
    return n*n*(n-1)/2

def energy(d, L=None):
    """ Computes the dataset complexity (energy) using the Euclidian distance. """
    if L is not None:
        transform(d,L)
        Lx = d.transformed_train
    else:
        Lx = d.train[d.features].values
    s = ct.simmatrix(Lx, ES())
    o = ct.simmatrix(d.train[d.tgt_att].values, d.outcome_scale)
    # energy computation
    start = time.time()
    energy = ct.energy(s, o)
    end = time.time()
    prediction_time = (end-start)
    return (energy,prediction_time)


def average_energy(d,N,nb_iter=10):
    """ Computes the dataset complexity (energy) of N random instances drawn from the training set of the dataset """
    energies = []
    i = 0
    total_time = 0.
    while i<nb_iter:
        train = d.train.sample(n=N)
        train_data = train[d.features].values
        train_targets = train[d.tgt_att].values
        s = ct.simmatrix(train_data, ES())
        o = ct.simmatrix(train_targets, d.outcome_scale)
        start = time.time()
        energy = ct.energy(s, o)
        end = time.time()
        total_time += (end-start)
        print(int(energy))
        energies.append(energy)
        i += 1
    av = sum(energies)/len(energies)
    std = np.std(energies)
    av_time = total_time / nb_iter
    return (av,std,av_time)

def predict(d,L=None):
    """ Predict using CoAT by 10-fold cross validation. """
    df = collect_all_instances(d).sample(frac=1).reset_index(drop=True)
    folds = nfolds(df, d)
    if L is None:
        L = identity(d)
    fold_nb = 1
    accuracies = []
    total_start = time.time()
    average_prediction_time = 0.
    for train, test in split(folds):
        print(f'fold {fold_nb} ({len(train)}/{len(test)})')
        Lx = np.matmul(train[d.features].values,L)
        train_targets = train[d.tgt_att].values
        test_data = np.matmul(test[d.features].values,L)
        test_targets = test[d.tgt_att].values
        s = ct.simmatrix(Lx, ES())
        o = ct.simmatrix(train_targets, d.outcome_scale)
        m = MultiClassConfusionMatrix(d.outcomes)
        prediction_time = 0.0
        n = len(test_data)
        for k in range(n):
            s_test = test_data[k]
            real_o = test_targets[k]
            start = time.time()
            v = ct.predict(s, o, s_test, d.outcomes)
            end = time.time()
            prediction_time += (end-start)
            m.add_prediction(v, real_o)
        print(m.m)
        print(f'average prediction time: {prediction_time/n:.5f}s')
        average_prediction_time += prediction_time/n
        accuracy = m.accuracy()
        print('accuracy {}'.format(accuracy))
        accuracies.append(accuracy)
        fold_nb += 1
    average_prediction_time /= (fold_nb - 1)
    average_accuracy = sum(accuracies)/len(accuracies)
    std = np.std(np.array(accuracies))
    return (average_accuracy,std,average_prediction_time)


    