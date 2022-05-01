import os

import pandas as pd

from data import FOLDS_DIR, SAMPLES_DIR
from ml.evaluation import collect_all_instances, nfolds


def folds_path(dataset): return FOLDS_DIR+dataset.name


def create_folds(dataset):
    """ Creates folds for a given dataset, if not present. """
    d = dataset
    path = folds_path(d)
    if not os.path.exists(path+'_fold1'):
        print(f'No folds found. Creating them...')
        df = collect_all_instances(d).sample(frac=1).reset_index(drop=True)
        folds = nfolds(df, d)
        fold_nb = 1
        for fold in folds:
            fold.to_csv(path+'_fold'+str(fold_nb))
            fold_nb += 1
        print(f'done.')
    else:
        print(f'Folds found in {path}')
        folds = []
        for k in range(1, 11):
            folds.append(pd.read_csv(path+'_fold'+str(k), index_col=0))
    return folds


def samples_path(dataset, fold_nb, nb_samples, sample_size, dir): return dir + \
    dataset.name+'_fold'+str(fold_nb)+'_'+str(nb_samples)+'_'+str(sample_size)


def create_samples(train, nb_samples, sample_size, path):
    samples = []
    if not os.path.exists(path+'_sample1'):
        print(f'No samples found. Creating them...')
        for k in range(nb_samples):
            df = train.sample(frac=1).iloc[:sample_size]
            df.to_csv(path+'_sample'+str(k+1))
            samples.append(df)
    else:
        print(f'Samples found in {path}')
        for k in range(nb_samples):
            samples.append(pd.read_csv(path+'_sample'+str(k+1), index_col=0))
    return samples


def create_training_samples(dataset, train, fold_nb, nb_samples, sample_size):
    return create_samples(train, nb_samples, sample_size, samples_path(
        dataset, fold_nb, nb_samples, sample_size, SAMPLES_DIR))
