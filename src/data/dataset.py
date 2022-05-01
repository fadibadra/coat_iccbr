import csv
import math
import os
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from coat.similarity import Equal, Polynomial
from coat.similarity.variation import Variation

from . import DATASETS_DIR

__all__ = ['Apartments', 'BreastW', 'Voting', 'Spect', 'TicTacToe', 'Pima', 'Balance', 'Wine', 'Iris', 'Monks1', 'Monks2', 'Monks3',
           'User', 'Zoo', 'MaCourbe', 'Droite', 'Overlap']


class Apartments(object):
    """ Apartments small dataset. """

    def __init__(self) -> None:
        self.name = 'apartments'
        self.features = ['nb_rooms', 'area']
        self.tgt_att = 'price'
        self.outcomes = list(range(400, 1200, 1))
        self.feature_scales = [Polynomial(2, 6), Equal()]
        self.outcome_scale = Polynomial(2, 800)
        self.train = pd.read_csv(DATASETS_DIR+"apartments-num.csv",
                                 sep=';')
        self.test = self.train
        super().__init__()


class Car(object):
    """ Car Evaluation dataset. """

    def __init__(self) -> None:
        self.name = 'car'
        self.features = ['buying', 'maint', 'doors',
                         'persons', 'lug_boot', 'safety']
        self.tgt_att = 'class'
        self.outcomes = ['unacc', 'acc', 'good', 'vgood']
        self.feature_scales = [Polynomial(2, 3), Polynomial(2, 3), Polynomial(
            2, 3), Polynomial(2, 2), Polynomial(2, 2), Polynomial(2, 2)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(
            DATASETS_DIR+'car/car-numeric.data', sep=';', names=self.features+[self.tgt_att])
        self.test = self.train
        super().__init__()


class BreastW(object):
    """ W. B. Cancer dataset (without unknowns). """

    def __init__(self) -> None:
        self.name = 'breastw'
        self.features = ['id', 'clump', 'uniformity_cell_size', 'uniformity_cell_shape', 'marginal_adhesion',
                         'single_epithelial_cell_size', 'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']
        self.tgt_att = 'class'
        self.outcomes = [2, 4]
        self.feature_scales = [Polynomial(2, 10) for k in range(1, 10)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"breast-w/breast-cancer-wisconsin.data.no-unknown",
                                 sep=',', names=self.features+["class"])[1:]
        self.test = self.train
        super().__init__()


class Voting(object):
    """ Voting dataset. """

    def __init__(self) -> None:
        self.name = 'voting'
        self.features = ['f'+str(k) for k in range(1, 17)]
        self.tgt_att = 'class'
        self.outcomes = [0, 1]
        self.feature_scales = [Equal() for k in range(1, 17)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"voting/voting.train",
                                 sep=',', names=["class"]+self.features)
        self.test = self.train
        super().__init__()


class Spect(object):
    """ Spect dataset. """

    def __init__(self) -> None:
        self.name = 'spect'
        self.features = ['f'+str(k) for k in range(1, 23)]
        self.tgt_att = 'class'
        self.outcomes = [0, 1]
        self.feature_scales = [Equal() for k in range(1, 23)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"spect/SPECT.train",
                                 sep=',', names=["class"]+self.features)
        self.test = pd.read_csv(
            DATASETS_DIR+"spect/SPECT.test", sep=',', names=['class']+self.features)
        super().__init__()


class TicTacToe(object):
    """ Tic Tac Toe dataset. """

    def __init__(self) -> None:
        self.name = 'tictactoe'
        self.features = ["top-left-square", "top-middle-square", "top-right-square",
                         "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square"]
        self.tgt_att = 'class'
        self.outcomes = ["positive", "negative"]
        self.feature_scales = [Equal(), Equal(), Equal(
        ), Equal(), Equal(), Equal(), Equal(), Equal(), Equal()]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"tictactoe/tic-tac-toe.data",
                                 sep=',', names=self.features+["class"])
        self.test = self.train
        super().__init__()


class PedMan(Variation):
    def apply(self, x, y):
        return pow(abs(2.42-abs(y-x)), 1.17100003734231)/pow(2.42, 1.17100003734231) if y < x else 1


class Pima(object):
    """ Pima dataset. """

    def __init__(self) -> None:
        self.name = 'pima'
        self.features = ["NumTimesPrg", "PlGlcConc", "BloodP",
                         "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"]
        self.tgt_att = 'HasDiabetes'
        self.outcomes = [0, 1]
        self.feature_scales = [Polynomial(15, 125), Polynomial(20, 125), Polynomial(30, 67.1), PedMan(
        ), Polynomial(4, 199), Polynomial(30, 846), Polynomial(10, 17), Polynomial(5, 99)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"pima/pima-indians-diabetes.csv",
                                 sep=',', names=self.features+["HasDiabetes"])
        self.test = self.train
        super().__init__()


class Balance(object):
    """ Balance dataset. """

    def __init__(self) -> None:
        self.name = 'balance'
        self.features = ['left_weight', 'left_distance',
                         'right_weight', 'right_distance']
        self.tgt_att = 'class'
        self.outcomes = ['L', 'B', 'R']
        self.feature_scales = [Polynomial(2, 5), Polynomial(
            2, 5), Polynomial(2, 5), Polynomial(2, 5)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(
            DATASETS_DIR+"balance-scale/balance-scale.data", sep=',', header=None, names=['class']+self.features)
        self.test = self.train
        super().__init__()


class Wine(object):
    """ Wine dataset. """

    def __init__(self) -> None:
        self.name = 'wine'
        self.features = ['alcohol', 'malic_acid',
                         'ash', 'alcalinity', 'magnesium', 'phenols', 'flavonoids', 'non_flavonoid', 'proanthocyanins', 'color', 'hue', 'od', 'proline']
        self.tgt_att = 'class'
        self.outcomes = [1, 2, 3]
        self.feature_scales = [Polynomial(2, 14.83), Polynomial(2, 5.8), Polynomial(2, 3.23), Polynomial(2, 30.0), Polynomial(2, 162), Polynomial(
            2, 3.88), Polynomial(2, 5.08), Polynomial(2, 0.66), Polynomial(2, 3.58), Polynomial(2, 13.0), Polynomial(2, 1.71), Polynomial(2, 4.0), Polynomial(2, 1680)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(
            DATASETS_DIR+"wine/wine.data", sep=',', header=None, names=['class']+self.features)
        self.test = self.train
        super().__init__()


class Iris(object):
    """ Iris dataset. """

    def __init__(self):
        self.name = 'iris'
        self.features = ['sepal_length', 'sepal_width',
                         'petal_length', 'petal_width']
        self.tgt_att = 'class'
        self.outcomes = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
        self.feature_scales = [Polynomial(2, 3.6), Polynomial(
            2, 2.4), Polynomial(2, 5.9), Polynomial(2, 2.4)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"iris/iris.data", sep=',', header=None, names=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
        self.test = self.train


class Monks(object):
    def __init__(self, k) -> None:
        self.k = k
        self.name = 'monks'+str(self.k)
        self.features = ['a'+str(i) for i in range(1, 7)]
        self.tgt_att = 'class'
        self.outcomes = [0, 1]
        self.feature_scales = [Polynomial(2, 2), Polynomial(2, 2), Polynomial(
            2, 1), Polynomial(2, 2), Polynomial(2, 3), Polynomial(2, 1)]
        self.outcome_scale = Equal()
        self.train = pd.read_csv(DATASETS_DIR+"monks/monks"+str(self.k)+"/monks-"+str(
            self.k)+".train", sep=' ', names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id'])
        self.test = pd.read_csv(DATASETS_DIR+"monks/monks"+str(self.k)+"/monks-"+str(
            self.k)+".test", sep=' ', names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id'])
        super().__init__()


def Monks1(): return Monks(1)
def Monks2(): return Monks(2)
def Monks3(): return Monks(3)


class User(object):
    def __init__(self) -> None:
        self.name = 'user'
        self.features = ['STG', 'SCG', 'STR', 'LPR', 'PEG']
        self.tgt_att = 'UNS'
        self.outcomes = ['very_low', 'Low', 'Middle', 'High']
        self.feature_scales = [Polynomial(2, 1)]*len(self.features)
        self.outcome_scale = Equal()
        self.train = pd.read_csv(
            DATASETS_DIR+"user-modeling/user.train", sep=';', names=self.features+['UNS'])
        self.test = pd.read_csv(
            DATASETS_DIR+"user-modeling/user.test", sep=';', names=self.features+['UNS'])
        super().__init__()


class Zoo(object):
    """ Zoo dataset """

    def __init__(self):
        self.name = 'zoo'
        self.features = ["animal_name", "hair", "feathers", 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
                         'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tails', 'domestic', 'catsize']
        self.tgt_att = 'type'
        self.outcomes = list(range(1, 8))
        self.feature_scales = []
        i = 0
        while i < len(self.features):
            if i == 13:  # legs
                self.feature_scales.append(Polynomial(2, 8))
            else:
                self.features.append(Equal())
            i += 1
        self.outcome_scale = Equal()
        self.train = pd.read_csv(
            DATASETS_DIR+"zoo/zoo.data", sep=',', names=self.features+self.tgt_att)
        self.test = self.train


class SyntheticBinaryClass(object):
    """ Synthetic 2D dataset of size n where the instances are divided into two classes A and B. """

    def __init__(self, name, n):
        self.n = n  #  number of instances
        self.name = name + '_'+str(n)
        self.features = ['x', 'y']
        self.tgt_att = 'A'
        self.outcomes = [0, 1]
        self.feature_scales = [Polynomial(2, 3), Polynomial(2, 3)]
        self.outcome_scale = Equal()
        self.csv_file = DATASETS_DIR+'courbes/'+self.name+'.csv'
        self.train = SyntheticBinaryClass.as_df(*self.load_distribution())
        self.test = self.train

    @ staticmethod
    def as_df(*distribution):
        (xA, yA, xB, yB) = distribution
        return pd.DataFrame(data={'x': xA+xB, 'y': yA+yB, 'A': [1]*len(xA)+[0]*len(xB)})

    @ staticmethod
    def split_by_class(x_list, y_list, A_list):
        (xA, yA, xB, yB) = ([], [], [], [])
        for i in range(len(x_list)):
            if A_list[i] == 1:
                xA.append(x_list[i])
                yA.append(y_list[i])
            else:
                xB.append(x_list[i])
                yB.append(y_list[i])
        return (xA, yA, xB, yB)

    def save_to_csv(self, xA, yA, xB, yB):
        with open(self.csv_file, 'w') as g:
            out = csv.writer(g, delimiter=',')
            out.writerows(zip(*[xA+xB, yA+yB, ['A']*len(xA)+['B']*len(xB)]))

    def load_distribution(self):
        (xA, yA, xB, yB) = ([], [], [], [])
        filename = self.csv_file
        if not os.path.exists(filename):
            print(f'{filename} does not exist. Creating it... ', end='')
            self.generate_distribution()
            print(f'done.')
        with open(filename) as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            for row in r:
                (x, y, c) = (float(row[0]), float(row[1]), row[2])
                if c == 'A':
                    xA.append(x)
                    yA.append(y)
                else:
                    xB.append(x)
                    yB.append(y)
        return (xA, yA, xB, yB)

    def draw(self, xA, yA, xB, yB, marker_sizeA=None, marker_sizeB=None):
        if not marker_sizeA:
            marker_sizeA = [30]*len(xA)
        if not marker_sizeB:
            marker_sizeB = [30]*len(xB)
        plt.scatter(xA, yA, marker='.',
                    color='royalblue', s=marker_sizeA)
        plt.scatter(xB, yB, marker='.', color='coral', s=marker_sizeB)
        plt.xlim(*self.xlim)
        plt.ylim(*self.ylim)
        #plt.legend(fontsize='x-small', loc='lower left')
        plt.tight_layout()
        # plt.axis('equal')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.show()

    def show(self):
        """ Plots the dataset using Matplotlib. """
        self.draw(*self.load_distribution())


class Courbe(SyntheticBinaryClass):
    """ Synthetic 2D dataset of n points of which n/2 are on each side of a given curve. """

    def __init__(self, name, f, n, error_rate=0.):
        self.f = f                  # curve function
        self.error_rate = error_rate  # rate of misclassified points
        super(Courbe, self).__init__(name + '_'+str(n)+'_'+str(error_rate), n)
        self.xlim=(0, 2)
        self.ylim=(0, 3)

    def distance_courbe(self, x0, y0):
        """ La distance minimale entre la courbe et un point (x0,y0) """
        def dist(x): return (y0-self.f(x))**2+(x0-x)**2
        xmin = optimize.fminbound(dist, 0, 2)
        return math.sqrt((y0-self.f(xmin))**2+(x0-xmin)**2)

    def generate_distribution(self):
        (xA, yA, xB, yB) = ([], [], [], [])
        k = 0
        while k < self.n:
            x = 2*random()
            y = 3*random()
            isA = (y < self.f(x))
            d = self.distance_courbe(x, y)
            # if abs(f(x-y)<0.2): # close to the border -> error_rate% chances to be misclassified
            if d < 0.3:  # close to the border -> error_rate% chances to be misclassified
                if random() <= self.error_rate/100:
                    isA = not isA
            if isA:
                xA.append(x)
                yA.append(y)
            else:
                xB.append(x)
                yB.append(y)
            k += 1
        self.save_to_csv(xA, yA, xB, yB)

    def with_f(self):
        u = np.linspace(0, 2, 100)
        plt.plot(u, self.f(u), color='#55557fff')
        return self

    


class MaCourbe(Courbe):
    """ La jolie courbe. """

    def __init__(self, n, error_rate=0.):
        super(MaCourbe, self).__init__('ma_courbe', lambda x: (
            3-x*x)/(2-x), n, error_rate=error_rate)


class Droite(Courbe):
    """ La droite y=-x+2.5 """

    def __init__(self, n, error_rate=0.):
        super(Droite, self).__init__(
            'droite', lambda x: -x+2.5, n, error_rate=error_rate)


class Overlap(SyntheticBinaryClass):
    """ Two overlapping classes in 2D. """

    def __init__(self, n, scale):
        self.scale = scale
        super(Overlap, self).__init__('overlap_scale_'+str(scale), n)
        self.xlim=(-50, 50)
        self.ylim=(-50, 50)

    def generate_distribution(self):
        centerA = (-5.,0.)
        centerB = (5.,0.)
        cov = [[1.,0.],[0.,1.]]
        (xA, yA) = np.random.multivariate_normal(centerA, np.multiply(cov,self.scale), round(self.n/2)).T
        (xB, yB) = np.random.multivariate_normal(centerB, np.multiply(cov,self.scale), round(self.n/2)).T
        self.save_to_csv(list(xA), list(yA), list(xB), list(yB))
