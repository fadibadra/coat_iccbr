import numpy as np
from numpy.linalg import norm

__all__ = ['Equal', 'ES', 'Polynomial', 'WeightedSum', 'LowerOrEqual',
           'GreaterOrEqual', 'Minus', 'AbsDiff', 'Step']


class Variation:
    def __init__(self):
        pass

    def apply(self, x, y):
        pass


class Equal(Variation):
    def apply(self, x, y):
        return 1 if x == y else 0


class GreaterOrEqual(Variation):
    def apply(self, x, y):
        return 1 if x >= y else 0


class LowerOrEqual(Variation):
    def apply(self, x, y):
        return 1 if x <= y else 0


class Minus(Variation):
    def apply(self, x, y):
        return y - x


class AbsDiff(Variation):
    def apply(self, x, y):
        return abs(y - x)


class Step(Variation):
    def __init__(self, step):
        self.step = step

    def apply(self, x, y):
        return 1 if abs(y-x) <= self.step else 0


class Polynomial(Variation):

    def __init__(self, power, value_range):
        self.power = power
        self.value_range = value_range

    def apply(self, x, y):
        return pow(abs(self.value_range-abs(y-x)), self.power)/pow(self.value_range, self.power)


class WeightedSum(Variation):
    def __init__(self, weights, scales):
        self.w = weights
        self.scales = scales

    def apply(self, si, sj):
        r = 0.
        for k in range(len(self.scales)):
            r += self.w[k]*self.scales[k].apply(si[k], sj[k])
        return r/sum(self.w)


class ES(Variation):
    def apply(self, si, sj):
        r = 0.
        if type(si) is np.ndarray:
            for k in range(len(si)):
                r += AbsDiff().apply(si[k], sj[k])**2
        else:
            r += AbsDiff().apply(si, sj)**2
        return np.exp(-r)
        # return 1/(1+math.sqrt(r))
