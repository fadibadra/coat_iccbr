import numpy as np


class ReCoAT(object):
    """(Re)fined measure of the (Co)mplexity of a dataset for (A)nalogical (T)ransfer."""

    def __init__(self, s, o):
        self.s = s
        self.o = o
        self.n = s.shape[0]

    def rank(self, u):
        inv = np.unique(u, return_inverse=True)[1]
        ind, pos = np.unique(np.sort(inv), return_index=True)
        return np.array([self.n-pos[list(ind).index(e)]-1 for e in inv])

    def get_outcome(self, i): return self.o.df[i, ] if type(
        self.o.df[i]) is np.ndarray else self.o.df[i]

    def set_outcome(self, i, o):
        if type(self.o.df[i]) is np.ndarray:
            self.o.df[i, ] = o
        else:
            self.o.df[i] = o

    def gamma(self, i, return_triples=False):
        """ Computes complexity for triples (i,.,.) starting with instance i. """
        si = self.s[i]
        oi = self.o[i]

        # ranks of o and s (with possible ties)
        ranks_o = self.rank(oi)
        inv_ranks_o = [self.n-1-e for e in ranks_o]
        ranks_s = self.rank(si)

        # ranks of s,o (= according to s in decreasing order and then o in increasing order)
        ranks_s_o = np.arange(self.n)[np.argsort(
            list(np.lexsort((inv_ranks_o, ranks_s))))]

        # create n x n matrix
        m = np.zeros((self.n, self.n))

        # now read the matrix and count the inversions
        if return_triples:
            triples = []
            for k in np.argsort(np.subtract(ranks_s_o, ranks_o)):
                a = ranks_s_o[k]
                b = ranks_o[k]
                m[a][b] = 1
                c = np.sum(m[:a+1, b+1:])
                if c > 0:
                    for i_rank in list(np.nonzero(m[:a+1, b+1:])[0]):
                        triples.append((i, np.where(
                            ranks_s_o == i_rank)[0][0], k))
            return triples
        else:
            g = 0.
            for k in np.argsort(np.subtract(ranks_s_o, ranks_o)):
                a = ranks_s_o[k]
                b = ranks_o[k]
                m[a][b] = 1
                g += np.sum(m[:a+1, b+1:])
                # m[a][b]=si[k]
                # g+=np.sum(m[:a+1,b+1:])-np.count_nonzero(m[:a+1,b+1:])*m[a][b]

            return g

    def energy(self):
        """ Computes the dataset complexity. """
        return sum(self.gamma(i) for i in range(self.n))

    def similarity_inversions(self):
        """ Returns the triples for which there is a similarity inversion. """
        return [e for i in range(self.n) for e in self.gamma(i, return_triples=True)]

    def energy_increase(self, k):
        """ Computes the number of inversions in which the kth instance is involved. """
        r = self.gamma(k)
        sk = self.s[k]
        ok = self.o[k]
        mo = self.o.delete(k) - np.transpose(np.delete(ok, k)
                                             [np.newaxis, :])  # matrix of oi[j] - oi[k]
        ms = self.s.delete(k) - np.transpose(np.delete(sk, k)
                                             [np.newaxis, :])  # matrix of si[j] - si[k]
        r += np.count_nonzero(np.multiply(ms, mo) <= 0) - \
            np.count_nonzero(mo == 0)
        return r

    def induced_inversions(self, k):
        """ Returns the inversions in which the kth instance is involved. """
        n = self.n
        triples = []
        for i in range(n):
            if i == k:
                triples.extend(self.gamma(i, return_triples=True))
            else:
                si = self.s[i]
                oi = self.o[i]
                for j in range(n):
                    if oi[j] < oi[k]:
                        if si[j] >= si[k]:
                            triples.append((i, j, k))
                    elif oi[j] > oi[k]:
                        if si[j] <= si[k]:
                            triples.append((i, k, j))
        return triples

    def outcome_energies(self, i, potential_outcomes):
        """ Computes the energies for each potential outcome for instance i. """
        oi = self.get_outcome(i)
        outcomes = [o for o in potential_outcomes if o != oi]+[oi]
        inversions = []
        for o in outcomes:
            self.set_outcome(i, o)
            self.o.fill_column(i)
            self.o.fill_row(i)
            inversions.append(self.energy_increase(i))
        return inversions

    def set_loss_functional(self, loss_functional):
        self.loss_functional = loss_functional
        return self

    def loss(self): return np.sum(
        [self.loss_functional(i, self) for i in range(self.n)])/self.n

    def energy_loss(self, i):
        """ Computes the energy loss for instance i. """
        return self.energy_increase(i)

    def perceptron_loss(self, i, potential_outcomes):
        """ Computes the generalized perceptron loss for instance i. """
        energies = self.outcome_energies(i, potential_outcomes)
        return energies[-1]-min(energies)

    def hinge_loss(self, i, potential_outcomes, beta):
        """ Computes the hinge loss for instance i. """
        energies = self.outcome_energies(i, potential_outcomes)
        return max(0, beta+energies[-1]-min(energies[:-1]))

    def log_loss(self, i, potential_outcomes):
        """ Computes the log loss for instance i. """
        energies = self.outcome_energies(i, potential_outcomes)
        return np.log(1+np.exp((energies[-1]-min(energies[:-1]))/(self.n*(self.n-1))*2))

    def square_loss(self, i, potential_outcomes, beta):
        """ Computes the square square loss for instance i. """
        energies = self.outcome_energies(i, potential_outcomes)
        # return (inversions[-1])**2 + (max(0, beta+inversions[-1]-min(inversions[:-1])))**2
        # return inversions[-1] * max(0, beta+inversions[-1]-min(inversions[:-1]))
        return (1+(energies[-1]-min(energies))**2) * max(0, beta+energies[-1]-min(energies[:-1]))


def energy(s, o):
    """ Computes the energy of the dataset. """
    return ReCoAT(s, o).energy()


def energy_functional(s, o, k):
    """ Computes the energy increase induced by the presence of the kth instance. """
    return ReCoAT(s, o).energy_increase(k)


def loss(s, o, loss_functional): return ReCoAT(
    s, o).set_loss_functional(loss_functional).loss()


def energy_loss(s, o):
    """ Computes the energy loss. """
    return loss(s, o, lambda i, self: self.energy_loss(i))


def perceptron_loss(s, o, potential_outcomes):
    return loss(s, o, lambda i, self: self.perceptron_loss(i, potential_outcomes))


def perceptron_functional(s, o, i, potential_outcomes):
    return ReCoAT(s, o).perceptron_loss(i, potential_outcomes)


def hinge_loss(s, o, potential_outcomes, beta):
    return loss(s, o, lambda i, self: self.hinge_loss(i, potential_outcomes, beta))


def hinge_functional(s, o, i, potential_outcomes, beta):
    return ReCoAT(s, o).hinge_loss(i, potential_outcomes, beta)


def log_loss(s, o, potential_outcomes):
    return loss(s, o, lambda i, self: self.log_loss(i, potential_outcomes))


def square_loss(s, o, potential_outcomes, beta):
    return loss(s, o, lambda i, self: self.square_loss(i, potential_outcomes, beta))


def predict(s, o, t, potential_outcomes, return_complexities=False):
    """ Predicts the outcome of a new situation t. """
    min_outcome = potential_outcomes[0]
    coat = ReCoAT(s.add(t), o.add([min_outcome]))
    i = coat.n-1
    min_complexity = coat.energy_increase(i)
    if return_complexities:
        complexities = [min_complexity]
    for o in potential_outcomes[1:]:
        if type(coat.o.df[i]) is np.ndarray:
            coat.o.df[i, ] = o
        else:
            coat.o.df[i] = o
        coat.o.fill_column(i)
        coat.o.fill_row(i)
        c = coat.energy_increase(i)
        if return_complexities:
            complexities.append(c)
        if c < min_complexity:
            min_complexity = c
            min_outcome = o
    if return_complexities:
        return (complexities, min_outcome)
    else:
        return min_outcome


def energy_increase(s, o, t, rt, return_inversions=False):
    """ Computes the energy increase resulting from the addition of the case (t,rt) to the case base. """
    coat = ReCoAT(s.add(t), o.add(rt))
    if return_inversions:
        return coat.induced_inversions(coat.n-1)
    else:
        return coat.energy_increase(coat.n-1)


def similarity_inversions(s, o):
    """ Returns the triples for which there is a similarity inversion. """
    return ReCoAT(s, o).similarity_inversions()


def induced_inversions(s, o, k):
    """ Computes the inversions in which the kth instance is involved. """
    return ReCoAT(s, o).induced_inversions(k)
