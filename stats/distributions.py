import sys
import numpy as np
import math
from abc import ABC, abstractmethod
from stats.combinatorics import ncr
from numerical.special_functions import erf, ierf
from scipy.special import gammaincc, gammainccinv


class Distribution(ABC):

    def __init__(self, support):
        self._support = support

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, value):
        self._support = value

    @abstractmethod
    def probability(self, value):
        raise NotImplementedError

    @abstractmethod
    def cumulative_probability(self, value):
        raise NotImplementedError

    @abstractmethod
    def mean(self):
        raise NotImplementedError

    @abstractmethod
    def median(self):
        raise NotImplementedError

    @abstractmethod
    def mode(self):
        raise NotImplementedError

    @abstractmethod
    def variance(self):
        raise NotImplementedError

    @abstractmethod
    def std(self):
        raise NotImplementedError

    @abstractmethod
    def skewness(self):
        raise NotImplementedError

    @abstractmethod
    def excess_kurtosis(self):
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    @abstractmethod
    def value(self, probability):
        raise NotImplementedError

    @abstractmethod
    def entropy(self):
        raise NotImplementedError

    @abstractmethod
    def kl_divergence(self, distribution):
        if self.support != distribution.support:
            raise ValueError("Both distributions must have the same support. Found {} and {}"
                             .format(self.support, distribution.support))


class Bernoulli(Distribution):

    def __init__(self, p):
        super().__init__(support=[0, 1])
        self.p = p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if not 0 <= value <= 1:
            raise ValueError("p must be in [0, 1]")
        self._p = value

    def mean(self):
        return self.p

    def median(self):
        if self.p < 0.5:
            return 0
        else:
            return 1

    def mode(self):
        if self.p < 0.5:
            return 0
        else:
            return 1

    def variance(self):
        return self.p * (1 - self.p)

    def std(self):
        return math.sqrt(self.variance())

    def skewness(self):
        return ((1 - self.p) - self.p) / (math.sqrt(self.p * (1 - self.p)) + 1e-6)

    def excess_kurtosis(self):
        return (1 - 6 * self.p * (1 - self.p)) / (self.p * (1 - self.p) + 1e-6)

    def probability(self, value):
        if value == 0:
            return 1 - self.p
        elif value == 1:
            return self.p
        else:
            raise ValueError("The value is not in the support of this probability distribution")

    def cumulative_probability(self, value):
        if value < 0:
            return 0
        elif 0 <= value < 1:
            return 1 - self.p
        else:
            return 1

    def entropy(self):
        cument = 0
        for x in self.support:
            cument += self.probability(x) * math.log2(self.probability(x))
        return - cument

    def kl_divergence(self, distribution):
        super().kl_divergence(distribution)
        div = 0
        for v in self.support:
            div += self.probability(v) \
                   * math.log2(distribution.probability(v) / self.probability(v))
        return - div

    def get_params(self):
        return {'theta': self.p}

    def value(self, probability):
        return int(probability > (1 - self.p))


class Categorical(Distribution):

    def __init__(self, probs):
        super().__init__(support=list(range(len(probs))))
        self.probabilities = probs

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, value):
        if sum(value) != 0:
            raise ValueError("Probabilities must sum to 1.")
        self._probabilities = value

    def probability(self, value):
        if value not in self.support:
            raise ValueError("Value is not in the support of this distribution.")
        return self.probabilities[value]

    def cumulative_probability(self, value):
        maxval = 0
        for i in self.support:
            if value > i:
                maxval = i
            else:
                break
        return sum(self.probabilities[0:maxval + 1])

    def mean(self):
        return sum([x * p for x, p in zip(self.support, self.probabilities)])

    def median(self):
        cumprob = 0.
        for x, prob in zip(self.support, self.probabilities):
            cumprob += prob
            if cumprob >= 0.5:
                return x

    def mode(self):
        return max(self.support, key=self.probabilities)

    def variance(self):
        var = 0
        mean = self.mean()
        for x in self.support:
            var += self.probabilities[x] * (x - mean) ** 2
        return var

    def std(self):
        return math.sqrt(self.variance())

    def skewness(self):
        moment = 0
        mean = self.mean()
        for x in self.support:
            moment += self.probabilities[x] * (x - mean) ** 3
        normalized_moment = (moment ** 2) / (self.std() ** 6)
        return math.copysign(normalized_moment, moment)

    def excess_kurtosis(self):
        moment = 0
        mean = self.mean()
        for x in self.support:
            moment += self.probabilities[x] * (x - mean) ** 4
        return moment / (self.std() ** 4)

    def get_params(self):
        return {
            'p{}'.format(x): p for x, p in zip(self.support, self.probabilities)
        }

    def value(self, probability):
        pass

    def entropy(self):
        cument = 0
        for x in self.support:
            cument += self.probability(x) * math.log2(self.probability(x))
        return - cument

    def kl_divergence(self, distribution):
        super().kl_divergence(distribution)
        div = 0
        for v in self.support:
            div += self.probability(v) \
                   * math.log2(distribution.probability(v) / self.probability(v))
        return - div


class Binomial(Distribution):

    def __init__(self, n, p):
        super().__init__(list(range(n)))
        self.p = p
        self.n = n

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if not 0 <= value <= 1:
            raise ValueError("p must be in [0, 1]")
        self._p = value

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if not 2 <= value:
            raise ValueError("n must be greater or equal than 2.")
        self._n = value

    def probability(self, value):
        if value not in self.support:
            raise ValueError("The value is not in the support of this probability distribution")
        return ncr(self.n, value) * (self.p ** value) * ((1 - self.p) ** (self.n - value))

    def cumulative_probability(self, value):
        cumprob = 0.
        for x in self.support:
            if value >= x:
                cumprob += self.probability(x)
            else:
                return cumprob
        return cumprob

    def value(self, probability):
        cumprob = 0.
        for x in self.support:
            cumprob += self.probability(x)
            if cumprob >= probability:
                return x
        return self.n - 1

    def mean(self):
        return self.n * self.p

    def median(self):
        return math.floor(self.n * self.p)

    def mode(self):
        return math.floor((self.n + 1) * self.p)

    def variance(self):
        return self.n * self.p * (1 - self.p)

    def std(self):
        return math.sqrt(self.variance())

    def skewness(self):
        return ((1 - self.p) - self.p) / (math.sqrt(self.n * self.p * (1 - self.p)) + 1e-6)

    def excess_kurtosis(self):
        return (1 - 6 * self.p * (1 - self.p)) / (self.n * self.p * (1 - self.p) + 1e-6)

    def get_params(self):
        return {'n': self.n, 'p': self.p}

    def entropy(self):
        return (1. / 2) * math.log2(2 * math.pi * math.e * self.n * self.p * (1 - self.p) + 1e-6)

    def kl_divergence(self, distribution):
        div = 0.
        for x in self.support:
            div += self.probability(x) * math.log2(distribution.probability(x) / self.probability(x) + 1e-6)
        return - div


class Poisson(Distribution):

    def __init__(self, lamb):
        super().__init__(range(sys.maxsize))
        self.lamb = lamb

    @property
    def lamb(self):
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        if value <= 0:
            raise ValueError("Lambda parameter must be a positive real.")
        self._lamb = value

    def probability(self, value):
        return (self.lamb ** value) * math.exp(-self.lamb) / math.factorial(value)

    def cumulative_probability(self, value):
        # TODO: Check if args are in the right order
        return gammaincc(math.floor(value + 1), self.lamb)

    def value(self, probability):
        # TODO: Check if you need to subtract 1 from the result
        # TODO: Check if args are in the right order
        return gammainccinv(probability, self.lamb)

    def mean(self):
        return self.lamb

    def median(self):
        return math.floor(self.lamb + 1 / 3 - 0.02 / self.lamb)

    def mode(self):
        return math.floor(self.lamb)

    def variance(self):
        return self.lamb

    def std(self):
        return math.sqrt(self.lamb)

    def skewness(self):
        return self.lamb ** (-1 / 2)

    def excess_kurtosis(self):
        return 1. / self.lamb

    def get_params(self):
        return {'lambda': self.lamb}

    def entropy(self):
        # TODO: Check if it works well for small lambda
        return (1 / 2) * math.log2(2 * math.pi * math.e * self.lamb) \
               - 1 / (12 * self.lamb) - 1 / (24 * self.lamb ** 2) - 19 / (360 * self.lamb ** 3)

    def kl_divergence(self, distribution):
        if type(distribution) is not Poisson:
            raise NotImplementedError("Only KL between Poisson distributions is implemented.")
        else:
            return distribution.lamb - self.lamb + self.lamb * math.log2(self.lamb / distribution.lamb)


class Normal(Distribution):

    def __init__(self, mu, sigma):
        super().__init__(support='R')
        self.mu = mu
        self.sigma = sigma

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("Sigma must be a positive real.")
        self._sigma = value

    def probability(self, value):
        return 1. / (self.sigma * np.sqrt(2 * math.pi)) * math.exp(-1. / 2 * ((value - self.mu) / self.sigma) ** 2)

    def cumulative_probability(self, value):
        return 1/2 * (1 + erf((value - self.mean())/(self.std()*math.sqrt(2))))

    def value(self, probability):
        return self.mean() + self.std() * math.sqrt(2) * ierf(2*probability - 1)

    def mean(self):
        return self.mu

    def median(self):
        return self.mu

    def mode(self):
        return self.mu

    def variance(self):
        return self.sigma ** 2

    def std(self):
        return self.sigma

    def skewness(self):
        return 0

    def excess_kurtosis(self):
        return 0

    def get_params(self):
        return {'mu': self.mu, 'sigma': self.sigma}

    def entropy(self):
        return (1/2) * math.log2(2*math.pi*math.e*(self.sigma**2))

    @abstractmethod
    def kl_divergence(self, distribution):
        if type(distribution) is not Normal:
            raise NotImplementedError("Only KL between Normal distributions is implemented.")
        else:
            return (1 / 2) * ((self.sigma / distribution.sigma) ** 2 + (distribution.mu - self.mu) ** 2 /
                              (distribution.sigma ** 2) - 1 + 2 * math.log(distribution.sigma / self.sigma))
