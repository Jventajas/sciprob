import sys
import numpy as np
import math
from abc import ABC, abstractmethod
from stats.combinatorics import ncr
from numerical.special_functions import erf, ierf
from scipy.special import gamma, gammainc, gammaincc, gammainccinv, digamma, beta, betainc, hyp2f1


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
    def probability(self, x):
        raise NotImplementedError

    @abstractmethod
    def cumulative_probability(self, x):
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
    def quantile(self, p):
        raise NotImplementedError

    @abstractmethod
    def entropy(self):
        raise NotImplementedError

    @abstractmethod
    def kl_divergence(self, d):
        if self.support != d.support:
            raise ValueError("Both distributions must have the same support. Found {} and {}"
                             .format(self.support, d.support))


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

    def probability(self, x):
        if x == 0:
            return 1 - self.p
        elif x == 1:
            return self.p
        else:
            raise ValueError("The value is not in the support of this probability distribution")

    def cumulative_probability(self, x):
        if x < 0:
            return 0
        elif 0 <= x < 1:
            return 1 - self.p
        else:
            return 1

    def entropy(self):
        cument = 0
        for x in self.support:
            cument += self.probability(x) * math.log2(self.probability(x))
        return - cument

    def kl_divergence(self, d):
        super().kl_divergence(d)
        div = 0
        for v in self.support:
            div += self.probability(v) \
                   * math.log2(d.probability(v) / self.probability(v))
        return - div

    def get_params(self):
        return {'theta': self.p}

    def quantile(self, p):
        return int(p > (1 - self.p))


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

    def probability(self, x):
        if x not in self.support:
            raise ValueError("Value is not in the support of this distribution.")
        return self.probabilities[x]

    def cumulative_probability(self, x):
        maxval = 0
        for i in self.support:
            if x > i:
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

    def quantile(self, p):
        pass

    def entropy(self):
        cument = 0
        for x in self.support:
            cument += self.probability(x) * math.log2(self.probability(x))
        return - cument

    def kl_divergence(self, d):
        super().kl_divergence(d)
        div = 0
        for v in self.support:
            div += self.probability(v) \
                   * math.log2(d.probability(v) / self.probability(v))
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

    def probability(self, x):
        if x not in self.support:
            raise ValueError("The value is not in the support of this probability distribution")
        return ncr(self.n, x) * (self.p ** x) * ((1 - self.p) ** (self.n - x))

    def cumulative_probability(self, x):
        cumprob = 0.
        for x in self.support:
            if x >= x:
                cumprob += self.probability(x)
            else:
                return cumprob
        return cumprob

    def quantile(self, p):
        cumprob = 0.
        for x in self.support:
            cumprob += self.probability(x)
            if cumprob >= p:
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

    def kl_divergence(self, d):
        div = 0.
        for x in self.support:
            div += self.probability(x) * math.log2(d.probability(x) / self.probability(x) + 1e-6)
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

    def probability(self, x):
        return (self.lamb ** x) * math.exp(-self.lamb) / math.factorial(x)

    def cumulative_probability(self, x):
        # TODO: Check if args are in the right order
        return gammaincc(math.floor(x + 1), self.lamb)

    def quantile(self, p):
        # TODO: Check if you need to subtract 1 from the result
        # TODO: Check if args are in the right order
        return gammainccinv(p, self.lamb)

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

    def kl_divergence(self, d):
        if type(d) is not Poisson:
            raise NotImplementedError("Only KL between Poisson distributions is implemented.")
        else:
            return d.lamb - self.lamb + self.lamb * math.log2(self.lamb / d.lamb)


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

    def probability(self, x):
        return 1. / (self.sigma * np.sqrt(2 * math.pi)) * math.exp(-1. / 2 * ((x - self.mu) / self.sigma) ** 2)

    def cumulative_probability(self, x):
        return 1/2 * (1 + erf((x - self.mean()) / (self.std() * math.sqrt(2))))

    def quantile(self, p):
        return self.mean() + self.std() * math.sqrt(2) * ierf(2 * p - 1)

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
    def kl_divergence(self, d):
        if type(d) is not Normal:
            raise NotImplementedError("Only KL between Normal distributions is implemented.")
        else:
            return (1 / 2) * ((self.sigma / d.sigma) ** 2 + (d.mu - self.mu) ** 2 /
                              (d.sigma ** 2) - 1 + 2 * math.log(d.sigma / self.sigma))


class Exponential(Distribution):

    def __init__(self, lamb):
        super().__init__(support='R0+')
        self.lamb = lamb

    @property
    def lamb(self):
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        if value <= 0:
            raise ValueError("Lambda must be a positive real.")
        self._lamb = value

    def probability(self, x):
        if x < 0:
            raise ValueError("x must be in [0-inf)")
        else:
            return self.lamb*math.exp(-self.lamb*x)

    def cumulative_probability(self, x):
        return 1 - math.exp(-self.lamb*x)

    def quantile(self, p):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0-1]")
        else:
            return -math.log(1 - p) / self.lamb

    def mean(self):
        return 1/self.lamb

    def median(self):
        return math.log(2)/self.lamb

    def mode(self):
        return 0

    def variance(self):
        return 1 / (self.lamb**2)

    def std(self):
        return 1/self.lamb

    def skewness(self):
        return 2

    def excess_kurtosis(self):
        return 6

    def get_params(self):
        return {'lambda': self.lamb}

    def entropy(self):
        return 1 - math.log(self.lamb)

    def kl_divergence(self, d):
        if type(d) != Exponential:
            raise ValueError("Only KL between Normal distributions is implemented.")
        return math.log(self.lamb / d.lamb) + (d.lamb / self.lamb) - 1


class Beta(Distribution):

    def __init__(self, a, b):
        super().__init__(support='[0-1]')
        self.a = a
        self.b = b

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value <= 0:
            raise ValueError("a must be a positive real.")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value <= 0:
            raise ValueError("a must be a positive real.")
        self._b = value

    def probability(self, x):
        return (x**(self.a - 1) * (1 - x)**(self.b - 1)) / beta(self.a, self.b)

    def cumulative_probability(self, x):
        return betainc(self.a, self.b, x) / beta(self.a, self.b)

    def quantile(self, p):
        pass

    def mean(self):
        return self.a / (self.a + self.b)

    def median(self):
        pass

    def mode(self):
        if self.a > 1 and self.b > 1:
            return (self.a-1)/(self.a + self.b - 2)
        elif self.a == 1 and self.b == 1:
            # any value between 0 and 1
            return 0.5
        elif self.a < 1 and self.b < 1:
            # for these parameter values the pdf is bimodal.
            return [0, 1]
        elif self.a <= 1 < self.b:
            return 0
        elif self.a > 1 >= self.b:
            return 1

    def variance(self):
        return self.a*self.b / ((self.a+self.b)**2 *(self.a+self.b+1))

    def std(self):
        return math.sqrt(self.variance())

    def skewness(self):
        return (2*(self.b - self.a) * math.sqrt(self.a + self.b + 1))\
               / ((self.a + self.b + 2) * math.sqrt(self.a * self.b))

    def excess_kurtosis(self):
        num = 6 * ((self.a - self.b)**2 * (self.a + self.b + 1) - self.a*self.b*(self.a + self.b + 2))
        den = self.a*self.b*(self.a+ self.b + 2)*(self.a + self.b + 3)
        return num / den

    def get_params(self):
        return {'alpha': self.a, 'beta': self.b}

    def entropy(self):
        pass

    def kl_divergence(self, d):
        pass


class Degenerate(Distribution):

    def __init__(self, k):
        super().__init__('{}'.format(k))
        self.k = k

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if not type(value) not in [float, int]:
            raise ValueError("Expected numerical value.")
        self._k = value

    def probability(self, x):
        if x == self.k:
            return 1
        else:
            return 0

    def cumulative_probability(self, x):
        if x < self.k:
            return 0
        else:
            return 1

    def quantile(self, p):
        return self.k

    def mean(self):
        return self.k

    def median(self):
        return self.k

    def mode(self):
        return self.k

    def variance(self):
        return 0

    def std(self):
        return 0

    def skewness(self):
        raise NotImplementedError("Skewness is undefined for the Degenerate distribution.")

    def excess_kurtosis(self):
        raise NotImplementedError("Excess kurtosis is undefined for the Degenerate distribution.")

    def get_params(self):
        return {'k': self.k}

    def entropy(self):
        return 0

    def kl_divergence(self, d):
        if type(d) is not Degenerate:
            raise NotImplementedError("Only KL divergence between degenerate distributions is implemented.")
        elif self.k != d.k:
            raise ValueError('KL divergence is only defined on degenerate distributions with the same k value.')
        else:
            return 0


class TStudent(Distribution):

    def __init__(self, v):
        super().__init__(support='R')
        self.v = v

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        if value < 1:
            raise ValueError("v should be greater than 1.")
        self._v = value

    def probability(self, x):
        return 1 / (math.sqrt(self.v) * beta(1/2, self.v/2)) * (1 + x**2 / self.v) ** (- (self.v + 1) / 2)

    def cumulative_probability(self, x):
        return (1/2) + x * gamma((self.v + 1) / 2) * hyp2f1(1/2, (self.v + 1)/2, 3/2, x**2/self.v)\
               / (math.sqrt(math.pi*self.v) * gamma(self.v/2))

    def quantile(self, p):
        pass

    def mean(self):
        return 0

    def median(self):
        return 0

    def mode(self):
        return 0

    def variance(self):
        if self.v > 2:
            return self.v / (self.v-2)
        else:
            return math.inf

    def std(self):
        return math.sqrt(self.variance())

    def skewness(self):
        if self.v > 3:
            return 0
        else:
            raise NotImplementedError("Skewness is undefined when v <= 3")

    def excess_kurtosis(self):
        if self.v > 4:
            return 6 / (self.v - 4)
        elif 2 < self.v <= 4:
            return math.inf
        else:
            raise NotImplementedError("Skewness is undefined when v <= 2")

    def get_params(self):
        return {'v': self.v}

    def entropy(self):
        raise NotImplementedError("We are working on it, be patient.")

    def kl_divergence(self, d):
        raise NotImplementedError("We are working on it, be patient.")


class Dirichlet(Distribution):

    def __init__(self, alpha_vec):
        super().__init__(support='(0-1)^{}'.format(len(alpha_vec)))
        self.alpha_vec = alpha_vec

    @property
    def alpha_vec(self):
        return self._alpha_vec

    @alpha_vec.setter
    def alpha_vec(self, value):
        if len(value) < 2:
            raise ValueError("Alpha vector must have at least two values.")
        for a in value:
            if a <= 0:
                raise ValueError('Alpha values must be > 0.')
        self._alpha_vec = value

    def probability(self, x):
        pass

    def cumulative_probability(self, x):
        pass

    def mean(self):
        pass

    def median(self):
        pass

    def mode(self):
        pass

    def variance(self):
        pass

    def std(self):
        pass

    def skewness(self):
        pass

    def excess_kurtosis(self):
        pass

    def get_params(self):
        pass

    def quantile(self, p):
        pass

    def entropy(self):
        pass

    def kl_divergence(self, d):
        pass

    class Gamma(Distribution):

        def __init__(self, a, b):
            super().__init__(support='R0+')
            self.a = a
            self.b = b

        @property
        def a(self):
            return self._a

        @a.setter
        def a(self, value):
            if value <= 0:
                raise ValueError("a must be > 0.")
            self._a = value

        @property
        def b(self):
            return self._b

        @b.setter
        def b(self, value):
            if value <= 0:
                raise ValueError("b must be > 0.")
            self._b = value

        def probability(self, x):
            return self.b**self.a / gamma(self.a) * x**(self.a - 1) * math.exp(- self.b * x)

        def cumulative_probability(self, x):
            return 1 / gamma(self.a) * gammainc(self.a, self.b*x)

        def quantile(self, p):
            pass

        def mean(self):
            return self.a / self.b

        def median(self):
            pass

        def mode(self):
            pass

        def variance(self):
            return self.a / (self.b**2)

        def std(self):
            return math.sqrt(self.variance())

        def skewness(self):
            return 2 / math.sqrt(self.a)

        def excess_kurtosis(self):
            return 6 / self.a

        def get_params(self):
            return {'alpha': self.a, 'beta': self.b}

        def entropy(self):
            # TODO: It's in nats cause of the exp. What to do?
            return self.a - math.log(self.b) + math.log(gamma(self.a)) + (1 - self.a)*digamma(self.a)

        def kl_divergence(self, d):
            a = (self.a - d.a)*digamma(self.a)
            b = math.log2(gamma(self.a))
            c = math.log2(gamma(d.a))
            e = d.a*(math.log2(self.b) - math.log2(d.b))
            f = self.a * (d.b - self.b) / self.b
            return a - b + c + e + f


