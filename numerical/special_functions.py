import math

A = (8 / (3 * math.pi) * (math.pi - 3) / (4 - math.pi))


# Taken from Winitzki 2008: "A handy approximation for the error function and its inverse"
def erf(x):
    erf_sq = 1 - math.exp(-x ** 2 * ((4 / math.pi) + (A * x ** 2)) / (1 + A * x ** 2))
    return math.sqrt(erf_sq)


def ierf(y):
    ierf_sq = - (2 / (math.pi * A)) - (math.log(1 - y ** 2) / 2) + \
              math.sqrt(((2 / (math.pi * A)) + (math.log(1 - y ** 2) / 2) - (1 / A) * math.log(1 - y ** 2)))
    return math.sqrt(ierf_sq)
