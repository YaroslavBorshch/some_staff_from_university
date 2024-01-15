import numpy

from plot import make_plot


def func(x):
    return -0.5 * numpy.cos(x * 0.5) + 1


make_plot(func, -2.5, 1.5, 0.1)
