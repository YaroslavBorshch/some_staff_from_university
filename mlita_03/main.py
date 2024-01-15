import numpy
from plot import make_plot
import matplotlib.pyplot as plt


def func_1(x):
    return -0.5 * numpy.cos(x * 0.5) + 1


def func_2(x):
    return func_1(x) * numpy.sin(5 * x)


make_plot(func_1, -2.5, 1.5, 1, 211, '-0.5cos(0.5x) + 1')
make_plot(func_2, -2.5, 1.5, 1, 212, 'f(x)sin(5x)')

plt.show()