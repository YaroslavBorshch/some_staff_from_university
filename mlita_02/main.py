from math import floor

import numpy
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from plot import make_plot

from random_search import steps_count, search

a = -2.5
b = 1.5
p = numpy.arange(0.9, 0.999, 0.01)
q = numpy.arange(0.01, 0.11, 0.01)

pretty_table = PrettyTable()
pretty_table.field_names = list(["q\\P"]) + list([round(p_i, 2) for p_i in p])


def func_1(x):
    return -0.5 * numpy.cos(x * 0.5) + 1


def func_2(x):
    return func_1(x) * numpy.sin(5 * x)


def make_table(f):
    for q_i in q:
        n = [steps_count(b - a, q_i, p_i) for p_i in p]
        x_min = [search(func_1, a, b, n_i) for n_i in n]
        x_min = [format(floor(x_i * 10000) / 10000, '.4f') for x_i in x_min]
        row = [round(q_i, 2)] + x_min
        pretty_table.add_row(row)

    print(pretty_table)
    pretty_table.clear_rows()


for i in q:
    n = [steps_count(b - a, i, p_i) for p_i in p]
    row = [round(i)] + n
    pretty_table.add_row(row)

print(pretty_table)
pretty_table.clear_rows()

make_table(func_1)
make_table(func_2)

make_plot(func_1, a, b, 0.1, 0.95, 211, '-0.5cos(0.5x) + 1')
make_plot(func_2, a, b, 0.1, 0.95, 212, 'f(x)sin(5x)')

plt.show()