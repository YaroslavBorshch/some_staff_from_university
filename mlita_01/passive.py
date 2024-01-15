from math import floor

from numpy import linspace
from prettytable import PrettyTable


def steps_count_opt(l, e):
    return 2 * l / e + 1


def opt_passive_method(f, a, b, n):
    print("Optimal passive method")
    pretty_table = PrettyTable()
    pretty_table.field_names = ["Step", "x", "f(x)"]

    x_prev = a
    f_prev = f(a)
    i = 0
    for x in linspace(a, b, num=round(n + 2)):
        fi = f(x)
        pretty_table.add_row(
            [i, format(floor(x * 10000) / 10000, '.4f'),
             format(floor(fi * 1000000) / 1000000, '.6f')])
        i += 1
        if fi > f_prev:
            print(pretty_table)
            return x_prev
        else:
            x_prev = x
            f_prev = fi