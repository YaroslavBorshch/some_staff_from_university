from math import exp, floor
import random
from prettytable import PrettyTable


def search(f, a, b, d):
    pretty_table = PrettyTable()
    pretty_table.field_names = ["N", "T", "x", "f(x)", "Вероятность", "Причина"]

    t = 10000
    t_min = 0.1

    x = random.uniform(a, b)
    y = f(x)

    x_min = x
    y_min = y
    i = 1

    pretty_table.add_row(
        [i, format(t, '.4f'), format(x, '.4f'), format(y, '.4f'), 1, "Лучше +"])

    while t > t_min:
        x_new = random.uniform(x - d, x + d)

        if x_new < a:
            x_new = random.uniform(a, x + d)
        elif x_new > b:
            x_new = random.uniform(x - d, b)

        y_new = f(x_new)
        t *= 0.95
        i += 1

        if y_new < y:
            x = x_new
            y = y_new
            pretty_table.add_row(
                [i, format(t, '.4f'), format(x_new, '.4f'),
                 format(y_new, '.4f'), 1, "Лучше +"])
        else:
            propability = exp(-(y_new - y) / t)
            if random.uniform(0, 1) < propability:
                x = x_new
                y = y_new
                pretty_table.add_row(
                    [i, format(t, '.4f'), format(x_new, '.4f'),
                     format(y_new, '.4f'),
                     format(floor(propability * 10000) / 10000, '.4f'),
                     "Случайно +"])
            else:
                pretty_table.add_row(
                    [i, format(t, '.4f'), format(x_new, '.4f'),
                     format(y_new, '.4f'),
                     format(floor(propability * 10000) / 10000, '.4f'),
                     "Игнор _"])
                continue

        if y < y_min:
            x_min = x
            y_min = y

    print(pretty_table)
    return x_min