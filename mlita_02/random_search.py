import random
from math import log


def steps_count(l, e, p):
    return int(round(log(1 - p) / log(1 - e / l)))


def search(f, a, b, n):
    x_min = a
    f_min = f(a)
    for i in range(n):
        x = random.uniform(a, b)
        y = f(x)
        if y < f_min:
            x_min = x
            f_min = y

    return x_min