from copy import copy
from math import log, floor
from random import uniform

from numpy.ma import arange
from prettytable import PrettyTable

from plot import show_points


def filter_signal(signal, r, a):
    filtered = copy(signal)
    for i in range(len(signal) - floor(r / 2) - 1):
        filtered[i] = sum(
            (signal[i - floor(r / 2) + j] * a[j] for j in range(r))
        )

    return filtered


def search_alpha_vector(signal, r, e, p, l):
    pretty_table = PrettyTable()
    pretty_table.field_names = ["h", "dis", "alpha", "w", "d"]
    points = {'w': list(), 'd': list()}

    dis_min = 1

    for h in arange(0, 1.05, 0.1):
        j, a, w, d = search_optimal_j(signal, r, h, e, p, l)
        dis = dist(w, d)

        pretty_table.add_row(
            [
                format(h, '.1f'),
                format(dis, '.4f'),
                [format(ai, '.4f') for ai in a],
                format(w, '.4f'),
                format(d, '.4f')
            ]
        )

        points['w'].append(w)
        points['d'].append(d)

        if dis < dis_min:
            dis_min = dis
            h_opt = h
            j_opt = j
            a_opt = a
            w_opt = w
            d_opt = d

    print(pretty_table)

    pretty_table = PrettyTable()
    pretty_table.field_names = ["h*", "J", "w", "d"]
    pretty_table.add_row(
        [
            format(h_opt, '.1f'),
            format(j_opt, '.4f'),
            format(w_opt, '.4f'),
            format(d_opt, '.4f')
        ]
    )
    print(pretty_table)

    show_points(points['w'], points['d'])

    return a_opt


def search_optimal_j(signal, r, h, e, p, l):
    n = round(log(1 - p) / log(1 - e / l))

    j_min = 1
    a_opt = [0] * r

    for _ in range(n):
        a = [0] * r

        a[floor(r / 2)] = uniform(0, 1)

        for m in range(1, floor(r / 2)):
            a[m] = uniform(0, 1 - sum(a[m + 1:r - m])) * 0.5
            a[r - m - 1] = a[m]

        a[0] = (1 - sum(a[1:r - 1])) * 0.5
        a[r - 1] = a[0]

        filtered = filter_signal(signal, r, a)
        w = pollution(filtered)
        d = difference(signal, filtered)
        j = J(w, d, h)

        if j < j_min:
            j_min = j
            a_opt = a
            w_opt = w
            d_opt = d

    return j_min, a_opt, w_opt, d_opt


def J(w, d, h):
    return h * w + (1 - h) * d


def pollution(filtered):
    return max(
        (abs(filtered[i] - filtered[i - 1]) for i in range(len(filtered) - 1))
    )


def difference(noise, filtered):
    return max((abs(noise[i] - filtered[i - 1]) for i in range(len(noise) - 1)))


def dist(w, d):
    return max(w, d)