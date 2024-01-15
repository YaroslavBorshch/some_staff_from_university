from copy import copy

import matplotlib.pyplot as plt


def pareto(choices, dirs, x, y, labels):
    pareto_set = list(copy(choices))

    for a in choices:
        for b in choices:
            if a[x] * dirs[x] < b[x] * dirs[x] and a[y] * dirs[y] < b[y] * dirs[
                y]:
                pareto_set.remove(a)
                break

    max_x = max(choices, key=lambda a: a[x])[x]
    max_y = max(choices, key=lambda a: a[y])[y]
    utopia = (max_x if dirs[x] == 1 else 0, max_y if dirs[y] == 1 else 0)

    show_pareto(choices, labels, choices, utopia, x, y)
    show_pareto(choices, labels, pareto_set, utopia, x, y)

    return choices.index(
        min(pareto_set, key=lambda a: pareto_dist((a[x], a[y]), utopia)))


def pareto_dist(choice, utopia):
    return abs(choice[0] - utopia[0]) ** 2 + abs(choice[1] - utopia[1]) ** 2


def show_pareto(choices, labels, pareto_set, utopia, x, y):
    for choice in pareto_set:
        plt.plot(choice[x], choice[y], 'o', label=labels[choices.index(choice)])
    plt.plot(utopia[0], utopia[1], 'x', label='Utopia')
    plt.grid(True)
    plt.legend()
    plt.show()