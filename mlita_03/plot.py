import numpy
import matplotlib.pyplot as plt

from simulated import search


def make_plot(f, a, b, d, sub_plot, label):
    fig = plt.figure()
    x = numpy.arange(a, b, 0.001)
    y = f(x)
    rand_min = search(f, a, b, d)

    ax = fig.add_subplot(sub_plot)

    lines = ax.plot(x, y, '-', label=label)
    ax.plot(rand_min, f(rand_min), 'o')

    plt.setp(lines[0], markersize=2)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()