import matplotlib.pyplot as plt
import numpy

from fibonacci import fibonacci_method, steps_count_fib
from passive import steps_count_opt, opt_passive_method


def make_plot(f, a, b, e):
    x = numpy.arange(a, b, 0.02)
    y = f(x)

    n = steps_count_opt(4, e)
    opt_passive_min = opt_passive_method(f, a, b, n)

    n = steps_count_fib(4, e)
    fibonacci_min = fibonacci_method(f, a, b, n)

    lines = plt.plot(x, y, '-', label="-0.5cos(0.5x)+1")
    plt.plot(opt_passive_min, f(opt_passive_min), 'o', label='passive')
    plt.plot(fibonacci_min, f(fibonacci_min), 'o', label='fibonacci')
    plt.setp(lines[0], markersize=2)
    plt.grid(True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()