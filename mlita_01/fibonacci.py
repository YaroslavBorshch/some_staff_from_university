from math import sqrt, floor
from prettytable import PrettyTable


def fibonacci_method(f, a, b, n):
    print("Fibonacci method")
    pretty_table = PrettyTable()
    pretty_table.field_names = ["Step", "a", "b", "len", "x1", "x2", "f(x1)",
                                "f(x2)"]

    fn = fibonacci_number(n + 1)
    fn_prev = fibonacci_number(n)

    x1 = a + (fn - fn_prev) / fn * (b - a)
    x2 = a + fn_prev / fn * (b - a)

    for i in range(n):
        fx1 = f(x1)
        fx2 = f(x2)
        len = b - a

        pretty_table.add_row(
            [i + 1, format(floor(a * 10000) / 10000, '.4f'),
             format(floor(b * 10000) / 10000, '.4f'),
             format(floor(len * 10000) / 10000, '.4f'),
             format(floor(x1 * 10000) / 10000, '.4f'),
             format(floor(x2 * 10000) / 10000, '.4f'),
             format(floor(fx1 * 1000000) / 1000000, '.6f'),
             format(floor(fx2 * 1000000) / 1000000, '.6f')]
        )

        if fx1 > fx2:
            a = x1
            x1 = x2
            x2 = b - x1 + a
        else:
            b = x2
            x2 = x1
            x1 = b - x2 + a

    print(pretty_table)
    return (a + b) / 2


# Функция вычисления числа итераций
def steps_count_fib(l, e):
    i = 2
    fn = l / e
    while fibonacci_number(i) < fn:
        i += 1
    return i - 1


# Функция вычисления числа Фиббоначи
def fibonacci_number(n):
    return int(((1 + sqrt(5)) ** n - (1 - sqrt(5)) ** n) / (2 ** n * sqrt(5)))