from math import pi

import numpy as np

from filtering import *
from plot import plot


def func(x):
    return np.sin(x) + 0.5


x_min = 0
x_max = pi
l = x_max - x_min

d = 0.25  # noise
k = 100  # sampling

# Random search params
e = 0.1
p = 0.95

x = np.array(list(x_min + i * (x_max - x_min) / k for i in range(k)))
y = func(x)
y_noise = np.random.normal(y, d)

# r = 3
a = search_alpha_vector(y_noise, r=3, e=e, p=p, l=l)
y_filtered = filter_signal(y_noise, r=3, a=a)
plot(x, y, y_noise, y_filtered)

# r = 5
a = search_alpha_vector(y_noise, r=5, e=e, p=p, l=l)
y_filtered = filter_signal(y_noise, r=5, a=a)
plot(x, y, y_noise, y_filtered)