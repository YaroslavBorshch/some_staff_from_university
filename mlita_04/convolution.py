import numpy as np


def convolution(choices, weights):
    choices = np.array(choices)
    weights = np.array(weights)

    choices_sum = np.abs(choices).sum(axis=0)
    weights_sum = np.abs(weights).sum(axis=0)

    choices = choices / choices_sum
    weights = weights / weights_sum

    res = np.dot(choices, weights)
    print("Convolution:", res)
    return np.where(res == np.amax(res))[0][0]