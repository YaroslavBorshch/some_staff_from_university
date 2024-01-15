from abc import ABC, abstractmethod
from typing import List
import matplotlib.pyplot as plt
import itertools
import math


class Neuron(ABC):
    def __init__(self, weights: List, learning_rate: float, t: List):
        self.weights = weights
        self.learning_rate = learning_rate
        self.t = t
        self.X = list(itertools.product([0, 1], repeat=4))

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def forward(self, x) -> int:
        pass

    def result(self) -> List[int]:
        res = []
        for i in range(16):
            res.append(self.forward(self.X[i]))
        return res

    def hamming_dist(self, y: List[int]) -> int:
        dist = 0
        for i in range(16):
            dist += y[i] ^ self.t[i]
        return dist

    @staticmethod
    def plot(epochs: int, E: List, title: str):
        plt.title(title)
        plt.xlabel("k")
        plt.ylabel("E(k)")
        plt.grid()
        plt.plot(range(epochs + 1), E)
        plt.show()

    @staticmethod
    def to_dec(number):
        dec = 0
        for i in range(len(number)):
            dec += int(number[i]) * (2 ** (len(number) - i - 1))
        return dec


class NeuronWithUnitStepFunction(Neuron):
    def train(self):
        epochs = 100
        E_list = []
        for epoch in range(epochs):
            Y = self.result()
            E = self.hamming_dist(Y)
            print(f"Epoch {epoch}: Y = {Y}, W = {self.weights} E = {E}")
            E_list.append(E)
            if E == 0:
                self.plot(epoch, E_list, "Зависимоть E от k")
                break

            for i in range(16):
                y = self.forward(self.X[i])
                for j in range(5):
                    if j == 0:
                        self.weights[j] += self.learning_rate * (self.t[i] - y) * 1
                    else:
                        self.weights[j] += self.learning_rate * (self.t[i] - y) * self.X[i][j - 1]

    def forward(self, x) -> int:
        net = 0
        for i, w in enumerate(self.weights[1:]):
            net += w * x[i]
        net += self.weights[0]
        return 1 if net >= 0 else 0


class NeuroneWithSigmoidActivationFunction(Neuron):
    def train(self, X=None, logging=True) -> bool:
        if X is None:
            X = self.X

        epochs = 300
        E_list = []
        for epoch in range(epochs):
            Y = self.result()
            E = self.hamming_dist(Y)
            if logging:
                print(f"Epoch {epoch}: Y = {Y}, W = {self.weights} E = {E}")
            E_list.append(E)
            if E == 0:
                if logging:
                    self.plot(epoch, E_list, "Зависимость E от k, задания №2 и №3")
                return True

            for i in range(len(X)):
                y = self.forward(X[i])
                delta = self.t[self.to_dec(X[i])] - y
                derivative = self.sigmoid_function(X[i]) * (1 - self.sigmoid_function(X[i]))
                for j in range(5):
                    if j == 0:
                        self.weights[j] += self.learning_rate * delta * derivative
                    else:
                        self.weights[j] += self.learning_rate * delta * X[i][j - 1] * derivative
        return False

    def train_partly(self):
        for i in range(2, 16):
            combinations = itertools.combinations(self.X, i)
            flag = False
            for item in combinations:
                self.weights = [0, 0, 0, 0, 0]
                successful_learning = self.train(item, False)
                if successful_learning:
                    flag = True
                    print(f"Набор из {i} векторов: {item}")
                    self.weights = [0, 0, 0, 0, 0]
                    self.train(item, True)
                    break
            if flag:
                break

    def forward(self, x) -> int:
        return 1 if self.sigmoid_function(x) >= 0.5 else 0

    def sigmoid_function(self, x) -> float:
        net = 0
        for i, w in enumerate(self.weights[1:]):
            net += w * x[i]
        net += self.weights[0]
        return 1 / (1 + math.exp(-net))
