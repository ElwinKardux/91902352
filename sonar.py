# imports

import numpy as np
import random
import math
import pickle
with open('sonar_data.pkl', 'rb') as f:
    sonar_data1 = pickle.load(f)

# activation functions

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def perceptron(z):
    return -1 if z<=0 else 1

# loss functions

def ploss(yhat, y):
    return max(0, -yhat*y)

def lrloss(yhat, y):
    return 0.0 if yhat==y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))

# prediction functions

def ppredict(self, x):
    return self(x)

def lrpredict(self, x):
    return 1 if self(x)>0.5 else 0

# extra

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


class Sonar:

    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x)):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation

    def __str__(self):
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):
        yhat = self._a(np.dot(self.w, np.array(x)))
        return yhat


class Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = ploss
        data = None
        with open('sonar_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.dataset = [(np.concatenate(([1], mine), axis=0), 1) for mine in data["m"]] + \
                           [(np.concatenate(([1], rock), axis=0), -1) for rock in data["r"]]
            np.random.shuffle(self.dataset)
            self.dim = len(self.dataset[0][0])

    def cost(self, data):

        return np.mean([self.loss(self.model.predict(x), y) for x, y in data])

    def accuracy(self, data):

        return 100 * np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])

    def predict(self, v):
        return self.model._a(np.dot(self.w, v))

    def train(self, lr, ne):

        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))

        for epoch in range(ne):
            error = 0
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model.predict(x)
                error += max(0, -y * np.dot(self.model.w, x)) ** 2
                self.model.w += lr * (y - yhat) * x
            accuracy = self.accuracy(self.dataset)
            # MSE = np.sum(error**2)
            MSE = error / len(self.dataset)
            print('>epoch=%d, learning_rate=%.4f, accuracy=%.4f, MeanSquaredErro=%.4f' % (epoch + 1, lr, accuracy, MSE))

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))

    def load_model(filename):
        pickle_in = open(filename, "rb")
        self.dataset = pickle.load(pickle_in)

    def save_model(filename):
        with open(self.dataset, 'rb') as f:
            db = pickle.load(f)
        pickle_out = open(filename, "wb")
        pickle.dump(db, pickle_out)
        pickle_out.close()


model = Sonar(dimension=61, activation=perceptron)

trainer = Trainer(sonar_data1, model)
trainer.train(0.001, 500)

