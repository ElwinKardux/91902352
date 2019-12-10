# imports

import numpy as np
import random
import math
import pickle

#with open('cat_data.pkl', 'rb') as f:
#    cat_data = pickle.load(f)
with open('\\Users\\Elwin\\PycharmProjects\\data\\assignment1\\digits_data.pkl', 'rb') as f:
    digits_data = pickle.load(f)
print((digits_data['train'][9].shape))
x = []
x = np.array(x)
print(x)
#b = np.concatenate((x,digits_data['train'][9]))
new_digit = [(np.concatenate(([1], digit.flatten()),axis = 0), digit) for digit in digits_data['train'][9]]
new_digit = np.array(new_digit)
print(new_digit.shape)
processed_data = np.zeros((1,2))
print(processed_data)
d = [2, 3]
D = d[1:]
print(D)
print(len(digits_data['train']))

processed_data = np.zeros((1,2))
new_digit = []
for x in range(len(digits_data['train'])):
    new_digit = [(np.concatenate(([1], digit.flatten()),axis = 0), x) for digit in digits_data['train'][x]]
    processed_data = np.concatenate((processed_data,new_digit), axis = 0)
processed_data = processed_data[1:]
print(len(processed_data))
#print(processed_data)
#a = [[2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20]]
#print(len(a))
#a = np.array(a)
#print(a.shape)

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

def newloss(yhat,y):
    return -(y*np.log(yhat) + (1-y) * np.log(1-yhat))

# prediction functions

def ppredict(self, x):
    return self(x)

def lrpredict(self, x):
    return 1 if self(x)>0.5 else 0

# extra

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


class Neuron:

    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x), predict=(lambda x: x)):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):
        yhat = self._a(np.dot(self.w, np.array(x)))
        return yhat

    def predict(self, x):
        return self._a(np.sum(np.dot(self.w, x)))


class Trainer:

    def __init__(self, model, filename):
        self.model = model
        self.loss = lrloss
        data = None
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.dataset = [(np.concatenate(([1], cat.flatten()), axis=0), 1) for cat in data["train"]['cat']] + \
                       [(np.concatenate(([1], no_cat.flatten()), axis=0), 0) for no_cat in data["train"]['no_cat']]
        np.random.shuffle(self.dataset)

    def cost(self, data):

        return np.mean([self.loss(self.model.predict(x), y) for x, y in data])

    def accuracy(self, data):
        return 100 * np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])

    def train(self, lr, ne):

        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))

        for epoch in range(ne):
            J = 0
            dw = 0
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                z = np.dot(self.model.w, x)
                self.model.a = sigmoid(z)
                J += -1.0 * (y * np.log(self.model.a) + (1 - y) * np.log(1 - self.model.a))
                dz = self.model.a - y
                dw += np.dot(x, dz)
            J /= len(self.dataset)
            dw /= len(self.dataset)
            self.model.w -= lr * dw
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f, J = %.3f' % (epoch + 1, lr, accuracy, J))

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))
        print(np.array(self.dataset).shape)

model = Neuron(dimension= 12289, activation= sigmoid, predict= lrpredict)

trainer = Trainer(model,'C:\\Users\\Elwin\\PycharmProjects\\data\\assignment1\\cat_data.pkl')

trainer.train(0.0001, 10)