# imports

import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import math
import pickle
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

    def load_model(self, file_path):
        '''
        open the pickle file and update the model's parameters
        '''
        pickle_in = open(file_path, "rb")
        self.dataset = pickle.load(pickle_in)
        pass

    def save_model(self):
        '''
        save your model as 'sonar_model.pkl' in the local path
        '''
        pickle.dump(self, open('Cats_model.pkl', 'wb'))
        pass


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

model = Neuron(dimension= 12289, activation= sigmoid, predict= lrpredict)
print(model)
trainer = Trainer(model,'C:\\Users\\Elwin\\PycharmProjects\\data\\assignment1\\cat_data.pkl')

trainer.train(0.0001, 3000)
model.save_model()