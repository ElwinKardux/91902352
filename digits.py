################################################################################
#                                                                              #
#                               INTRODUCTION                                   #
#                                                                              #
################################################################################

# In order to help you with the first assignment, this file provides a general
# outline of your program. You will implement the details of various pieces of
# Python code grouped in functions. Those functions are called within the main
# function, at the end of this source file. Please refer to the lecture slides
# for the background behind this assignment.
# You will submit three python files (sonar.py, cat.py, digits.py) and three
# pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain
# trained models for each tasks.
# Good luck!

################################################################################
#                                                                              #
#                                    CODE                                      #
#                                                                              #
################################################################################


import numpy as np
import random
import matplotlib.pyplot as plt
#%matplotlib inline
import math
import pickle

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def softmax(u):

    return np.exp(u) / np.sum(np.exp(u))

def cross_entropy_loss(yhat, y):
    logprobs = np.multiply(y,np.log(yhat))
    cost = - np.sum(logprobs) 
    #logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    #cost = - np.sum(logprobs) / m
    return cost

def softmax_cross_entropy_error_back(y, t):
    #np.dot(softmax(y)*()
    return

def make_one_hot(d):
    # return a one-hot vector representation of digit d
    zero_vector = np.zeros((10,), dtype=int)
    zero_vector[d] = 1
    return zero_vector


class Digits_Model:

    def __init__(self, dim_input, dim_hidden, dim_out, weights1=None, weights2=None, bias1=None, bias2=None,
                 activation1=(lambda x: x), activation2=(lambda x: x)):
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.w1 = weights1 or np.random.randn(self.dim_hidden, self.dim_input)
        self.w2 = weights2 or np.random.randn(self.dim_out, self.dim_hidden)
        self.b1 = bias1 if bias1 is not None else np.random.randn(self.dim_hidden, 1)
        self.b2 = bias2 if bias2 is not None else np.random.randn(self.dim_out, 1)
        self._a1 = activation1
        self._a2 = activation2
        pass

    def __str__(self):
        info = "Digits model neuron\n\
        \tInput dimension: %d\n\
        \tHidden dimension: %d\n\
        \tdim out: %d\n\
        \tweights1: %s\n\
        \tweights2: %s\n\
        \tbias1: %s\n\
        \tbias2: %s\n\
        \tActivation1: %s\n\
        \tActivation2: %s\n"% (self.dim_input, self.dim_hidden, self.dim_out, self.w1, self.w2, self.b1, self.b2, self._a1.__name__, self._a2.__name__)
        return info

    def __call__(self, x):
        '''
        return the output of the model for a given input
        '''
        z1 = np.dot(self.w1, np.expand_dims(x, axis=1)) + self.b1
        a1 = self._a1(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self._a2(z2)
        return a2

    def predict(self, x):
        '''
        returns a digit
        '''
        a2 = self(x)
        probabilities = softmax(a2)
        # d = np.argmax(probabilities)
        return probabilities

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
        pickle.dump(self, open('sonar_model.pkl', 'wb'))
        pass


class Digits_Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        pass

    def accuracy(self, data):
        '''
        return the accuracy on data given data iterator
        '''
        # acc = 100*np.mean([1 if self.model.predict(x) == np.argmax(y) else 0 for x, y in data])
        acc = 100 * np.mean([np.dot(self.model.predict(x).T, np.expand_dims(y, axis=1)) for x, y in data])
        return acc

    def train(self, lr, ne):

        '''
        This method should:
        1. display initial accuracy on the training data loaded in the constructor
        2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        3. display final accuracy
        '''
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        m = 1.0
        for epoch in range(ne):
            logprobs = 0
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                # print("x = " ,x.shape)
                yhat = self.model(x)
                # print('y =' , y.shape)
                # print('w1 =', self.model.w1.shape)
                # print("b1 = ",self.model.b1.shape)
                z1 = np.dot(self.model.w1, np.expand_dims(x, axis=1)) + self.model.b1
                # print("z1 = " , z1.shape)
                a1 = self.model._a1(z1)
                # print("a1 = " , a1.shape)
                z2 = np.dot(self.model.w2, a1) + self.model.b2
                # print('w2 =', self.model.w2.shape)
                # print("b2 = ",self.model.b2.shape)
                # print("z2 = ", z2.shape)
                a2 = self.model._a2(z2)
                logprobs -= np.sum(np.multiply(y, np.log(yhat)))
                # print("a2 = ", a2.shape)
                dz2 = a2 - np.expand_dims(y, axis=1)
                # print("dz2 = ", dz2.shape)
                dw2 = np.dot(dz2, a1.T)
                # print("dw2 = ", dw2.shape)
                db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
                # print("db2 = ", db2.shape)
                dz1 = np.multiply(np.dot(self.model.w2.T, dz2), sigmoid(z1) * (1 - sigmoid(z1)))
                # print("dz1 = ", dz1.shape)
                dw1 = (1 / m) * np.dot(dz1, np.expand_dims(x, axis=1).T)
                # print("dw1 = ", dw1.shape)
                db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
                # print("db1 = ", db1.shape)
                self.model.w1 -= lr * dw1
                self.model.b1 -= lr * db1
                self.model.w2 -= lr * dw2
                self.model.b2 -= lr * db2
            accuracy = self.accuracy(self.dataset)
            cost = logprobs / m
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f, cost=%.3f' % (epoch + 1, lr, accuracy, cost))
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))


class Digits_Data:
    def __init__(self, relative_path, data_file_name, batch_size=None):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''

        with open(relative_path + data_file_name, 'rb') as f:
            self.digits_data = pickle.load(f)

        self.processed_data = []
        for x in self.digits_data['train']:
            new_digit = [(digit.flatten(), make_one_hot(x)) for digit in self.digits_data['train'][x]]
            self.processed_data += new_digit
        self.index = -1
        if batch_size == None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size
        self.m = len(self.processed_data)
        print(self.batch_size)
        print(self.m)
        self.num_complete_minibatches = math.floor(self.m / self.batch_size)
        self.lastbatch = self.m - self.num_complete_minibatches
        self._shuffle(self.processed_data)
        pass

    def __iter__(self):
        '''
        See example code (ngram) in lecture slides
        '''
        return self

    def __next__(self):
        '''
        See example code (ngram) in slides
        '''
        self.index += 1
        if self.index == len(self.processed_data):
            self.index = -1
            raise StopIteration
        return self.processed_data[self.index]
        # else  self.index== len(self.processed_data) + batchsize-1

    def _shuffle(self, dataset):
        '''
        shuffle the data iterator
        '''
        np.random.shuffle(dataset)
        pass

    def __len__(self):
        return len(self.processed_data)

def main():
    data = Digits_Data('data/assignment1/', 'digits_data.pkl')
    model = Digits_Model(dim_input = 784, dim_hidden = 15, dim_out=10, activation1 = sigmoid, activation2 = softmax)  # specify the necessary arguments
    trainer = Digits_Trainer(data, model)
    trainer.train(lr =  0.001, ne = 5) # experiment with learning rate and number of epochs
    model.save_model()

if __name__ == '__main__':
    main()
