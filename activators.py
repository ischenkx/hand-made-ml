import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)


def relu(x):
    return np.maximum(0, x)


def relu_d(z):
    return np.greater(z, 0).astype(int)


# softmax
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)


# derivative of softmax
def softmax_d(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))


sigmoid_activator = (sigmoid, sigmoid_d)
relu_activator = (relu, relu_d)
softmax_activator = (softmax, softmax_d)
