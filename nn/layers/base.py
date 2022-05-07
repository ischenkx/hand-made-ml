from autograd import Tensor


class Base(object):
    def __init__(self):
        pass

    # returns an iterator of learnable parameters
    # of this block of neural network
    def params(self):
        pass

    def forward(self, tensor: Tensor):
        raise Exception('unreachable: base layer can\'t be used for backpropagation')
