from autograd import Tensor
import numpy as np

from nn.layers.base import Base


class Linear(Base):
    def __init__(self, inputs, outputs, random_weight_init=True):
        super().__init__()

        w_shape = (inputs, outputs)

        if random_weight_init:
            self.weights = Tensor(np.random.uniform(0, 1, w_shape))
        else:
            self.weights = Tensor(np.zeros(w_shape))

    def forward(self, tensor: Tensor):
        return tensor @ self.weights

    def params(self):
        yield self.weights
