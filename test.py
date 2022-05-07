import numpy as np

from autograd.tensor import Tensor
from nn.layers import Sequential, Linear

if __name__ == '__main__':
    t = Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print(t / Tensor([
        [1, 2, 3],
    ]))


    def optimize(layer, alpha=0.001):
        for param in layer.params():
            param.data -= param.grad * alpha


    seq = Sequential(
        Linear(10, 20),
        Linear(20, 30),
        Linear(30, 10),
    )

    inp = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    res = seq.forward(inp)
    res.backward()

    optimize(seq)

    res.zero_grad()

    print('done!')