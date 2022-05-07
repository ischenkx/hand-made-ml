import numpy as np
import torch
from .derivatives import Derivatives
from .utils import prepare


class Tensor(object):
    @staticmethod
    def as_expr(t):
        if t.ntype == 'var':
            return str(t.data)
        if t.ntype == 'sum':
            return f'({Tensor.as_expr(t.children[0])} + {Tensor.as_expr(t.children[1])})'
        if t.ntype == 'mul':
            return f'({Tensor.as_expr(t.children[0])} * {Tensor.as_expr(t.children[1])})'
        if t.ntype == 'neg':
            return f'-({Tensor.as_expr(t.children[0])})'
        if t.ntype == 'div':
            return f'({Tensor.as_expr(t.children[0])} / {Tensor.as_expr(t.children[1])})'

    _id_generator = 0

    def __init__(self,
                 data,
                 autograd=True,
                 children=None,
                 derivative=None,
                 dtype=np.float64,
                 ntype="var"):
        self.data = prepare(data, dtype)
        self.grad = None
        self.autograd = autograd
        self.children = children
        self.derivative = derivative
        self.ntype = ntype
        self.dtype = dtype
        self.id = None
        self._generate_id()

    def _generate_id(self):
        self.id = Tensor._id_generator
        Tensor._id_generator += 1

    def __str__(self):
        return f'type=\'{self.ntype}\' data={self.data}'

    def __add__(self, other: 'Tensor'):
        return Tensor(data=self.data + other.data,
                      autograd=True,
                      children=(self, other),
                      derivative=Derivatives.sum,
                      dtype=self.dtype,
                      ntype='sum')

    def __matmul__(self, other):
        if self.data.ndim > 2 or other.data.ndim > 2:
            raise Exception('currently, matmul is supported only for <=2-dimensional tensors')

        b = other.data
        a = self.data

        return Tensor(data=a @ b,
                      autograd=True,
                      children=(self, other),
                      derivative=Derivatives.matmul,
                      dtype=self.dtype,
                      ntype='matmul')

    def __mul__(self, other: 'Tensor'):
        return Tensor(data=self.data * other.data,
                      autograd=True,
                      children=(self, other),
                      derivative=Derivatives.multiplication,
                      dtype=self.dtype,
                      ntype='mul')

    def __neg__(self):
        return Tensor(data=-self.data,
                      autograd=True,
                      children=(self,),
                      derivative=Derivatives.negation,
                      dtype=self.dtype,
                      ntype='neg')

    def __sub__(self, other: 'Tensor'):
        a = self
        b = -other
        return b + a

    def __truediv__(self, other: 'Tensor'):
        return Tensor(data=self.data / other.data,
                      autograd=True,
                      children=(self, other),
                      derivative=Derivatives.division,
                      dtype=self.dtype,
                      ntype='div')

    # def cosine(self):

    def topsort(self, blacklist=None):
        if blacklist is None:
            blacklist = {}
        if self.id not in blacklist:
            blacklist[self.id] = 0
            yield self
        if self.children is not None:
            for child in self.children:
                for n in child.topsort(blacklist):
                    yield n

    def accumulate(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def backward(self, grad=None):
        if not self.autograd:
            return

        if self.grad is None:
            if grad is None:
                grad = np.ones(self.data.shape, dtype=self.dtype)
            self.grad = prepare(grad)

        for n in self.topsort():
            if n.children is None:
                continue

            gradient = n.derivative([x.data for x in n.children], n.grad)
            for child, g in zip(n.children, gradient):
                child.accumulate(g)

    def zero_grad(self):
        self.grad = None
        if self.children is not None:
            for child in self.children:
                child.zero_grad()
