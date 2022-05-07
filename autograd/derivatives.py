import numpy as np

from autograd.utils import as_matrix


class Derivatives(object):
    @staticmethod
    def sum(tensors, grad):
        return [np.ones(t.shape) * grad for t in tensors]

    @staticmethod
    def multiplication(tensors, g):
        grad = []
        for i, t1 in enumerate(tensors):
            partial_grad = np.ones(t1.shape)
            for j, t2 in enumerate(tensors):
                if i == j:
                    continue
                partial_grad *= t2
            partial_grad *= g
            grad.append(partial_grad)
        return grad

    @staticmethod
    def division(tensors, grad):
        if len(tensors) != 2:
            raise Exception('unexpected amount of tensors for division (currently only two operands are supported)')

        [f, g] = tensors
        return [
            grad / g,
            -f * grad / (g * g)
        ]

    @staticmethod
    def negation(tensors, grad):
        return [-1 * grad * np.ones(tensors[0].shape)]

    @staticmethod
    def matmul(tensors, grad):
        grad = as_matrix(grad)
        [a, b] = map(as_matrix, tensors)
        return [grad @ b.T, a.T @ grad]
