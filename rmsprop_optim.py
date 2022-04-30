import numpy as np

class RMSPropOptimizer(object):
    def __init__(self, model, alpha, momentum, forget_factor):
        self.alpha = alpha
        self.momentum = momentum
        self.forget_factor = forget_factor
        self.model = model
        self.deltas = [np.zeros(wmat) for wmat in self.model.weights_layout()]
        self.rms_prop_deltas = [np.zeros(wmat) for wmat in self.model.weights_layout()]

    def step(self, grad):
        eps = 1e-8
        for i, wg in enumerate(grad):
            # RMSProp
            v = self.forget_factor * self.rms_prop_deltas[i] + \
                (1 - self.forget_factor) * (wg ** 2)

            # delta for momentum
            delta = self.alpha * wg / (v ** 0.5 + eps) + \
                    self.momentum * self.deltas[i]

            self.model.weights[i] -= delta
            self.deltas[i] = delta
            self.rms_prop_deltas[i] = v
