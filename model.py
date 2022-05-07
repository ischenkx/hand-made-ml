import numpy as np

def layer(neurons, fset, bias=True):
    act = fset[0] if fset is not None else None
    actd = fset[1] if fset is not None else None

    return {
        'activation': act,
        'activation_d': actd,
        'neurons': neurons,
        'bias': bias
    }

class Model(object):
    def __init__(self,
                 layers,
                 loss_set,
                 initialize=True,
                 dtype=np.float64,
                 ):
        self.layers = layers
        self.dtype = dtype
        self.loss = loss_set[0]
        self.loss_d = loss_set[1]
        # self.trainers = []

        self.weights = None
        self.inputs = None
        self.outputs = None

        if initialize:
            self.initialize()

    def zero_array(self, shape):
        return np.zeros(shape, dtype=self.dtype)

    def random_array(self, shape):
        return np.random.random(shape).astype(self.dtype)

    def layer(self, layer):
        return self.layers[layer]

    def has_bias(self, layer):
        # output layer can't have a bias
        if layer == self.count_layers() - 1:
            return False
        return self.layer(layer)['bias']

    # returns an activation function for the layer
    def activator(self, layer):
        return self.layer(layer)['activation']

    # returns a derivative of an activation function for the layer
    def activator_d(self, layer):
        return self.layer(layer)['activation_d']

    # returns the amount of neurons in the layer
    def count_neurons(self, layer):
        bias = int(self.has_bias(layer))
        return self.layer(layer)['neurons'] + bias

    # returns the amount of nn in the model
    def count_layers(self):
        return len(self.layers)

    def reset_outputs(self):
        self.outputs = []
        for i in range(self.count_layers()):
            neurons = self.count_neurons(i)
            self.outputs.append(self.zero_array((neurons,)))

    def reset_inputs(self):
        self.inputs = []
        for i in range(self.count_layers()):
            neurons = self.count_neurons(i)
            self.inputs.append(self.zero_array((neurons,)))

    def reset_weights(self):
        self.weights = []
        for i in range(self.count_layers()):
            neurons = self.count_neurons(i)
            if i > 0:
                w_shape = (self.count_neurons(i - 1), neurons)
                self.weights.append(self.random_array(w_shape))

    def initialize(self):
        self.reset_inputs()
        self.reset_outputs()
        self.reset_weights()

    def forward(self, x):
        if len(self.inputs[0]) - int(self.has_bias(0)) != len(x):
            raise Exception('incorrect input size')

        for i, val in enumerate(x):
            self.inputs[0][i] = val
            self.outputs[0][i] = val

        if self.has_bias(0):
            self.inputs[0][-1] = 1
            self.outputs[0][-1] = 1

        for i in range(1, self.count_layers()):
            activate = self.activator(i)
            self.inputs[i] = self.outputs[i - 1] @ self.weights[i - 1]
            if activate is not None:
                self.outputs[i] = activate(self.inputs[i])
            else:
                self.outputs[i] = self.inputs[i].copy()

            if self.has_bias(i):
                self.outputs[i][-1] = 1
                self.inputs[i][-1] = 1

        return self.outputs[-1].copy()

    def calc_input_grad(self, target):
        grad = [
            self.zero_array((self.count_neurons(i),))
            for i in range(self.count_layers())
        ]

        output_layer = self.count_layers() - 1
        o_activate_d = self.activator_d(output_layer)
        o_outputs = self.outputs[output_layer]
        o_inputs = self.inputs[output_layer]
        grad[output_layer] = self.loss_d(o_outputs, target)

        if o_activate_d is not None:
            grad[output_layer] *= o_activate_d(o_inputs)
        for i in range(self.count_layers() - 2, 0, -1):
            activate_d = self.activator_d(i)
            grad[i] = grad[i + 1] @ self.weights[i].T
            if activate_d is not None:
                grad[i] *= activate_d(self.inputs[i])
        return grad

    def weights_layout(self):
        return [wmat.shape for wmat in self.weights]

    def calc_weight_grad(self, input_grad):
        layers = self.count_layers()
        w_grad = [None] * (layers - 1)
        for output_layer in range(layers - 1):
            inputs = self.outputs[output_layer]
            grad = input_grad[output_layer + 1]
            w_grad[output_layer] = np.outer(inputs, grad)
        return w_grad