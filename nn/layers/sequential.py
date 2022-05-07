from autograd import Tensor
from nn.layers.base import Base


class Sequential(Base):
    def __init__(self, *layers: [Base]):
        super().__init__()
        self.layers = layers

    def params(self):
        for layer in self.layers:
            for params in layer.params():
                yield params

    def forward(self, tensor: Tensor):
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor
