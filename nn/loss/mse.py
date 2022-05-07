from autograd import Tensor
from nn.loss.base import Base


class MSE(Base):
    def __init__(self):
        super().__init__()

    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        diff = output - target
        square_diff = diff * diff
