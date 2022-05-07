from autograd import Tensor


class Base(object):
    def __init__(self):
        pass

    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        raise Exception('base loss is not supposed to be used')