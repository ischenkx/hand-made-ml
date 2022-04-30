from model import Model, layer
from activators import sigmoid_activator
from losses import mse_loss_set
from rmsprop_optim import RMSPropOptimizer
from gradient_calc import RayGradientCalculator
import numpy as np

print('testing...')

layout = (
    layer(2, None),
    layer(2, sigmoid_activator),
    layer(1, None)
)

xor_model = Model(layout, mse_loss_set)

grad_calc = RayGradientCalculator(xor_model, 4)
optimizer = RMSPropOptimizer(xor_model, 0.1, 0.01, 0.01)

dataset = [
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
]

for inp, tar in dataset:
    print(inp, xor_model.forward(inp))
print()

for epoch in range(4000):
#     print(epoch)
    if epoch % 400 == 0:
        print('epoch:', epoch)
        for inp, tar in dataset:
            print(inp, xor_model.forward(inp))
        # xor_model.log_pool_info()
        print('--------------------')
    grad = grad_calc.run(dataset)
    optimizer.step(grad)

for inp, tar in dataset:
    print(inp, xor_model.forward(inp))
