import random
import torch
import numpy as np
from tensor import Tensor


def generate_graph(ops, shape, depth, meta=None):
    if depth == 0:
        torch_tensor = torch.rand(size=shape, dtype=torch.float64, requires_grad=True)
        my_tensor = Tensor(torch_tensor.detach().numpy(), dtype=np.float64)
        if meta is not None:
            if 'nodes' not in meta:
                meta['nodes'] = []
            meta['nodes'].append((my_tensor, torch_tensor))
        return my_tensor, torch_tensor

    op = random.choice(ops)
    my_left, torch_left = generate_graph(ops, shape, depth - 1, meta)
    my_right, torch_right = generate_graph(ops, shape, depth - 1, meta)
    res1, res2 = op(my_left, my_right), op(torch_left, torch_right)
    res2.retain_grad()

    if meta is not None:
        if 'nodes' not in meta:
            meta['nodes'] = []
        meta['nodes'].append((res1, res2))
    return res1, res2


def test(ops, depth, shape=(1,)):
    meta = {}
    my_root, torch_root = generate_graph(ops, shape, depth, meta)

    my_root.backward()
    torch_root.backward(torch.ones(shape))

    for my_node, torch_node in meta['nodes']:
        ml, tl = my_node.grad, torch_node.grad
        if ml is None or tl is None:
            if ml != tl:
                return False
            continue
        tl = tl.numpy()

        epsilon = max(max(abs(ml.mean()), abs(tl.mean())) / 1000, 1e-4)
        diff = abs(ml - tl)
        if (diff > epsilon).any():
            print('failure!')
            print('my grad:')
            print(ml)
            print('torch grad:')
            print(tl)
            print('diff:')
            print(diff)
            # print(Tensor.as_expr(my_op))
            print(my_node.ntype)
            return False
    return True

OPS = [
    lambda a, b: a * b,
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: -a,
    lambda a, b: a + (-b),
    lambda a, b: a / b,
    lambda a, b: a @ b
]

print('Special cases:')


print('Automatic testing:')
TESTS = 10000

for i in range(TESTS):
    if not test(OPS, 10, shape=(2, 2)):
        print('stopped at', i + 1)
        break
    if i % 100 == 99:
        print('test:', i + 1)
else:
    print('PASSED')