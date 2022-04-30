import ray
from model import Model

class RayGradientCalculator(object):
    @ray.remote
    class Actor(object):
        # initializer is a dict of arguments to a model copy
        def __init__(self, initializer):
            initializer['initialize'] = False
            self.model = Model(**initializer)
            self.model.reset_inputs()
            self.model.reset_outputs()

        def calc_weight_grad(self, inp, tar, weights):
            self.model.weights = weights
            self.model.forward(inp)
            grad = self.model.calc_input_grad(tar)
            return self.model.calc_weight_grad(grad)

    def __init__(self, model, workers):
        self.max_workers = workers
        self.model = model
        self.initializer = None
        self.pool = None
        self._initialize()


    def _initialize(self):
        self.initializer = {
            'layers': self.model.layers,
            'loss_set': (self.model.loss, self.model.loss_d),
            'dtype': self.model.dtype,
            'initialize': False
        }

        self.pool = []
        for _ in range(self.max_workers):
            self.pool.append(RayGradientCalculator.Actor.remote(self.initializer))

    def _get_actor(self):
        if len(self.pool) == 0:
            return None
        return self.pool.pop()

    def _release_actors(self, *actors):
        for actor in actors:
            if actor not in self.pool:
                self.pool.append(actor)

    def run(self, batch):
        batch_size = len(batch)
        if batch_size == 0:
            return

        object_refs = []

        weights_ref = ray.put(self.model.weights)
        object_refs.append(weights_ref)

        result_w_grad = None

        idx = 0

        while idx < len(batch):
            gradient_tasks = []
            actors = []

            while idx < len(batch):
                trainer = self._get_actor()
                if trainer is None:
                    break

                inp, tar = batch[idx]
                idx += 1

                inp_ref = ray.put(inp)
                tar_ref = ray.put(tar)

                object_refs.append(inp_ref)
                object_refs.append(tar_ref)
                actors.append(trainer)

                task = trainer.calc_weight_grad.remote(inp_ref, tar_ref, weights_ref)
                gradient_tasks.append(task)

            grads = ray.get(gradient_tasks)

            w_grad = list(map(lambda x: sum(x) / batch_size, zip(*grads)))

            if result_w_grad is None:
                result_w_grad = w_grad
            else:
                for i in range(len(w_grad)):
                    result_w_grad[i] += w_grad[i]

            del gradient_tasks
            self._release_actors(*actors)

        del object_refs

        return result_w_grad