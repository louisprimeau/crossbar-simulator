import torch
from . import Linear

def add_crossbar(linear_method, *args):

    n_ims, n_channels, n_rows, n_cols = args[0].size()
    images = []
    for i in range(n_ims):
        channels = []
        for j in range(n_channels):
            rows = []
            for k in range(n_rows):
                cols = []
                for l in range(n_cols):
                    inp = torch.cat([arg[i,j,k,l].reshape(1, -1, 1) for arg in args], axis=1)
                    cols.append(linear_method(inp).reshape(1,1,1,1))
                rows.append(torch.cat(cols, axis=3))
            channels.append(torch.cat(rows, axis=2))
        images.append(torch.cat(channels, axis=1))
    return torch.cat(images, axis=0)

class add(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weights, linear_method, *tensors):
        ctx.save_for_backward(weights, *tensors)
        assert len(weights) == len(tensors)
        return add_crossbar(linear_method, *tensors) 

    @staticmethod
    def backward(ctx, d_output):
        weights, tensors = ctx.saved_tensors[0], ctx.saved_tensors[1:]
        return None, None, *(weight * d_output for weight in weights)

class Add(torch.nn.Module):

    def __init__(self, weights, cb):
        super(Add, self).__init__()
        self.weights = weights
        self.cb = cb
        self.linear_combo = Linear.Linear(weights.size(0), 1, cb, W=weights.reshape(1, -1), bias=False)
        self.adder = add()

        self.cbon = False

    def forward(self, *tensors):
        if self.cbon:
            return self.adder.apply(self.weights, self.linear_combo, *tensors)
        else:
            return sum(weight * tensor for weight, tensor in zip(self.weights, tensors))

    def use_cb(self, state):
        self.cbon = state
        self.linear_combo.use_cb(state)
        
