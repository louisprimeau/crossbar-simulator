import torch
from ..crossbar import crossbar

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, x, W, b):
        ctx.save_for_backward(x, W, b)
        assert len(x.size()) == 2 and x.size(1) == 1, "vector wrong shape"
        return ticket.vmm(x, v_bits=8) + b
        
    @staticmethod
    def backward(ctx, dx):
        x, W, b = ctx.saved_tensors
        return (None, 
                torch.transpose(torch.transpose(dx, 0, 1).matmul(W), 0, 1), 
                dx.matmul(torch.transpose(x,0,1)), 
                torch.eye(b.numel())
               )

# Implements A*x + b
# now with batching!
# handles inputs of size (N, input_size, 1), outputs (N, output_size, 1)
class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, cb, bias=None, W=None):
        super(Linear, self).__init__()

        self.W = W if W is not None else torch.nn.parameter.Parameter((torch.rand(output_size, input_size)-0.5)*2 / (output_size * input_size)**0.5)
        self.b = torch.zeros(output_size, 1, requires_grad=False) if bias is None else torch.nn.parameter.Parameter(torch.rand(output_size, 1))
        self.cb = cb 
        self.ticket = cb.register_linear(torch.transpose(self.W, 0, 1))
        self.f = linear()
        self.cbon = False
        
    def forward(self, x):
        assert len(x.size()) == 3 and x.size(2) == 1, "input invalid shape"
        if self.cbon: return torch.cat([self.f.apply(self.ticket, item, self.W, self.b) for item in x], axis=0)
        else: return self.W.unsqueeze(0).expand(x.size(0), -1, -1).bmm(x) + self.b.unsqueeze(0).expand(x.size(0), -1, -1)

    def remap(self, W=None):
        if W is None:
            self.ticket.remap(torch.transpose(self.W, 0, 1))
        else:
            self.ticket.remap(torch.transpose(W, 0, 1))
            self.W = W
    
    def use_cb(self, state):
        self.cbon = state
