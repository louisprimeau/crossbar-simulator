import torch
from .crossbar import crossbar

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, x, W, b):
        ctx.save_for_backward(x, W, b)
        return ticket.vmm(x) + b
        
    @staticmethod
    def backward(ctx, dx):
        x, W, b = ctx.saved_tensors
        return (None, 
                torch.transpose(torch.transpose(dx, 0, 1).matmul(W), 0, 1), 
                dx.matmul(torch.transpose(x,0,1)), 
                torch.eye(b.numel())
               )

# Implements A*x + b
class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, cb, bias=False):
        super(Linear, self).__init__()
        self.W = torch.nn.parameter.Parameter(torch.rand(output_size, input_size))
        if bias:
            self.b = torch.nn.parameter.Parameter(torch.rand(output_size, 1))
        else:
            self.b = torch.zeros(output_size, 1, requires_grad=False)
        self.cb = cb
        self.ticket = cb.register_linear(torch.transpose(self.W,0,1))
        self.f = linear()
        self.cbon = False
        
    def forward(self, x):
        if self.cbon:
            return self.f.apply(self.ticket, x, self.W, self.b)
        else:
            return self.W.matmul(x) + self.b

    def remap(self):
        self.ticket = self.cb.register_linear(torch.transpose(self.W,0,1))
    
    def use_cb(self, state):
        self.cbon = state
"""
# Implements A*x1 + B*x2 + b
class Linear_add(torch.nn.Module):
    def __init__(self, (input_size1, input_size2), output_size, cb, bias=False):
        super(Linear_add, self).__init__()
        self.W = torch.nn.parameter.Parameter(torch.rand(output_size, input_size1 + input_size2))
        if bias:
            self.b = torch.nn.parameter.Parameter(torch.rand(output_size, 1))
        else:
            self.b = torch.zeros(output_size, 1, requires_grad=False)
        self.cb = cb
        self.ticket = cb.register_linear(torch.transpose(self.W,0,1))
        
        self.f = linear()
        self.cbon = False

    def forward(self, x1, x2):
        if self.cbon:
            return self.f.apply(self.ticket, torch.cat(x1, x2, axis=0), self.W, b)
        else:
            return self.W.matmul(torch.cat(x1, x2, axis=0)) + self.b

    def remap(self):
        self.ticket = cb.register_linear(torch.transpose(self.W,0,1))

    def use_cb(self, state):
        self.cbon = state
        
        
"""
