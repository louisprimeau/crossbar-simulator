import torch
from ..crossbar import crossbar

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, x, W, vbits=16):
        ctx.save_for_backward(x, W)
        assert len(x.size()) == 2 and x.size(1) == 1, "vector wrong shape"
        return ticket.vmm(x, v_bits=vbits)
        
    @staticmethod
    def backward(ctx, dx):
        x, W, b = ctx.saved_tensors
        return (None, 
                torch.transpose(torch.transpose(dx, 0, 1).matmul(W), 0, 1), 
                dx.matmul(torch.transpose(x,0,1)),
                None,
                None,
               )

# Implements A*x + b
# now with batching!
# handles inputs of size (N, input_size, 1), outputs (N, output_size, 1)
class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, cb, W=None, bias=False, vbits=16, bypass=False):
        super(Linear, self).__init__()

        self.bias = bias
        self.W = W if W is not None else torch.nn.parameter.Parameter((torch.rand(output_size, input_size + self.bias)-0.5)*2 / (output_size * input_size)**0.5)
        self.cb = cb 
        self.ticket = cb.register_linear(torch.transpose(self.W, 0, 1), bypass=bypass)
        self.f = linear()
        self.cbon = False
        self.vbits = vbits
        self.bypass = bypass
        
    def forward(self, x):
        
        assert (len(x.size()) == 3) and ((x.size(1) + self.bias) == self.W.size(1)) and (x.size(2) == 1), "input invalid shape"
        
        if self.bias: x = torch.cat((x, torch.ones(x.size(0), 1, 1)), axis=1)

        if self.cbon:
            out = torch.cat([self.f.apply(self.ticket, item, self.W, self.vbits).reshape(1, -1) for item in x], axis=0)
        else:
            out =  self.W.unsqueeze(0).expand(x.size(0), -1, -1).bmm(x).squeeze(2)
        
        return out

    def remap(self, W=None):
        if W is None:
            self.ticket.remap(torch.transpose(self.W, 0, 1), bypass=self.bypass)
        else:
            self.ticket.remap(torch.transpose(W, 0, 1), bypass=self.bypass)
            self.W = W
    
    def use_cb(self, state):
        self.cbon = state
