import torch

# autograd func for gemv.
# alpha, A, beta, and b are included only for gradient calculation. 
class gemv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, interface, alpha, A, x, beta, b):
        ctx.save_for_backward(alpha, A, x, beta, b)
        out = interface.vmm(torch.cat((x, torch.ones(1, 1))))
        return out
        
    @staticmethod
    def backward(ctx, dx):
        alpha, A, x, beta, b = ctx.saved_tensors
        return (None,
                torch.ones_like(alpha),
                torch.transpose(torch.transpose(dx, 0, 1).matmul(A), 0, 1),
                dx.matmul(torch.transpose(x,0,1)),
                torch.ones_like(beta),
                torch.eye(b.numel()),
               )

# Implements α*A*x + β*b
# handles inputs of size (N, input_size, 1), outputs (N, output_size, 1)
#
# A is matrix of size m x n
# b is vector of size m x 1
# x is vector of size n x 1
# α, β are scalars
class GEMV(torch.nn.Module):
    def __init__(self, A, b, alpha, beta, sys):
        super(GEMV, self).__init__()

        self.A, self.b, self.alpha, self.beta = torch.nn.Parameter(A), torch.nn.Parameter(b), torch.nn.Parameter(alpha), torch.nn.Parameter(beta)
        
        self.rows, self.cols = self.A.size()
        assert b.size() == (self.rows, 1), "bias has incorrect shape {}, should be {}".format(bias.size(), (1, self.cols))
        assert alpha.size() == (1,), "alpha has incorrect shape {}, should be {}".format(alpha.size(), (1,))
        assert beta.size() == (1,), "beta has incorrect shape {}, should be {}".format(beta.size(), (1,))

        matrix = torch.cat((self.A.transpose(0, 1) * self.alpha, self.b.transpose(0, 1) * self.beta))

        self.interface = sys.program(matrix) #crossbar interface
        self.f = gemv() #autograd wrapper

    def forward(self, x):
        assert x.size() == (self.cols, 1), "Input x is wrong shape {}, should be {}".format((self.cols, 1), (x.size()))        
        return self.f.apply(self.interface, self.alpha, self.A, x, self.beta, self.b)

    def free(self):
        self.interface.free()