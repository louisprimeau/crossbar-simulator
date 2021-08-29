import torch
from ..crossbar import crossbar
from . import Linear
from . import Conv
# Wx + Iy + b
import time
class simple_EF_block(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, cb):
        super(simple_EF_block, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = in_channels
        self.cb = cb

        self.conv1 = Conv.Conv2d(in_channels, in_channels, 3, self.cb)
        self.relu1 = torch.nn.ReLU()

        self.kernel2 = torch.nn.parameter.Parameter(torch.rand(self.input_size, self.input_size, self.kernel_size, self.kernel_size) - 0.5)
        kernel2_block = self.kernel2.reshape(self.input_size, self.input_size * self.kernel_size**2)

        self.identity_kernel = torch.zeros(self.input_size, self.input_size, self.kernel_size, self.kernel_size)
        torch.nn.init.dirac_(self.identity_kernel, groups=1)
        id_kernel_block = self.identity_kernel.reshape(self.input_size, self.input_size * self.kernel_size**2)

        self.kernel_and_id = Linear.Linear(self.input_size * 2 * self.kernel_size**2, self.input_size, cb, W=torch.cat((id_kernel_block, kernel2_block), axis=1))
        self.bias2 = torch.nn.parameter.Parameter(torch.rand(in_channels))
        self.bias3 = torch.nn.parameter.Parameter(torch.rand(in_channels))
        
        coord = self.cb.mapped[self.kernel_and_id.ticket.index]
        self.conv2 = conv2d_plusId()

        #print("ef", self.cb.mapped)
        
        self.t0 = 0
        self.t1 = 1
        self.N = 3

    def forward(self, x):
        h = (self.t1 - self.t0) / self.N
        
        #x2 = self.conv2.apply(x1, x, self.kernel2, self.bias2, self.kernel_and_id)
        for i in range(self.N):
            x1 = self.relu1(self.conv1(x))
            x = h * torch.nn.functional.conv2d(x1, self.kernel2, self.bias2, padding=1) + x1
  
        return x

    def remap(self):
        kernel2_block = self.kernel2.reshape(self.input_size, self.input_size * self.kernel_size**2)
        self.conv1.remap()
        self.kernel_and_id.remap(torch.cat((self.identity_kernel.reshape(self.input_size, self.input_size * self.kernel_size**2), kernel2_block), axis=1))

# handles autograd so that I can specify derivative in terms of fast torch functions
# instead of making torch backpopagate through concatenation of a bunch of linear functions.
class conv2d_plusId(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image1, image2, kernel, bias, func_method):
        assert image1.size() == image2.size(), "input images not same shape: {} and {}".format(image1.size(), image2.size())
        ctx.save_for_backward(image1, image2, kernel, bias)

        
        id_kernel = torch.zeros(kernel.size())
        torch.nn.init.dirac_(id_kernel, groups=1)
        id_kernel_as_matrix = id_kernel.reshape(id_kernel.size(0), -1)
        kernel_as_matrix = kernel.reshape(kernel.size(0), kernel.size(1) * kernel.size(2) * kernel.size(3))
        big_kernel = torch.cat((id_kernel_as_matrix, kernel_as_matrix), axis=1)
        
        #x1 = torch.nn.functional.conv2d(image1, kernel, bias.squeeze(), padding=1) + image2
        pd = 1
        pad_image_1 = torch.nn.functional.pad(image1, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
        pad_image_2 = torch.nn.functional.pad(image2, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
        torch.set_printoptions(precision=10)
        batches = []
        for batch in range(pad_image_1.size(0)):
            rows = []
            for row in range(1, pad_image_1.size(-2)-1):
                cols = []
                for col in range(1, pad_image_1.size(-1)-1):
                    inp = torch.cat((pad_image_2[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1),
                                     pad_image_1[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1)), axis=1)
                    cols.append(func_method(inp).reshape(1, -1, 1, 1))
                    if bias is not None: cols[-1] += bias.reshape(1, -1, 1, 1)

                rows.append(torch.cat(cols, axis=3))
            batches.append(torch.cat(rows, axis=2))
        output = torch.cat(batches, axis=0)

        return output
    
    @staticmethod
    def backward(ctx, d_output):
        image1, image2, kernel, bias = ctx.saved_tensors
        pad_image_t = torch.transpose(torch.nn.functional.pad(image1, (1, 1, 1, 1)), 0, 1)
        d_output_t = torch.transpose(d_output, 0, 1)
        id_kernel = torch.zeros(kernel.size())
        torch.nn.init.dirac_(id_kernel, groups=1)
        
        out = ( torch.nn.functional.conv2d(d_output, torch.transpose(kernel, -1, -2), padding=1),
                d_output,
                torch.transpose(torch.cat([torch.nn.functional.conv2d(pad_image_t, d_output_t[i, :, :, :].unsqueeze(1), groups=image1.size(0)) for i in range(d_output.size(1))]), 0, 1).reshape(image1.size(0), kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)),
                torch.sum(d_output, (-1, -2)),
                None,
                )
        return out
