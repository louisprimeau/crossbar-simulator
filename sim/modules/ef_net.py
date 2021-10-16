import torch
from ..crossbar import crossbar
from . import Linear
from . import Conv
from . import util
# Wx + Iy + b
import time
class simple_EF_block(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, cb):
        super(simple_EF_block, self).__init__()
        
        self.kernel_size = kernel_size
        self.input_size = in_channels
        self.cb = cb

        self.t0, self.t1, self.N = 0, 1, 3
        self.h = (self.t1 - self.t0) / self.N
        
        self.conv1 = Conv.Conv2d(in_channels, in_channels, 3, self.cb)
        self.relu1 = torch.nn.ReLU()

        self.id_kernel, id_kernel_as_m = util.identity_kernel(self.input_size, self.kernel_size)
        self.conv2_kernel, conv2_kernel_as_m = util.kaiming_kernel(self.input_size, self.input_size, self.kernel_size)
        self.bias2, bias2_as_m = util.normal_kernel(self.input_size)
        
        stacked_kernel_as_m = torch.cat((id_kernel_as_m, conv2_kernel_as_m * self.h, bias2_as_m * self.h), axis=1)
        self.kernel_and_id = Linear.Linear(stacked_kernel_as_m.size(1), stacked_kernel_as_m.size(0), cb, W=stacked_kernel_as_m, bias=True)
        
        self.conv2 = conv2d_plusId()

        self.cbon = False
        
    def forward(self, x):

        for _ in range(self.N):
            x_temp = self.relu1(self.conv1(x))
            if self.cbon: x = self.conv2.apply(x_temp, x, self.conv2_kernel, self.bias2, self.kernel_and_id)
            else: x = x + self.h * torch.nn.functional.conv2d(x_temp, self.conv2_kernel, self.bias2, padding=1)

        return torch.nn.functional.relu(x)

    def remap(self):
        conv2_kernel_as_m = util.kernel_to_matrix(self.conv2_kernel)
        id_kernel_as_m = util.kernel_to_matrix(self.id_kernel)
        bias2_as_m = self.bias2.unsqueeze(1)
        
        self.conv1.remap()
        self.kernel_and_id.remap(torch.cat((id_kernel_as_m, conv2_kernel_as_m * self.h, bias2_as_m * self.h), axis=1))

    def use_cb(self, state):
        self.cbon = state
        self.conv1.use_cb(state)

# handles autograd so that I can specify derivative in terms of fast torch functions
# instead of making torch backpopagate through concatenation of a bunch of linear functions.
# kernel and bias are passed in solely for backprop.
# note that kernel should be the actual conv kernel and not include the id or bias term. 
class conv2d_plusId(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image1, image2, kernel, bias, func_method):
        assert image1.size() == image2.size(), "input images not same shape: {} and {}".format(image1.size(), image2.size())
        ctx.save_for_backward(image1, image2, kernel, bias)
        
        pd = 1
        pad_image_1 = torch.nn.functional.pad(image1, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
        pad_image_2 = torch.nn.functional.pad(image2, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
        batches = []
        for batch in range(pad_image_1.size(0)):
            rows = []
            for row in range(1, pad_image_1.size(-2)-1):
                cols = []
                for col in range(1, pad_image_1.size(-1)-1):
                    inp = torch.cat((pad_image_2[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1),
                                     pad_image_1[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1)), axis=1)
                    cols.append(func_method(inp).reshape(1, -1, 1, 1))
                    #if bias is not None: cols[-1] += bias.reshape(1, -1, 1, 1)

                rows.append(torch.cat(cols, axis=3))
            batches.append(torch.cat(rows, axis=2))
        return torch.cat(batches, axis=0)
    
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

    
class conv2d_plus_rk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image1, image2, kernel, bias, func_method):
        assert image1.size() == image2.size(), "input images not same shape: {} and {}".format(image1.size(), image2.size())
        ctx.save_for_backward(image1, image2, kernel, bias)
        
        pd = 1
        pad_image_1 = torch.nn.functional.pad(image1, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
        pad_image_2 = torch.nn.functional.pad(image2, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
        batches = []
        for batch in range(pad_image_1.size(0)):
            rows = []
            for row in range(1, pad_image_1.size(-2)-1):
                cols = []
                for col in range(1, pad_image_1.size(-1)-1):
                    inp = torch.cat((pad_image_2[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1),
                                     pad_image_1[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1)), axis=1)
                    cols.append(func_method(inp).reshape(1, -1, 1, 1))
                    #if bias is not None: cols[-1] += bias.reshape(1, -1, 1, 1)

                rows.append(torch.cat(cols, axis=3))
            batches.append(torch.cat(rows, axis=2))
        return torch.cat(batches, axis=0)
    
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
