import torch
from . import Linear, util
import torch.nn.functional as f
import torch.nn.grad as grad

def conv2d_crossbar(image, kernel, bias, linear_method, padding, stride):
    pd = padding
    pad_image = f.pad(image, (pd, pd, pd, pd), mode='constant')
    batches = []
    for batch in range(pad_image.size(0)):
        rows = []
        for row in range(1, pad_image.size(-2)-1, stride):
            cols = []
            for col in range(1, pad_image.size(-1)-1, stride):
                cols.append(linear_method(pad_image[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1)).reshape(1,-1,1,1))
            rows.append(torch.cat(cols, axis=-1))
        batches.append(torch.cat(rows, axis=-2))
    return torch.cat(batches, axis=0)
        
class conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, kernel, bias, linear_method, padding=1, stride=1):

        ctx.save_for_backward(image, kernel, bias)
        ctx.params = stride, padding        
        return conv2d_crossbar(image, kernel, bias, linear_method, padding, stride)

    def backward(ctx, d_output):
        image, kernel, bias = ctx.saved_variables
        stride, padding = ctx.params

        d_input = grad.conv2d_input(image.shape, kernel, d_output, stride, padding, 1, 1)
        d_weight = grad.conv2d_weight(image, kernel.shape, d_output, stride, padding, 1, 1)
        d_bias = torch.sum(d_output, (-1, -2)).view(-1)

        return d_input, d_weight, d_bias, None, None, None, None

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, cb, padding=1, stride=1, dtype=torch.float):
        super(Conv2d, self).__init__()
        
        self.kernel_size, self.in_channels, self.out_channels  = kernel_size, in_channels, out_channels
        self.padding, self.stride = padding, stride
        self.cb = cb
        
        self.conv_kernel, conv_kernel_as_m = util.kaiming_kernel(self.in_channels, self.out_channels, self.kernel_size, dtype=dtype, param=True)
        self.bias, bias_as_m = util.normal_kernel(self.out_channels, dtype=dtype, param=True)

        stacked_kernel_as_m = torch.cat((conv_kernel_as_m, bias_as_m), axis=1)
        self.kernel = Linear.Linear(stacked_kernel_as_m.size(1), stacked_kernel_as_m.size(0), cb, W=stacked_kernel_as_m, bias=True)
        
        self.cbon = False

    def forward(self, inp):
        assert inp.size(1) == self.in_channels, "inp channels = {}, but in channels was {} on declaration".format(inp.size(1), self.in_channels)

        if self.cbon: return conv2d.apply(inp, self.conv_kernel, self.bias, self.kernel, padding=self.padding, stride=self.stride)
        else: return f.conv2d(inp, self.conv_kernel, self.bias, padding=self.padding, stride=self.stride)
            
    def remap(self):
        conv_kernel_as_m = util.kernel_to_matrix(self.conv_kernel)
        bias_as_m = self.bias.unsqueeze(1)
        self.kernel.remap(torch.cat((conv_kernel_as_m, bias_as_m), axis=1))
        
    def use_cb(self, state):
        self.cbon = state

