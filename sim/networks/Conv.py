import torch
import time
import torch.nn.functional as f
from . import Linear, ODERNN

# Torch module wrapper for conv2d
class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, cb):
        super(Conv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.input_size = in_channels
        self.output_size = out_channels
        self.cb = cb

        self.conv_weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        conv_weight_as_matrix = self.conv_weight.reshape(out_channels, in_channels * kernel_size**2)
        self.kernel = Linear.Linear(self.input_size * kernel_size**2, out_channels, cb, W=conv_weight_as_matrix)
        self.bias = torch.nn.parameter.Parameter(torch.rand(out_channels, 1, 1))
        self.cbon = False
        
    def forward(self, inp):
        assert inp.size(1) == self.input_size, "inp channels = {}, but in channels was {} on declaration".format(inp.size(1), self.input_size)

        #if self.cbon:
        return conv2d.apply(inp, self.kernel.W.reshape(self.input_size, self.output_size, 3, 3), self.bias, self.kernel)
        #else:
        #return f.conv2d(inp, self.kernel.W.reshape(self.input_size, self.output_size, 3, 3), bias=self.bias.squeeze(), padding=1) 
    
    def remap(self):
        self.kernel.remap()
        
    def use_cb(self, state):
        self.cbon = state

# handles autograd so that I can specify derivative in terms of fast torch functions
# instead of making torch backpopagate through concatenation of a bunch of linear functions.
class conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, kernel, bias, linear_method):

        ctx.save_for_backward(image, kernel, bias)
        
        padding=1
        pad_image = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode='constant')
        kernel_as_matrix = kernel.reshape(kernel.size(0), kernel.size(1) * kernel.size(2) * kernel.size(3))
        batches = []
        for batch in range(pad_image.size(0)):
            rows = []
            for row in range(1, pad_image.size(-2)-1):
                cols = []
                for col in range(1, pad_image.size(-1)-1):
                    cols.append(linear_method(pad_image[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(1, -1, 1)).reshape(1,-1,1,1))
                    #cols.append(kernel_as_matrix.mm(pad_image[batch:batch+1, :, row-1:row+2, col-1:col+2].reshape(-1, 1)).reshape(1, -1, 1, 1))
                    if bias is not None: cols[-1] += bias.reshape(1, -1, 1, 1)
                rows.append(torch.cat(cols, axis=-1))
            batches.append(torch.cat(rows, axis=-2))
        return torch.cat(batches, axis=0)
    
    @staticmethod
    def backward(ctx, d_output):
        st = time.time()
        image, kernel, bias = ctx.saved_tensors
        pad_image_t = torch.transpose(f.pad(image, (1, 1, 1, 1)), 0, 1)
        d_output_t = torch.transpose(d_output, 0, 1)
        out =  (torch.nn.functional.conv2d(d_output, torch.transpose(kernel, -1, -2), padding=1),
                torch.transpose(torch.cat([torch.nn.functional.conv2d(pad_image_t, d_output_t[i, :, :, :].unsqueeze(1), groups=image.size(0)) for i in range(d_output.size(1))]), 0, 1).reshape(image.size(0), kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)),
                torch.sum(d_output, (-1, -2)).unsqueeze(2).unsqueeze(2),
                None,
                )

        #print(time.time() - st)
        return out
