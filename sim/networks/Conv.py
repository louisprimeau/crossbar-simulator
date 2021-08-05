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
        self.kernel = Linear.Linear(self.input_size * kernel_size**2, out_channels, cb)
        self.bias = torch.nn.parameter.Parameter(torch.rand(out_channels, 1, 1))
        self.cbon = False
        print(in_channels, out_channels)

    def forward(self, inp):
        assert inp.size(1) == self.input_size, "inp channels = {}, but in channels was {} on declaration".format(inp.size(1), self.input_size)

        if self.cbon:
            return conv2d.apply(inp, self.kernel.W.reshape(self.input_size, self.output_size, 3, 3), self.bias, self.kernel)
        else:
            return f.conv2d(inp, self.kernel.W.reshape(self.input_size, self.output_size, 3, 3), bias=self.bias, padding=1) 
    
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

        batches = []
        for x in image:            
            x = x.unsqueeze(0) # So the tensors are still rank 4
            pd = 1
            padded = torch.nn.functional.pad(x, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
            rows = []
            for i in range(pd, x.size(2) + pd):
                cols = []
                for j in range(pd, x.size(3) + pd):
                    cols.append(linear_method(padded[:, :, i-1:i+2, j-1:j+2].reshape(-1, 1)).reshape(1,-1,1,1) + bias.unsqueeze(0))
                rows.append(torch.cat(cols, axis=3))
            batches.append(torch.cat(rows, axis=2))
        output = torch.cat(batches, axis=0)
        return output
    
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
