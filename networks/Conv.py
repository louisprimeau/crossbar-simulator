import torch
from . import Linear, ODERNN

class Conv(torch.nn.Module):
    def __init__(self, kernel_size, input_size, output_size, cb, padding=True):
        super(Conv, self).__init__()
        
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.output_size = output_size
        self.cb = cb
        self.padding = padding

        print(output_size, input_size * kernel_size**2)
        self.kernel = Linear.Linear(self.input_size * kernel_size**2, output_size, cb)
        print(self.kernel.W.size())
        self.nonlinear = torch.nn.ReLU()
        
    def forward(self, x):
        pd = self.kernel_size // 2 if self.padding else zero
        padded = torch.nn.functional.pad(x, (0, 0, pd, pd, pd, pd), mode='constant')
        rows = []
        for i in range(pd, x.size(0) + pd):
            cols = []
            for j in range(pd, x.size(0) + pd):
                cols.append(self.kernel(padded[i-1:i+2, j-1:j+2, :].reshape(-1)).reshape(1,1,-1))
            rows.append(torch.cat(cols, axis=1))
        return self.nonlinear(torch.cat(rows, axis=0))
    
    def remap(self):
        self.kernel.remap()

