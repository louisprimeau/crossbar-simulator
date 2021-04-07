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
        self.nonlinear = torch.nn.ReLU()
        
    # Torch does N, C, H, W
    def forward(self, inp):
        batches = []
        for x in inp:
            x = x.unsqueeze(0) # So the tensors are still rank 4
            pd = self.kernel_size // 2 if self.padding else zero
            padded = torch.nn.functional.pad(x, (pd, pd, pd, pd, 0, 0, 0, 0), mode='constant')
            rows = []
            for i in range(pd, x.size(2) + pd):
                cols = []
                for j in range(pd, x.size(3) + pd):
                    cols.append(self.kernel(padded[:, :, i-1:i+2, j-1:j+2].reshape(-1, 1)).reshape(1,-1,1,1))
                rows.append(torch.cat(cols, axis=3))
            batches.append(torch.cat(rows, axis=2))
        output = torch.cat(batches, axis=0)
        return self.nonlinear(output)
    
    def remap(self):
        self.kernel.remap()

    def use_cb(self, state):
        self.kernel.cbon = state
