import torch
from ..crossbar import crossbar

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, cb):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.linear_in = Linear(input_size, hidden_layer_size, cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = torch.nn.Tanh()
        
    def forward(self, x):
        
        h_i = torch.zeros(self.hidden_layer_size, 1)
        for x_i in x:
            h_i = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))
        return h_i 
    
    def remap(self):
        self.linear_in.remap()
        self.linear_hidden.remap()
