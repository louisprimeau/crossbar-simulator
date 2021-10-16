import torch
from . import Linear, ODERNN
from ..crossbar import crossbar
import seaborn as sns

class RNN_ODE(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, device_params, time_steps):
        super(RNN_ODE, self).__init__()
        self.cb = crossbar.crossbar(device_params, deterministic=False)
        self.node_rnn = ODERNN.NODERNN(input_size, hidden_layer_size, self.cb, time_steps)
        self.linear = Linear.Linear(hidden_layer_size, output_size, self.cb)

    def forward(self, data):
        return self.linear(self.node_rnn(*data))
    
    def remap(self):
        self.cb.clear()
        self.node_rnn.remap()
        self.linear.remap() 

    def cmap(self, name):
        self.remap()
        cmap = sns.diverging_palette(250, 300, s=90, as_cmap=True)
        ax = sns.heatmap(self.cb.W, cmap=cmap, square=True, cbar=False)
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.figure.savefig("pic/model" + str(name) + ".png")
        return ax
        
    def use_cb(self, state):
        self.node_rnn.use_cb(state)
        self.linear.use_cb(state)
