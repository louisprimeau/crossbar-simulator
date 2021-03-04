
import torch
import cbroutines
import crossbartesting
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import train
pi = 3.14159265359

# MODEL DEF
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(RNN, self).__init__()
        self.cb = crossbartesting.crossbar(device_params)
        self.node_rnn = cbroutines.NODERNN(input_size, hidden_layer_size, self.cb)
        self.linear = cbroutines.Linear(hidden_layer_size, output_size, self.cb)
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

# DEVICE PARAMS for convenience.        
device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 32,
                 "n": 32,
                 "r_on_mean": 1e4,
                 "r_on_stddev": 1e3,
                 "r_off_mean": 1e5,
                 "r_off_stddev": 1e4,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "device_resolution": 4,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20, 
                 "p_stuck_on": 0.001,
                 "p_stuck_off": 0.001}


# MAKE DATA
n_pts = 150
size = 1
tw = 10

x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5

data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)), (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)] 
train_data, test_data = data[:30], data[30:]

fig1, ax1 = plt.subplots()
for i in range(1):
    model = RNN(1, 4, 1)
    losses = train.train(train_data, model)
    print(model.node_rnn.observer.history[0])
    model.node_rnn.observe(True)    
    output, times = train.test(test_data[0][0][0], test_data[0][0][1], 40, model)
    print(losses[-1], torch.linalg.norm(output - y[50:]))
    ax1.plot(torch.cat((x.view(-1)[39].view(-1), times.view(-1)), axis=0), torch.cat((y.view(-1)[39].view(-1), output.view(-1)), axis=0), linewidth=1.0, color='pink')
    ax1.plot(x.squeeze(), y.squeeze(), linewidth=0.5, color='k')

ax1.axvline(x=float(x.squeeze()[39]), color='k')
plt.show()

plt.plot(model.node_rnn.observer.history[1], model.node_rnn.observer.history[0], color='k')
plt.show()

