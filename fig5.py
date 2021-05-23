"""
Louis Primeau
University of Toronto
Feb 2021

This python script outputs graphs for Figure 5 of the paper. This includes:

part b) Crossbar heatmap at final epoch.
     - 4.png
        - .png (heatmap of ODE network weight W)
        - h.png (heatmap of hidden layer -> hidden_layer RNN weight)
        - x.png (heatmap of input -> hidden_layer RNN weights)
        - o.png (heatmap of output linear layer)
part c) RMS error vs. Epoch for 20 models.
     - training.png
part d) Diagram of evolution of hidden state and output.
     - hidden_layer.png (hidden layer vs. time)
     - output.png (output vs. time)
part e) Prediction of RNN-ODE vs. prediction of RNN with similar amount of weights.
     - comparison.png 

"""

import torch
import sim.networks.rnn_ode as rnn_ode
import train

import time
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pi = 3.14159265359


save = True

# DEVICE PARAMS for convenience.


device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 32,
                 "n": 32,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.05,
}

# MAKE DATA
n_pts = 150
size = 1
tw = 10
cutoff = 50
#x = torch.rand(1, n_pts) * 24 * pi
#x = torch.sort(x, axis=1)[0]
x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5 + torch.rand(x.size()) / 10
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)), (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)] 
train_data, test_start = data[:cutoff], data[cutoff]

# CONFIGURE PLOTS
fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

fig3, ax3 = plt.subplots()

fig4, (ax4, ax5) = plt.subplots(nrows=2, figsize=(8,6))

fig5, ax_cmap = plt.subplots(ncols=4, figsize=(20, 3))
cmap = sns.blend_palette(("#fa7de3", "#ffffff", "#6ef3ff"), n_colors=9, as_cmap=True, input='hex')

for ax in ax_cmap:
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])


# TRAIN MODELS AND PLOT
time_steps = 50
epochs = 15
num_predict= 30
start_time = time.time()

model = rnn_ode.RNN_ODE(1, 4, 1, device_params, time_steps)
losses = train.train(train_data, model, epochs)
model.node_rnn.observe(True)
# model.use_cb(True)

output, times = train.test(test_start[0][0], test_start[0][1], num_predict, model)

# model.use_cb(False)

#ax1.scatter(torch.cat((x.view(-1)[cutoff + tw - 1].view(-1), times.view(-1)), axis=0),
#         torch.cat((y.view(-1)[cutoff + tw - 1].view(-1), output.view(-1)), axis=0),
#         edgecolors='k',
#         facecolors='none')

# Predictions
ax1.plot(torch.cat((x.view(-1)[cutoff + tw - 1].view(-1), times.view(-1)), axis=0),
         torch.cat((y.view(-1)[cutoff + tw - 1].view(-1), output.view(-1)), axis=0),
         'o-',
         linewidth=0.5,
         color='deepskyblue',
         markerfacecolor='none',
         )

H = model.node_rnn.observer.history[0].detach()
t = model.node_rnn.observer.history[1].view(-1).detach()

# Hidden Layer Output
ax1.plot(t,
         model.linear(torch.transpose(H, 0, 1)).view(-1).detach(),
         ':',
         linewidth=0.5,
         color='crimson')

# Hidden layer norm
ax2.plot(t,
         torch.linalg.norm(H, ord=2, dim=1).view(-1),
         ':',
         linewidth=0.5,
         color='crimson')

# Hidden layer norm dots
ax2.scatter(t[::(time_steps + 2)],
            torch.linalg.norm(H, ord=2, dim=1)[::(time_steps + 2)],
            linewidth=0.5,
            edgecolors='crimson',
            facecolors='none')


unmapped_weights = torch.cat([tensor.reshape(-1).detach() for tensor in model.cb.tensors], axis=0)
ax4.hist(unmapped_weights.numpy().reshape(-1), bins=20, color='pink')

left_mapped_weights = torch.cat([model.cb.W[m[0]:m[0]+m[2], m[1]:m[1]+m[3]:2].reshape(-1).detach() for m in model.cb.mapped], axis=0).numpy().reshape(-1, 1)
right_mapped_weights = torch.cat([model.cb.W[m[0]+1:m[0]+m[2]+1, m[1]+1:m[1]+m[3]+1:2].reshape(-1).detach() for m in model.cb.mapped], axis=0).numpy().reshape(-1,1)
N, bins, patches = ax5.hist(np.concatenate((left_mapped_weights, right_mapped_weights), axis=1), stacked=True, bins=20, label=('left weights', 'right weights'))


for patch in patches[0]:
    patch.set_facecolor((230 / 255, 151 / 255, 151 / 255))

for patch in patches[1]:
    patch.set_facecolor('turquoise')

weights = [model.cb.W[coord[0]:coord[0]+coord[2], coord[1]*2:coord[1]*2+coord[3]*2] for coord in model.cb.mapped] + [model.cb.W]
vmax = max(torch.max(weight) for weight in weights)
vmin = min(torch.min(weight) for weight in weights)
for i, weight in enumerate(weights):
    sns.heatmap(weight.detach(), vmax=vmax, vmin=vmin, cmap=cmap, square=True, cbar=False if i!=len(weights)-1 else True, ax=ax_cmap[i])


print("Generating Training Graph")
losses = [] # make range of colors
for i in range(30):
    print("Model", i, "| elapsed time:", "{:5.2f}".format((time.time() - start_time) / 60), "min")
    model = rnn_ode.RNN_ODE(1, 8, 1, device_params, time_steps)
    losses.append(torch.cat([item.detach().view(-1) for item in train.train(train_data, model, epochs)]).view(-1).numpy())
    print(losses[-1])
    
for loss in losses:
    ax3.plot(list(range(epochs)),
         loss,
         linewidth=0.5,
         color='lightgrey')


for j in range(1):
    model = rnn_ode.RNN_ODE(1, 8, 1, device_params, time_steps)
    model.use_cb(True)
    losses_real = train.train(train_data, model, epochs)
    ax3.plot(list(range(epochs)),
         losses_real,
         linewidth=1.2,
         color='magenta')
    model.use_cb(False)
    
ax1.plot(x.squeeze()[:cutoff+num_predict+tw], y.squeeze()[:cutoff+num_predict+tw], linewidth=0.5, color='k', linestyle='dashed')
ax1.axvline(x=float(x.squeeze()[cutoff + tw - 1]), color='k')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.set_ylabel('real space', fontsize=16)
ax1.legend(('predictions', 'interpolation', 'data'), loc='lower left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.set_xlabel('t', fontsize=16)
ax2.set_ylabel('Norm of hidden state', fontsize=16)

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax3.set_xlabel('Epoch', fontsize=16)
ax3.set_ylabel('RMS Prediction Accuracy', fontsize=16)

ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.set_xlabel('Neural Network Weights', fontsize=16)

ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.set_xlabel('Mapped Conductances', fontsize=16)
ax5.legend(prop={'size': 16})

if save is True:
    fig1.savefig('output/fig5/1.png', dpi=600, transparent=True)
    fig3.savefig('output/fig5/2.png', dpi=600, transparent=True)
    fig4.savefig('output/fig5/3.png', dpi=600, transparent=True)
    fig5.savefig('output/fig5/4.png', dpi=600, transparent=True)

plt.show()

"""
plt.plot(model.node_rnn.observer.history[1], model.node_rnn.observer.history[0], color='k')
plt.show()

"""


# 230, 151, 151
