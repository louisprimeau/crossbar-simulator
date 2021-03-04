

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
import networks.rnn_ode as rnn_ode
import train

import time
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pi = 3.14159265359

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
cutoff = 50
#x = torch.rand(1, n_pts) * 24 * pi
#x = torch.sort(x, axis=1)[0]
x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)), (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)] 
train_data, test_start = data[:cutoff], data[cutoff]

# CONFIGURE PLOTS
fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

fig3, ax3 = plt.subplots()

fig4, (ax4, ax5) = plt.subplots(nrows=2, figsize=(8,6))

fig5, ax_cmap = plt.subplots(ncols=5, figsize=(20, 3))
cmap = sns.diverging_palette(250, 300, s=90, as_cmap=True)
for ax in ax_cmap:
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])


# TRAIN MODELS AND PLOT
time_steps = 50
epochs = 20
num_predict=30
start_time = time.time()
for i in range(1):
    print("Model", i, "| elapsed time:", "{:5.2f}".format((time.time() - start_time) / 60), "min")
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

    ax1.plot(torch.cat((x.view(-1)[cutoff + tw - 1].view(-1), times.view(-1)), axis=0),
             torch.cat((y.view(-1)[cutoff + tw - 1].view(-1), output.view(-1)), axis=0),
             'o-',
             linewidth=0.5,
             color='k',
             markerfacecolor='none',
             )

    H = model.node_rnn.observer.history[0].detach()
    t = model.node_rnn.observer.history[1].view(-1).detach()

    ax1.plot(t,
             model.linear(torch.transpose(H, 0, 1)).view(-1).detach(),
             ':',
             linewidth=0.5,
             color='k')


    ax2.plot(t,
             torch.linalg.norm(H, ord=2, dim=1).view(-1),
             ':',
             linewidth=0.5,
             color='k')

    ax2.scatter(t[::(time_steps + 2)],
                torch.linalg.norm(H, ord=2, dim=1)[::(time_steps + 2)],
                linewidth=0.5,
                edgecolors='k',
                facecolors='none')
    
    ax3.plot(list(range(epochs)),
             losses,
             linewidth=0.5,
             color='pink')


    unmapped_weights = torch.cat([tensor.reshape(-1).detach() for tensor in model.cb.tensors], axis=0)
    ax4.hist(unmapped_weights.numpy().reshape(-1), bins=20, color='pink')

    left_mapped_weights = torch.cat([model.cb.W[m[0]:m[0]+m[2], m[1]:m[1]+m[3]:2].reshape(-1).detach() for m in model.cb.mapped], axis=0).numpy().reshape(-1, 1)
    right_mapped_weights = torch.cat([model.cb.W[m[0]+1:m[0]+m[2]+1, m[1]+1:m[1]+m[3]+1:2].reshape(-1).detach() for m in model.cb.mapped], axis=0).numpy().reshape(-1,1)
    ax5.hist(np.concatenate((left_mapped_weights, right_mapped_weights), axis=1), stacked=True, bins=20)

    weights = [model.cb.W[coord[0]:coord[0]+coord[2], coord[1]*2:coord[1]*2+coord[3]*2] for coord in model.cb.mapped] + [model.cb.W]
    vmax = max(torch.max(weight) for weight in weights)
    vmin = min(torch.min(weight) for weight in weights)
    for i, weight in enumerate(weights):
        sns.heatmap(weight, vmax=vmax, vmin=vmin, cmap=cmap, square=True, cbar=False, ax=ax_cmap[i])
    
ax1.plot(x.squeeze()[:cutoff+num_predict+tw], y.squeeze()[:cutoff+num_predict+tw], linewidth=0.5, color='k', linestyle='dashed')
ax1.axvline(x=float(x.squeeze()[cutoff + tw - 1]), color='k')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.setp(ax1, ylabel='real space')
ax1.legend(('predictions', 'interpolation', 'data'), loc='lower left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.setp(ax2, xlabel='t')
plt.setp(ax2, ylabel='norm of hidden state')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.setp(ax3, xlabel='Epoch')
plt.setp(ax3, ylabel='RMS Prediction Accuracy')

ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plt.setp(ax4, xlabel='Unmapped Weights')

ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
plt.setp(ax5, xlabel='Mapped Weights')

fig1.savefig('output/fig5/1.png', dpi=600, transparent=True)
fig3.savefig('output/fig5/2.png', dpi=600, transparent=True)
fig4.savefig('output/fig5/3.png', dpi=600, transparent=True)
fig5.savefig('output/fig5/4.png', dpi=600, transparent=True)

plt.show()

"""
plt.plot(model.node_rnn.observer.history[1], model.node_rnn.observer.history[0], color='k')
plt.show()

"""
