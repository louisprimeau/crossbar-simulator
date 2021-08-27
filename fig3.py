import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import sim.networks.Conv as conv
import sim.networks.Linear as linear
import sim.networks.ef_net as ef_net
import sim.crossbar.crossbar as crossbar

import seaborn as sns
import matplotlib.pyplot as plt
transform = transforms.Compose([transforms.Resize((8,8)),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,)),
])

trainset = datasets.MNIST('~/mnist/', download=False, train=True, transform=transform)
valset = datasets.MNIST('~/mnist/', download=False, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 64,
                 "n": 128,
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

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.cb = crossbar.crossbar(device_params, deterministic=False)
        self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=2, padding=1)
        self.ode_ef_block = ef_net.simple_EF_block(3, 3, self.cb)
        self.linear = linear.Linear(4 * 4 * 3, 40, self.cb)
        self.linear2 = linear.Linear(40, 10, self.cb)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.ode_ef_block(x)

        x = F.relu(self.linear(torch.flatten(x, 1).unsqueeze(2)).squeeze(2))
        x = self.linear2(torch.flatten(x, 1).unsqueeze(2)).squeeze(2)
        return x
    
def train(network, epochs):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    losses, accuracies = [], []
    for epoch in range(1, epochs+1):
        print("Epoch", epoch)
        num_correct = 0
        epoch_loss, num_batches = 0, 0
        for i, (example, label) in enumerate(trainloader, 0):
            network.zero_grad()
            #if i == 100: break
        
            out = network(example) 
            loss = criterion(out, label)
            loss.backward()

            #print(network.ode_ef_block.kernel2.grad)
            #print(network.ode_ef_block.conv1.conv_weight.grad)

            network.ode_ef_block.remap()
            optimizer.step()
            
            epoch_loss += loss
            num_batches += 1
            num_correct += torch.sum(torch.argmax(out, 1) == label)

            
        losses.append(epoch_loss / num_batches)
        print("Train Score: {:.2f}%, ({} / {})".format(num_correct / i / 64 * 100, num_correct, i*64))

        num_correct = 0
        for example, label in valloader:
            out = torch.argmax(network(example), 1)
            num_correct += torch.sum(out == label)

        print("Validation Score: {:.2f}%, ({} / {})".format(num_correct / len(valloader) / 64 * 100, num_correct, len(valloader)*64))
        accuracies.append(num_correct / len(valloader) / 64)
        
    return network, losses, accuracies


fig5, ax_cmap = plt.subplots(ncols=5, figsize=(20, 3))
cmap = sns.blend_palette(("#fa7de3", "#ffffff", "#6ef3ff"), n_colors=9, as_cmap=True, input='hex')

for ax in ax_cmap:
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])

model = net()

coord = model.ode_ef_block.cb.mapped[model.ode_ef_block.kernel_and_id.ticket.index]

weights = [model.cb.W[coord[0]:coord[0]+coord[2], coord[1]*2:coord[1]*2+coord[3]*2] for coord in model.cb.mapped] + [model.cb.W]
vmax = max(torch.max(weight) for weight in weights)
vmin = min(torch.min(weight) for weight in weights)

for i, weight in enumerate(weights):
    sns.heatmap(weight.detach(), vmax=vmax, vmin=vmin, cmap=cmap, square=True, cbar=False if i!=len(weights)-1 else True, ax=ax_cmap[i])

test_network_1 = net()
test_network_2 = net()
#test_network.conv.use_cb(True)
#test_network.linear.use_cb(True)
#test_network.conv.use_cb(False)
#test_network.linear.use_cb(False)

epochs = 30
test_network_1, losses_1, accuracies_1 = train(test_network_1, epochs)
test_network_2, losses_2, accuracies_2 = train(test_network_2, epochs)

fig1, ax1 = plt.subplots(nrows=1)

ax1.plot(list(range(epochs)),
         losses_1,     
         'o-',
         linewidth=0.5,
         color='deepskyblue',
         markerfacecolor='none',
         )

ax1.plot(list(range(epochs)),
         losses_2,     
         'o-',
         linewidth=0.5,
         color='crimson',
         markerfacecolor='none',
         )

ax1.spines['top'].set_visible(False)
ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('Cross Entropy Loss', fontsize=16)
ax1.legend(('NODE', 'Conventional'), loc='top right')

plt.show()
