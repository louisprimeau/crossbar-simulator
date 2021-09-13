import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import sim.networks.Conv as conv
import sim.networks.Linear as linear
import sim.networks.rk_net as rk_net
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
                 "m": 128,
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

class ode_net(torch.nn.Module):
    def __init__(self):
        super(ode_net, self).__init__()

        self.cb = crossbar.crossbar(device_params, deterministic=False)
        self.conv1 = conv.Conv2d(1, 3, 3, self.cb, stride=2, padding=1)
        self.ode_block = rk_net.RK_net(3, self.cb)
        self.linear = linear.Linear(4 * 4 * 3, 40, self.cb)
        self.linear2 = linear.Linear(40, 10, self.cb)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.ode_block(x)
        x = F.relu(self.linear(torch.flatten(x, 1).unsqueeze(2)).squeeze(2))
        x = self.linear2(torch.flatten(x, 1).unsqueeze(2)).squeeze(2)

        return x
    
    def remap(self):
        self.conv1.remap()
        self.ode_block.remap()


def train(network, epochs):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    losses, accuracies = [], []
    for epoch in range(1, epochs+1):
        print("Epoch", epoch)
        num_correct = 0
        epoch_loss, num_batches = 0, 0

        with torch.enable_grad():
        
            for i, (example, label) in enumerate(trainloader, 0):
                network.zero_grad()
                #if i == 100: break

                out = network(example) 
                loss = criterion(out, label)
                loss.backward()

                #print(network.ode_ef_block.kernel2.grad)
                #print(network.ode_ef_block.conv1.conv_weight.grad)

                network.remap()
                optimizer.step()

                epoch_loss += loss
                num_batches += 1
                num_correct += torch.sum(torch.argmax(out, 1) == label)

            
        losses.append((epoch_loss / num_batches).detach())
        print("Train Score: {:.2f}%, ({} / {})".format(num_correct / i / 64 * 100, num_correct, i*64))

        with torch.no_grad():
        
            num_correct = 0
            for example, label in valloader:
                out = torch.argmax(network(example), 1)
                num_correct += torch.sum(out == label)

        print("Validation Score: {:.2f}%, ({} / {})".format(num_correct / len(valloader) / 64 * 100, num_correct, len(valloader)*64))
        accuracies.append(num_correct / len(valloader) / 64)
        
    return network, losses, accuracies


network = ode_net()
network, losses, accuracies = train(network, 30)
