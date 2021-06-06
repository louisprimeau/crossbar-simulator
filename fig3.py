from sim.crossbar import crossbar
from sim.networks import Linear, Conv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets


device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 208,
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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((8,8)),
     transforms.Normalize((0.5), (0.5))]
)

trainset = datasets.MNIST('~/mnist/', download=True, train=True, transform=transform)
valset = datasets.MNIST('~/mnist/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)

cb = crossbar.crossbar(device_params)

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv = Conv.Conv(3, 1, 3, cb)
        self.linear = Linear.Linear(8*8*3, 10, cb)
        
    def forward(self, x):
        x1 = self.conv(x).reshape(-1,1)
        x2 = self.linear(x1).reshape(-1)
        print(torch.norm(x1), torch.norm(x2))
        return x2

def train(network):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    epochs = 3
    
    for epoch in range(1, epochs+1):
        running_loss = 0
        for i, example in enumerate(trainloader, 0):
            out = network(example[0])
            loss = criterion(out.unsqueeze(0), example[1])

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
        
test_network = net()
#test_network.conv.use_cb(True)
#test_network.linear.use_cb(True)
print(train(test_network))


