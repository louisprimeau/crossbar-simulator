from networks.crossbar import crossbar
from networks import Linear, Conv
import torch
from torchvision import transforms, datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((8,8)),
     transforms.Normalize((0.5), (0.5))])

trainset = datasets.MNIST('~/mnist/', download=True, train=True, transform=transform)
valset = datasets.MNIST('~/mnist/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)


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

cb = crossbar.crossbar(device_params)

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv = Conv.Conv(3, 1, 3, cb)
        self.linear = Linear.Linear(8*8*3, 10, cb)
        self.softmax = torch.nn.Softmax()
    def forward(self, x):
        return self.softmax(self.linear(self.conv(x).reshape(-1,1)).view(-1))
    
test_network = net()
print(test_network(trainset[0][0].unsqueeze(0)))
