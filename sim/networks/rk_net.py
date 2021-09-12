import torch
from ..crossbar import crossbar
from . import Linear, Conv, Add
from . import util
        
class RK_net(torch.nn.Module):

    def __init__(self, in_channels, cb):
        super(RK_net, self).__init__()

        self.in_channels = in_channels
        self.cb = cb

        self.N, self.t0, self.t1 = 3, 0, 1
        self.h = (self.t1 - self.t0) / self.N

        self.tableau = ((1/2,),
                        (0, 1/2),
                        (0, 0, 1),
                        (1/6, 1/3, 1/3, 1/6),
                       )
        
        self.add = torch.nn.ModuleList([Add.Add(torch.cat((torch.ones(1), torch.Tensor(weight)*self.h)), cb) for weight in self.tableau])
        
        self.network = torch.nn.Sequential(Conv.Conv2d(in_channels, in_channels, 3, cb),
                                           torch.nn.ReLU(),
                                           Conv.Conv2d(in_channels, in_channels, 3, cb),
                                           )
        
        self.relu_out = torch.nn.ReLU()
        
        self.cbon = False        
        
    def forward(self, x):
        
        for _ in range(self.N):

            k1 = self.network(x)
            k2 = self.network(self.add[0](x, k1))
            k3 = self.network(self.add[1](x, k1, k2))
            k4 = self.network(self.add[2](x, k1, k2, k3))
            x = self.add[3](x, k1, k2, k3, k4)

        return self.relu_out(x)


    def remap(self):
        self.network[0].remap()
        self.network[2].remap()

    def use_cb(self, state):
        self.network[0].use_cb(state)
        self.network[2].use_cb(state)
        for adder in self.add:
            adder.use_cb(state)
        
