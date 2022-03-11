import math
import torch
from . import util
class adc:

    def __init__(self, crossbar, resolution):

        self.resolution = resolution

        crossbar.find_space(2, 2)

        crossbar.W = torch.clone(crossbar.g_off)
        crossbar.W[:,0] = crossbar.g_on[:, 0]
        vector = torch.ones(crossbar.size[0], 1) * crossbar.V
        max_current = torch.max(crossbar.solve(vector))

        crossbar.W = torch.clone(crossbar.g_off)
        vector = torch.ones(1, 1)
        Vs, _, _ = crossbar.make_V(vector, 1, 0, 1)
        min_current = torch.min(crossbar.solve(Vs))

        self.levels = torch.linspace(min_current, max_current, 2**self.resolution)

        crossbar.mapped =  crossbar.mapped[:-1]

    def read_currents(self, I):
        out = util.round_to(I, self.levels)
        return out / (2**self.resolution - 1)
