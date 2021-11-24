import torch
from ..crossbar import crossbar
from . import Linear
import numpy as np
class Random(torch.nn.Module):
    def __init__(self, size, cb, cov=None):

        super(Random, self).__init__()
        self.cb = cb
        self.target_conductance = (1 / cb.r_on - 1 / cb.r_off) / 2
        self.size = size
        self.cbon = False
        
        self.rand_ticket = cb.register_direct(torch.ones(1, size) * self.target_conductance)
        
        self.cov = cov
        if cov is not None:
            L = torch.linalg.cholesky(cov).type(torch.float)
            self.L = Linear.Linear(size, size, self.cb, W=L, bias=False)

            
    def compute_norm(self, N, dim=(0, 1)):
        ac = []
        for _ in range(N):
            self.rand_ticket.remap(torch.ones(1, self.size) * self.target_conductance)
            ac.append(self.rand_ticket.apply_voltage(torch.ones(1)))
        ac = torch.cat(ac)
        return torch.mean(ac, dim=dim), torch.std(ac, dim=dim)
        
    def forward(self):
        self.rand_ticket.remap(torch.ones(1, self.size) * self.target_conductance)
        
        if self.cbon: r_vect = (self.rand_ticket.apply_voltage(torch.ones(1)) - self.mean) / self.std
        else: r_vect = torch.randn(self.size, 1)

        if self.cov is None: return r_vect
        else: return self.L(r_vect.view(1, -1, 1))

    def remap(self, W=None):
        self.rand_ticket.remap(torch.ones(1, self.size) * self.target_conductance)
    
    def use_cb(self, state):
        self.cbon = state
        if self.cov is not None: self.L.use_cb(state)

class Random2(torch.nn.Module):
    def __init__(self, size, cb, cov=None, bypass=False):
        super(Random2, self).__init__()
        self.cb = cb
        self.size = size
        self.cbon = False

        self.rand_source = Linear.Linear(1, size, cb, W=torch.zeros(size, 1), vbits=1, bypass=bypass)
        self.cov_correction = None
        #self.mean, self.cov_correction = self.tune()
        #self.cov_correction = Linear.Linear(self.size, self.size, self.cb, W=self.cov_correction, vbits=8)

        self.mean, self.std = self.compute_norm(100)
        
    def tune(self):
        mean, std = self.compute_norm(1000, dim=0)
        print("mean, std:", mean, std)
        return mean, torch.diag(1 / std.view(-1))

            
    def compute_norm(self, N):
        self.rand_source.use_cb(True)
        ac = []
        for _ in range(N):
            if _ % 10 == 0: print("norm", _)
            self.rand_source.remap(W=torch.zeros(self.size, 1))
            out = self.rand_source(torch.ones(1, 1, 1))
            ac.append(out)
            
        ac = torch.cat(ac)
        self.rand_source.use_cb(self.cbon)
        return torch.mean(ac), torch.std(ac)

    def forward(self):
        self.rand_source.remap(W=torch.zeros(self.size, 1))
        if self.cbon:

            out = self.rand_source(torch.ones(1, 1, 1))
            #out = out - self.mean
            #out = self.cov_correction(out.view(1, -1, 1))
            #r_vect = (out - self.mean2) / self.std2
            r_vect = (out - self.mean)/self.std
            
            #r_vect = (self.rand_source(torch.ones(1, 1, 1)).view(-1, 1) - self.mean) / self.std
        else: r_vect = torch.randn(self.size, 1)

        return r_vect
        #if self.cov is None: return r_vect
        #else: return self.L(r_vect.view(1, -1, 1))

    def remap(self, W=None):
        self.rand_source.remap(W)

    def use_cb(self, state):
        self.cbon = state
        #self.cov_correction.use_cb(state)
        self.rand_source.use_cb(state)
        #if self.cov is not None: self.L.use_cb(state)
