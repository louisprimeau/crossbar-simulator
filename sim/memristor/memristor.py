import torch

class VTEAM():

    def __init__(self, params):

        self.k_off = params['k_off']
        self.a_off = params['a_off']
        self.v_off = params['v_off']
        self.r_off = params['r_off']
        self.w_off = params['w_off']

        self.k_on = params['k_on']
        self.a_on = params['a_on']
        self.v_on = params['v_on']
        self.r_on = params['r_on']
        self.w_on = params['w_on']

        self.w_off_n = 10.0
        self.w_on_n = 50.0
        self.w_stationary = 10.0
        
    def compute_dw(self, v, w):
        dw = torch.randn_like(v) * self.w_stationary
        high_v_mask = v > self.v_off
        low_v_mask = v < self.v_on
        dw[high_v_mask] = self.k_off * ((v[high_v_mask] / self.v_off) - 1)**self.a_off * self.window(w[high_v_mask]) + torch.randn_like(dw[high_v_mask]) * self.w_off_n
        dw[low_v_mask] = -self.k_on * ((v[low_v_mask] / self.v_on) - 1)**self.a_on * self.window(w[low_v_mask]) + torch.randn_like(dw[low_v_mask]) * self.w_on_n
        return dw

    def euler_step(self, v, w, dt):
        w = w + self.compute_dw(v, w) * dt
        return torch.clamp(w, self.w_on, self.w_off)

    def conductance(self, w):
        scale = (self.r_off - self.r_on) / (self.w_off - self.w_on)
        return 1 / (self.r_on + (w - self.w_on) * scale)

    def current(self, v, w):
        return self.conductance(w) * v

    def window(self, x):
        f = torch.ones_like(x)
        f[x > self.w_off] = 0
        f[x < self.w_on] = 0
        return f