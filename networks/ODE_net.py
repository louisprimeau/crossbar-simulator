import torch
import crossbar
import observer

class eulerforward(torch.nn.Module):
    def __init__(self, hidden_layer_size, N, cb, observer):
        super(eulerforward, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.N = N
        self.nonlinear = torch.nn.Tanh()
        self.observer = observer
        self.observer_flag = False
    def forward(self, x0, t0, t1):
        x, h = x0, (t1 - t0) / self.N
        for i in range(self.N):
            x = x + h * self.nonlinear(self.linear(x))
            if self.observer_flag: self.observer.append(x.view(1, -1), t0 + h*i)
        return x
    def remap(self):
        self.linear.remap()
       
    def use_cb(self, state):
        self.linear.use_cb(state)

    def observe(self, state):
        self.observer.on = state

