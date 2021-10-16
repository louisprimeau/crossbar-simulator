import torch
from ..crossbar import crossbar
from . import Linear, ODE_net
from . import observer
        
class NODERNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, cb, N):
        super(NODERNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.observer = observer.observer()

        #self.linear_in = Linear.Linear(input_size, hidden_layer_size, cb)
        #self.linear_hidden = Linear.Linear(hidden_layer_size, hidden_layer_size, cb)
        self.linear = Linear.Linear(hidden_layer_size + input_size, hidden_layer_size, cb)

        self.solve = ODE_net.eulerforward(hidden_layer_size, N, cb, self.observer)
        self.nonlinear = torch.nn.Tanh()

    # Taking a sequence, this predicts the next N points, where 
    def forward(self, x, t):

        h_i = torch.zeros(self.hidden_layer_size, 1)
        for i, x_i in enumerate(x):
            if i == (len(x) - 1) and self.observer.on == True: self.solve.observer_flag = True

            #print("STARTING ODE SOLVER")
            
            h_i = self.solve(h_i, t[i-1] if i>0 else t[i], t[i])
            self.solve.observer_flag = False

            #print("STARTING RNN LAYER")
            
            if i == (len(x) - 1): self.observer.append(h_i.view(1, -1), t[i])
            h_i = self.nonlinear(self.linear(torch.cat((h_i, x_i), axis=0)))
            
            if i == (len(x) - 1): self.observer.append(h_i.view(1, -1), t[i])
        return h_i 
    
    def remap(self):
        #self.linear_in.remap()
        #self.linear_hidden.remap()
        self.linear.remap()
        self.solve.remap()
    
    def use_cb(self, state):
        #self.linear_in.use_cb(state)
        #self.linear_hidden.use_cb(state)
        self.linear.use_cb(state)
        self.solve.use_cb(state)

    # vestigial function
    def observe(self, state):
        self.observer.on = state
        
        #self.solve.observe(state)
