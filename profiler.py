import torch
import sim.networks.rnn_ode as rnn_ode
import train

import time
import random
import numpy as np
pi = 3.14159265359

# DEVICE PARAMS for convenience.
device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 32,
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
                 "p_stuck_on": 0.0,
                 "p_stuck_off": 0.0,
                 "method": "viability",
                 "viability": 0.0001,
}

import numpy as np
import random
import os
deterministic = True
if deterministic:
    seed = 12
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# MAKE DATA
n_pts = 150
size = 1
tw = 10
cutoff = 50
#x = torch.rand(1, n_pts) * 24 * pi
#x torch.sort(x, axis=1)[0]
x = torch.linspace(0, 24*pi, n_pts).view(1, -1)
y = torch.sin(x) / 2 + 0.5# + torch.rand(x.size()) / 10
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)), (y[:, i+tw:i+tw+1].reshape(-1, size))) for i in range(y.size(1) - tw)] 
train_data, test_start = data[:cutoff], data[cutoff]

# TRAIN MODELS AND PLOT
time_steps = 50
epochs = 10

for j in range(1):
    model = rnn_ode.RNN_ODE(1, 4, 1, device_params, time_steps)
    model.use_cb(True)
    losses_real = train.train(train_data, model, epochs, verbose=True)

"""
model = rnn_ode.RNN_ODE(1, 4, 1, device_params, time_steps)
model.use_cb(True)
import torch.autograd.profiler as profiler
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        model(train_data[0][0])

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    
"""
