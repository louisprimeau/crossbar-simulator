import torch
import crossbar
import numpy as np
import random
import os
import time

# This testbench tries out a 100 matrices and vectors to multiply.

device_params = {"Vdd": 0.2,
                 "r_wl": 20.0,
                 "r_bl": 20.0,
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
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.05,
                 "device": torch.device("cpu")
}

cb = crossbar.crossbar(device_params)

seed = 12
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

max_rows = device_params["m"] // 2
max_cols = device_params["n"]

test_num = 1

matrices = [torch.randint(-10, 10, (max_rows, max_cols)) for _ in range(test_num)]
vectors = [torch.randint(-10, 10, (max_cols, 1)) for _ in range(test_num)]

cb_time, t_time, error = 0.0, 0.0, 0.0
for matrix, vector in zip(matrices, vectors):
    cb.clear()
    ticket = cb.register_linear(torch.transpose(matrix,0,1))
    
    start_time = time.time()
    output = ticket.vmm(vector, v_bits=4)
    cb_time += time.time() - start_time

    start_time = time.time()
    target = matrix.matmul(vector)
    t_time += time.time() - start_time

    error += torch.norm(target - output) / torch.norm(matrix.double())

#current_history = torch.cat(current_history, axis=1)
print("Average crossbar vmm time:", cb_time / test_num, "s")
print("Average torch vmm time:", t_time / test_num, "s")
print("Average relative error:", error / test_num)
