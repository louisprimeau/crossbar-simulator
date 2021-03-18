



import torch
import crossbar_viability as crossbar

# This testbench tries out a 100 matrices and vectors to multiply.

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
                 "p_stuck_on": 0.0,
                 "p_stuck_off": 0.0,
                 "method": 'viability',
                 "viability": 0.0,
}

cb = crossbar.crossbar(device_params)

torch.manual_seed(2)

torch.backends.cudnn.deterministic=True

max_rows = device_params["m"] // 2
max_cols = device_params["n"]

test_num = 20

matrices = [torch.randint(-20, 20, (max_rows, max_cols)) for _ in range(test_num)]
vectors = [torch.randint(-20, 20, (max_cols, 1)) for _ in range(test_num)]

for matrix, vector in zip(matrices, vectors):


    
    cb.clear()
    ticket = cb.register_linear(torch.transpose(matrix,0,1))
    output = ticket.vmm(vector, v_bits=4)
    target = matrix.matmul(vector)
    print(torch.norm(target - output) / torch.norm(matrix.double()))
