import torch
from crossbar import system, params
from ops_stateful import gemv

dp_variability ={  

    # Physical Parameters
                "Vdd": 1.0,
                "r_wl": 20,
                "r_bl": 20,
                "r_on": 1e3,
                "r_off": 1e5,
                "r_in": 1e3,
                "r_out": 1e3,

    # Device Parameters
                "dac_resolution": 8,
                "adc_resolution": 8,
                "device_resolution": 8,
                "bias_scheme": 1/3,

    # Crossbar Parameters
                "m": 3,
                "n": 3,
                "tile_rows": 3,
                "tile_cols": 3,

    # Variability
                "r_on_stddev": 1e3,
                "r_off_stddev": 1e4,
                "p_stuck_on": 0.0,
                "p_stuck_off": 0.0,
                "method": "viability",
                "viability": 0.0,
                }

#torch.set_printoptions(linewidth=160)
sys = system.System(8, dp_variability)

N = 2
A = torch.Tensor([[4, 2, 4],
                  [2, 7, 3],
                  [4, 2, 4]])

v = torch.Tensor([1, 2, 3]).reshape(3, 1)

interface = sys.program(A)
out = interface.vmm(v)
print(out)
print(A.T.mm(v))
