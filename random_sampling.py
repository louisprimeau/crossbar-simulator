import torch
from sim.crossbar.crossbar import crossbar
import sim.modules.Linear as linear
from sim.modules.Random import Random, Random2

import matplotlib
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.linalg
from scipy import stats



torch.set_printoptions(precision=3)



#mat = torch.randint(-3, 3, (n,n), dtype=torch.float)
#cov = mat.T @ mat + torch.eye(n)
#S_invhalf = torch.from_numpy(scipy.linalg.sqrtm(cov.inverse().numpy()).real)
"""
num_resistances, num_viabilities = 5, 5
all_data = np.zeros((num_resistances, num_viabilities))

for i in range(0, num_resistances):
    for j in range(0, num_viabilities):
        line_resistance = i
        viability = i / 100
        
        device_params = {"Vdd": 1.8,
                         "r_wl": line_resistance + 1e-9,
                         "r_bl": line_resistance + 1e-9,
                         "m": 16,
                         "n": 16,
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
                         "viability": viability,
        }

        cb = crossbar(device_params, deterministic=False)

        error = 0.0
        n, m, p = 10, 10, 10
        for k in range(n):
            rand_matrix = torch.randint(0, 1, (8, 8), dtype=torch.float) + torch.eye(8)
            mult = linear.Linear(8, 8, cb, W=rand_matrix)

            for l in range(m):
                test_vector = torch.randn(p, 8, 1)
                test_vector = test_vector / torch.norm(test_vector, dim=1).unsqueeze(1).repeat(1, 8, 1)

                mult.use_cb(True)
                cb_output = mult(test_vector)

                mult.use_cb(False)
                id_output = mult(test_vector)

                error += torch.sum(torch.norm(id_output - cb_output, dim=1)).item()

                mult.remap()

        error /= n*m*p
        all_data[i, j] = error
        
        print(i, j, error)
    

np.savetxt("grid_search.csv", all_data, delimiter=",")
"""

#fig, ax = plt.subplots(2)

for i in [4]:
    device_params = {"Vdd": 1.8,
                     "r_wl": i + 1e-9,
                     "r_bl": i + 1e-9,
                     "m": 8,
                     "n": 8,
                     "r_on": 1e4,
                     "r_off": 1e5,
                     "r_in": 1e9,
                     "r_out": 1,
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
                     "viability": 0.05,
    }

    cb = crossbar(device_params, deterministic=False)
    sampler = Random2(4, cb)

    sampler.use_cb(True)
    N = 1000

    all_samples = []
    for _ in range(N): all_samples.append(sampler().squeeze().unsqueeze(1))
    all_samples = torch.cat(all_samples, axis=1) #n x N

    print(torch.mean(all_samples, dim=1))
    print(torch.std(all_samples, dim=1))
    #vals, bins = np.histogram(all_samples[-1].view(-1).numpy(), bins=70, density=True)
    #ax[1].plot((bins[1:] + bins[:-1])/2, vals, label="{} Ω, RV 4".format(i))

    #vals, bins = np.histogram(all_samples[0].view(-1).numpy(), bins=70, density=True)
    #ax[0].plot((bins[1:] + bins[:-1])/2, vals, label="{} Ω, RV 1".format(i))


#ax[0].set_xlim(-3,3)
#ax[1].set_xlim(-3,3)

#ax[1].legend()
#ax[0].legend()
#plt.show()

#sample_mean = torch.mean(all_samples, axis=1)
#centered = (all_samples.T - sample_mean).T
#sample_cov = centered @ centered.T / N
#sample_S_invhalf = torch.from_numpy(scipy.linalg.sqrtm(sample_cov.inverse().numpy()).real)
#transformed = sample_S_invhalf @ centered # Should come from Z(0, 1)
