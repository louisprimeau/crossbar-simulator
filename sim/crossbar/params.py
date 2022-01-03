
dp_variability = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 16,
                 "n": 16,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "r_in": 1,
                 "r_out": 1e12,
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
}

dp_linear = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 16,
                 "n": 16,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 4,
                 "tile_cols": 4,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20, 
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "device_resolution": 4,
                 "method": "linear",
}
