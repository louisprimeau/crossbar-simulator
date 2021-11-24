"""
crossbar.py
Louis Primeau
University of Toronto Department of Electrical and Computer Engineering
louis.primeau@mail.utoronto.ca
July 29th 2020
Last updated: May 22nd 2021
"""

"""
Circuit Solver taken from:

A Comprehensive Crossbar Array Model With Solutions for Line Resistance and Nonlinear Device Characteristics
An Chen
IEEE TRANSACTIONS ON ELECTRON DEVICES, VOL. 60, NO. 4, APRIL 2013
"""

import torch
import numpy as np
import itertools
import time
import warnings

from .ticket import ticket

class crossbar:
    def __init__(self, device_params, deterministic=False):

        # Useful for debugging
        self.deterministic = deterministic

        # Power Supply Voltage
        self.V = device_params["Vdd"]

        # DAC resolution
        self.input_resolution = device_params["dac_resolution"]
        self.output_resolution = device_params["adc_resolution"]

        # Wordline Resistance 
        self.r_wl = torch.Tensor((device_params["r_wl"],))

        # Bitline Resistance
        self.r_bl = torch.Tensor((device_params["r_bl"],))

        # Number of rows, columns
        self.size = device_params["m"], device_params["n"]

        # Crossbar conductance model
        self.method = device_params["method"]

        # Device Programming Error approximation
        # 'linear' programming assumes that there is are 2^resolution states between the on and off resistances of the device.

        self.r_on = device_params["r_on"]
        self.r_off = device_params["r_off"]
        
        # Any number programmed onto the crossbar is rounded to one of those states.
        if (self.method == "linear"):

            if self.deterministic:
                self.g_on = torch.ones(self.size) / device_params["r_on"]
                self.g_off =  torch.ones(self.size) / device_params["r_off"]
            else:
                self.g_on = 1 / torch.normal(device_params["r_on"], device_params["r_on_stddev"], size=self.size)
                self.g_off = 1 / torch.normal(device_params["r_off"], device_params["r_off_stddev"], size=self.size)

            # Resolution
            self.resolution = device_params["device_resolution"]
            self.conductance_states = torch.cat([torch.cat([torch.linspace(self.g_off[i,j], self.g_on[i,j],2**self.resolution - 1).unsqueeze(0)
                                                        for j in range(self.size[1])],dim=0).unsqueeze(0)
                                             for i in range(self.size[0])],dim=0)

        # 'viability' assumes ideal programming to any conductance but the end result is perturbed by gaussian noise with spread
        # equal to some percentage (the "viability") of the conductance.
        elif self.method == "viability":

            
            self.g_on = torch.ones(self.size) / device_params["r_on"]
            self.g_off =  torch.ones(self.size) / device_params["r_off"]
            self.viability = device_params["viability"]

            #self.g_on = 1 / torch.normal(device_params["r_on"], device_params["r_on_stddev"], size=self.size)
            #self.g_off = 1 / torch.normal(device_params["r_off"], device_params["r_off_stddev"], size=self.size) 
            #self.viability = device_params["viability"]
            
        else:
            raise ValueError("device_params['method'] must be \"linear\" or \"viability\"")

        # conductance of the word and bit lines. 
        self.g_wl = torch.Tensor((1 / device_params["r_wl"],))
        self.g_bl = torch.Tensor((1 / device_params["r_bl"],))
                
        # Bias Scheme
        self.bias_voltage = self.V * device_params["bias_scheme"]
        
        # Tile size (1x1 = 1T1R, nxm = passive, etc.)
        self.tile_rows = device_params["tile_rows"]
        self.tile_cols = device_params["tile_cols"]
        assert self.size[0] % self.tile_rows == 0, "tile size does not divide crossbar size in row direction"
        assert self.size[1] % self.tile_cols == 0, "tile size does not divide crossbar size in col direction"
        
        # Resistance of CMOS lines (NOT IMPLEMENTED)
        self.r_cmos_line = device_params["r_cmos_line"]

        # WL & BL resistances
        self.r_in = device_params["r_in"]
        self.r_out = device_params["r_out"]
        
        self.g_s_wl_in = torch.ones(self.tile_rows) / self.r_in
        self.g_s_wl_out = torch.ones(self.tile_rows) * 1e-15 # floating
        self.g_s_bl_in = torch.ones(self.tile_rows) * 1e-15 # floating
        self.g_s_bl_out = torch.ones(self.tile_rows) / self.r_out

        # WL & BL voltages that are not the signal, assume bl_in, wl_out are tied low and bl_out is tied to 1 V. 
        self.v_bl_in = torch.zeros(self.size[1])
        self.v_bl_out = torch.zeros(self.size[1])
        self.v_wl_out = torch.zeros(self.size[0])

        
        E_B = torch.cat([torch.cat(((-self.v_bl_in[i] * self.g_s_bl_in[i]).view(1), torch.zeros(self.tile_cols-2), (-self.v_bl_in[i] * self.g_s_bl_out[i]).view(1))).unsqueeze(1) for i in range(self.tile_rows)])
        E_W = torch.cat([torch.cat(((torch.zeros(1) * self.g_s_wl_in[i]).view(1), torch.zeros(self.tile_cols-2), (self.v_wl_out[i].view(1) * self.g_s_wl_out[i]).view(1))) for i in range(self.tile_rows)]).unsqueeze(1)
        self.E = torch.cat((E_B, E_W))

        # Conductance Matrix; initialize each memristor at the on resstance
        self.W = torch.ones(self.size) * self.g_on + torch.randn(self.size) * self.g_on * 0.1

        # Stuck-on & stuck-on device nonideality 
        self.p_stuck_on = device_params["p_stuck_on"]
        self.p_stuck_off = device_params["p_stuck_off"]
        state_dist = torch.distributions.categorical.Categorical(probs=torch.Tensor([self.p_stuck_on, self.p_stuck_off, 1 - self.p_stuck_on - self.p_stuck_off]))
        self.state_mask = state_dist.sample(self.size)

        # Storage for all mapped tensors and their positions. Used to get data off the crossbar after simulation. 
        self.mapped = []
        self.tensors = [] #original data of all mapped weights
        self.saved_tiles = {}
        self.current_history = []

        # NOT TESTED: GPU CAPABILITY
        # self.device = device_params["device"]
        self.write_voltage = 1
        self.write_pulse_period = 200e-9
        self.read_pulse_period = 1e-3

        self.calculate_power = True
        self.read_energy = 0.0
        self.write_energy = 0.0
        self.read_ops = 0
        self.write_ops = 0
        self.num_pulses = 100
        
    # Iterates through the tiles and solves each and then adds their outputs together. 
    def solve(self, voltage, return_power=False):

        P = 0.0
        
        p_tiles = self.programmed_tiles()
        output = torch.zeros((voltage.size(1), self.size[1]))
        for i, j in p_tiles:
            coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_rows))
            if str(coords) not in self.saved_tiles.keys():
                self.make_M(coords) # Lazy hash
        
        # This part would be super easy to parallelize.
        Es_all = [None] * (self.size[0] // self.tile_rows)
        for i in set(j for j, _ in p_tiles):
            Es_all[i] = self.make_Es(voltage[i*self.tile_rows:(i+1)*self.tile_rows,:])
            
        for i, j in p_tiles:
            coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_rows))    
            M = self.saved_tiles[str(coords)]
            V = torch.transpose(-torch.sub(*torch.chunk(torch.matmul(M, Es_all[i]), 2, dim=0)), 0, 1).view(-1, self.tile_rows, self.tile_cols)
            I_all = V * self.W[coords]
            if self.calculate_power: P += torch.sum(I_all * V)
            I = torch.sum(I_all, axis=1)
            output[:, j*self.tile_cols:(j+1)*self.tile_cols] += I 

        if self.calculate_power: self.read_energy += P * 1e-7
        self.current_history.append(output)

        self.read_ops += 1
        if return_power: return P
        return output

    # extremely lazy implementation of finding which tiles have been programmed. But crossbars aren't very large so whatever.
    def programmed_tiles(self):
        tile_coords = []
        for coords in self.mapped:
            for i in range(self.size[0] // self.tile_rows):
                for j in range(self.size[1] // self.tile_cols):
                    if i * self.tile_rows <= coords[0] + coords[2] and j * self.tile_cols <= 2*(coords[1] + coords[3]) and (i,j) not in tile_coords:
                        tile_coords.append((i,j))
        return tile_coords

    def make_Es(self, v_wl_ins):
        return self.E.repeat(1, v_wl_ins.size(1)).index_put((torch.arange(self.tile_rows) * self.tile_cols,), v_wl_ins * self.g_s_wl_in.unsqueeze(1).repeat(1, v_wl_ins.size(1)))
    
    # Constructs the M matrix in MV = E. 
    def make_M(self, coords):

        #print(self.W)
        
        g = self.W[coords]
        m, n = self.tile_rows, self.tile_cols

        def makea(i):
            return torch.diag(g[i,:]) \
                 + torch.diag(torch.cat((self.g_wl, self.g_wl * 2 * torch.ones(n-2), self.g_wl))) \
                 + torch.diag(self.g_wl * -1 * torch.ones(n-1), diagonal = 1) \
                 + torch.diag(self.g_wl * -1 * torch.ones(n-1), diagonal = -1) \
                 + torch.diag(torch.cat((self.g_s_wl_in[i].view(1), torch.zeros(n - 2), self.g_s_wl_out[i].view(1))))
                                   
        def makec(j):
            return torch.zeros(m, m*n).index_put((torch.arange(m), torch.arange(m) * n + j), g[:, j])
        
        def maked(j):
            d = torch.zeros(m, m*n)
            
            i = 0
            d[i, j] = -self.g_s_bl_in[j] - self.g_bl - g[i, j]
            d[i, n*(i+1) + j] = self.g_bl
            
            for i in range(1, m):
                d[i, n*(i-1) + j] = self.g_bl
                d[i, n*i + j] = -self.g_bl - g[i,j] - self.g_bl
                d[i,j] = self.g_bl
                   
            i = m - 1
            d[i, n*(i-1) + j] = self.g_bl
            d[i, n*i + j] = -self.g_s_bl_out[j] - g[i,j] - self.g_bl
                
            return d
        
        A = torch.block_diag(*tuple(makea(i) for i in range(m)))
        B = torch.block_diag(*tuple(-torch.diag(g[i,:]) for i in range(m)))
        C = torch.cat([makec(j) for j in range(n)],dim=0)
        D = torch.cat([maked(j) for j in range(0,n)], dim=0)

        M = torch.cat((torch.cat((A,B),dim=1), torch.cat((C,D),dim=1)), dim=0)

        #print(A, B, C, D, sep='\n')
        #print("M", M)
        M = torch.inverse(M)

        self.saved_tiles[str(coords)] = M

        return M

    # Handles programming for the crossbar instance. 
    def register_linear(self, matrix, bias=None, row=None, col=None, index=None, bypass=False):
        assert not ((row is not None) and (col is not None) and (index is not None)), "either specify (row and col) or (index) or (neither). cannot specify both"
        
        with torch.no_grad():

            assert len(self.tensors) == len(self.mapped), "tensors and indices out of sync"

            if (row is None and col is None and index is None):

                index = len(self.tensors)
                self.tensors.append(matrix)
                row, col = self.find_space(matrix.size(0), matrix.size(1)*2)
                
            elif (row is None and col is None and index is not None):

                self.tensors[index] = matrix
                row, col = self.mapped[index][:2]
                
            elif (row is not None and col is not None and index is None):
                
                warnings.warn("Specifying row and column manually may result in collisions")
                index = len(self.tensors)
                self.tensors.append(matrix)
                self.mapped.append((row, col, matrix.size(0), matrix.size(1)))
                
            else:
                raise ValueError("You have encountered an edge case, please email louis")

            # Scale matrix                            
            if (self.method == "linear"):                
                mat_scale_factor = torch.max(torch.abs(matrix)) / torch.max(self.g_on) * 2
                scaled_matrix = matrix / mat_scale_factor
                midpoint = self.conductance_states.size(2) // 2
                for i in range(row, row + scaled_matrix.size(0)):
                    for j in range(col, col + scaled_matrix.size(1)):
                        shifted = self.conductance_states[i,j] - self.conductance_states[i,j,midpoint]
                        idx = torch.min(torch.abs(shifted - scaled_matrix[i-row,j-col]), dim=0)[1]
                        self.W[i,2*j+1] = self.conductance_states[i,j,idx]
                        self.W[i,2*j] = self.conductance_states[i,j,midpoint-(idx-midpoint)]

            elif (self.method == "viability"):
                new_W = self.W.clone().detach()
                mat_scale_factor = torch.max(torch.abs(matrix)) / (torch.max(self.g_on) - torch.min(self.g_off)) * 3
                if mat_scale_factor == 0.0: mat_scale_factor = 1.0
                scaled_matrix = matrix / mat_scale_factor
                midpoint = 5.5e-5
                right_state, left_state = midpoint + scaled_matrix / 2,  midpoint - scaled_matrix / 2
                right_state = right_state + torch.randn(right_state.size()) * right_state * self.viability
                left_state = left_state + torch.randn(left_state.size()) * left_state * self.viability
                out = torch.stack((left_state, right_state), dim=2).view(right_state.size(0), right_state.size(1) * 2)

                if bypass:
                    out = torch.ones_like(out) / self.r_off
                    out = out + torch.randn(out.size()) * 0.25 * out

                new_W[row:row+scaled_matrix.size(0), col:col + scaled_matrix.size(1) * 2] = out
                self.W = new_W

                if self.calculate_power:
                    Ns = torch.round((out - 1/self.r_off) * self.num_pulses * self.r_on)
                    if bypass: Ns = self.num_pulses * torch.ones_like(Ns)
                    write_energy_out = torch.sum(self.write_voltage**2 * Ns * out * self.write_pulse_period)
                    read_pulse_power_row = torch.zeros(scaled_matrix.size(0))
                    for i in range(scaled_matrix.size(0)):
                        read_voltage = torch.Tensor([0] * (row + i) + [self.V] + [0] * (self.size[0] - row - i - 1)).unsqueeze(1)
                        read_pulse_power_row[i] = self.solve(read_voltage, return_power=True)
                    verification_energy_out = torch.sum(read_pulse_power_row * torch.sum(Ns, dim=1)) * self.read_pulse_period
                    write_energy_out = write_energy_out + verification_energy_out
                    self.write_energy += write_energy_out
                    

               
            if not self.deterministic: self.apply_stuck()

        self.write_ops += 1
        self.saved_tiles = {}
        return ticket(index, row, col // 2, matrix.size(0), matrix.size(1), matrix, mat_scale_factor, self)
    
    def clip(self, tensor, i, j):
        assert self.g_off[i, j] < self.g_on[i, j]
        return torch.clip(tensor, min=self.g_off[i, j], max=self.g_on[i, j])

    def apply_stuck(self):
        self.W[self.state_mask == 0] = self.g_off[self.state_mask==0]
        self.W[self.state_mask == 1] = self.g_on[self.state_mask==1]

    def which_tiles(self, row, col, m_row, m_col):
        return itertools.product(range(row // self.tile_rows, (row + m_row) // self.tile_rows + 1),
                                 range(col // self.tile_cols,(col + m_col) // self.tile_cols + 1),
        )

    def find_space(self, m_row, m_col):
        print(m_row, m_col)
        if m_row > self.size[0] or m_col > self.size[1]:
                raise ValueError("Matrix with size ({}, {}) is too large for crossbar of size ({}, {})".format(m_row, m_col, self.size[0], self.size[1]))
            
        # Format is (*indexes of top left corner, *indexes of bottom right corner + 1 (it's zero indexed))
        if not self.mapped:
            self.mapped.append((0,0,m_row,m_col))
        else:
            if self.mapped[-1][3] + m_col < self.size[1]:
                self.mapped.append((self.mapped[-1][0], self.mapped[-1][1] + self.mapped[-1][3], m_row, m_col))
            else:
                if m_row > (self.size[0] - self.mapped[-1][2]):
                    print(m_col, self.size[0], self.mapped[-1][2])
                    raise ValueError("Matrix with {} rows does not fit on crossbar with {} free rows".format(m_col, self.size[0] - self.mapped[-1][2]))
                    
                self.mapped.append((self.mapped[-1][2], 0, m_row, m_col))
                
        #self.mapped.append((self.mapped[-1][0] + self.mapped[-1][2], self.mapped[-1][1] + self.mapped[-1][3], m_row, m_col))
        return self.mapped[-1][0], self.mapped[-1][1] 
    
    def clear(self):
        self.mapped = []
        self.tensors = []
        self.saved_tiles = {}
        self.W = torch.ones(self.size) * self.g_on
