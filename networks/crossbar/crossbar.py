"""
crossbar.py
Louis Primeau
University of Toronto Department of Electrical and Computer Engineering
louis.primeau@mail.utoronto.ca
July 29th 2020
Last updated: March 21st 2021
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

from ticket import ticket

class crossbar:
    def __init__(self, device_params):

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
        # Any number programmed onto the crossbar is rounded to one of those states.
        if (self.method == "linear"):
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
            self.g_on = 1 / torch.normal(device_params["r_on"], device_params["r_on_stddev"], size=self.size)
            self.g_off = 1 / torch.normal(device_params["r_off"], device_params["r_off_stddev"], size=self.size)
            self.viability = device_params["viability"]
            
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
        self.g_s_wl_in = torch.ones(self.tile_rows) * 1
        self.g_s_wl_out = torch.ones(self.tile_rows) * 1e-9
        self.g_s_bl_in = torch.ones(self.tile_rows) * 1e-9
        self.g_s_bl_out = torch.ones(self.tile_rows) * 1

        # WL & BL voltages that are not the signal, assume bl_in, wl_out are tied low and bl_out is tied to 1 V. 
        self.v_bl_in = torch.zeros(self.size[1])
        self.v_bl_out = torch.ones(self.size[1])
        self.v_wl_out = torch.zeros(self.size[0])
        
        # Conductance Matrix; initialize each memristor at the on resstance
        self.W = torch.ones(self.size) * self.g_on

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

    
    # Iterates through the tiles and solves each and then adds their outputs together. 
    def solve(self, voltage):
        output = torch.zeros((voltage.size(1), self.size[1]))
        for i in range(self.size[0] // self.tile_rows):
            for j in range(self.size[1] // self.tile_cols):
                coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_rows))
                vect = voltage[i*self.tile_rows:(i+1)*self.tile_rows,:]
                solution = self.tile_solve(coords, vect)
                output += torch.cat((torch.zeros(voltage.size(1), j*self.tile_cols), solution, torch.zeros((voltage.size(1), (self.size[1] // self.tile_cols - j - 1)*self.tile_cols))), axis=1)
        self.current_history.append(output)
        return output

    # solves a specific tile. 
    def tile_solve(self, coords, vectors):
        if str(coords) not in self.saved_tiles.keys(): M = self.make_M(coords) # Lazy hash
        else: M = self.saved_tiles[str(coords)]
        Es = torch.cat(tuple(self.make_E(vectors[:, i]).view(-1,1) for i in range(vectors.size(1))), axis=1)
        V = torch.transpose(-torch.sub(*torch.chunk(torch.matmul(M, Es), 2, dim=0)), 0, 1).view(-1, self.tile_rows, self.tile_cols)
        I = torch.sum(V * self.W[coords], axis=1)
        return I

    # Constructs the E matrix in MV = E.
    def make_E(self, v_wl_in):
        m, n = self.tile_rows, self.tile_cols
        E = torch.cat([torch.cat(((v_wl_in[i]*self.g_s_wl_in[i]).view(1), torch.zeros(n-2), (self.v_wl_out[i]*self.g_s_wl_out[i]).view(1))) for i in range(m)] +
                      [torch.cat(((-self.v_bl_in[i]*self.g_s_bl_in[i]).view(1), torch.zeros(m-2),(-self.v_bl_in[i]*self.g_s_bl_out[i]).view(1))) for i in range(n)]).view(-1, 1)
        return E

    # Constructs the M matrix in MV = E. 
    def make_M(self, coords):
        
        g = self.W[coords]
        m, n = self.tile_rows, self.tile_cols

        def makec(j):
            c = torch.zeros(m, m*n)
            for i in range(m):
                c[i,n*(i) + j] = g[i,j]
            return c
  
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
        
        A = torch.block_diag(*tuple(torch.diag(g[i,:])
                          + torch.diag(torch.cat((self.g_wl, self.g_wl * 2 * torch.ones(n-2), self.g_wl)))
                          + torch.diag(self.g_wl * -1 *torch.ones(n-1), diagonal = 1)
                          + torch.diag(self.g_wl * -1 *torch.ones(n-1), diagonal = -1)
                          + torch.diag(torch.cat((self.g_s_wl_in[i].view(1), torch.zeros(n - 2), self.g_s_wl_out[i].view(1))))
                                   for i in range(m)))
        B = torch.block_diag(*tuple(-torch.diag(g[i,:]) for i in range(m)))
        C = torch.cat([makec(j) for j in range(n)],dim=0)
        D = torch.cat([maked(j) for j in range(0,n)], dim=0)
        M = torch.inverse(torch.cat((torch.cat((A,B),dim=1), torch.cat((C,D),dim=1)), dim=0))

        self.saved_tiles[str(coords)] = M

        return M

    # Handles programming for the crossbar instance. 
    def register_linear(self, matrix, bias=None):

        self.tensors.append(matrix)
        row, col = self.find_space(matrix.size(0), matrix.size(1))
        # Need to add checks for bias size and col size
        
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
            mat_scale_factor = torch.max(torch.abs(matrix)) / (torch.max(self.g_on) - torch.min(self.g_off)) * 2
            scaled_matrix = matrix / mat_scale_factor
            for i in range(row, row + scaled_matrix.size(0)):
               for j in range(col, col + scaled_matrix.size(1)):
                   midpoint = (self.g_on[i,j] - self.g_off[i,j]) / 2 + self.g_off[i,j]
                   right_state = midpoint + scaled_matrix[i-row,j-col] / 2
                   left_state = midpoint - scaled_matrix[i-row,j-col] / 2
                   self.W[i,2*j+1] = self.clip(right_state + torch.normal(mean=0,std=right_state*self.viability), i, 2*j+1)
                   self.W[i,2*j] = self.clip(left_state + torch.normal(mean=0,std=left_state*self.viability), i, 2*j)

        self.apply_stuck()
        
        return ticket(row, col, matrix.size(0), matrix.size(1), matrix, mat_scale_factor, self)
    
    def clip(self, tensor, i, j):
        if self.g_off[i,j] < tensor < self.g_on[i,j]:
            return tensor
        elif tensor > self.g_on[i,j]:
            return self.g_on[i,j]
        else:
            return self.g_off[i,j]
    
    def apply_stuck(self):
        self.W[self.state_mask == 0] = self.g_off[self.state_mask==0]
        self.W[self.state_mask == 1] = self.g_on[self.state_mask==1]

    def which_tiles(self, row, col, m_row, m_col):
        return itertools.product(range(row // self.tile_rows, (row + m_row) // self.tile_rows + 1),
                                 range(col // self.tile_cols,(col + m_col) // self.tile_cols + 1),
        )

    def find_space(self, m_row, m_col):

        if m_row > self.size[0] or m_col*2 > self.size[1]:
                raise ValueError("Matrix with size ({}, {}) is too large for crossbar of size ({}, {})".format(m_row, m_col, self.size[0], self.size[1]))
            
        # Format is (*indexes of top left corner, *indexes of bottom right corner + 1 (it's zero indexed))
        if not self.mapped:
            
            self.mapped.append((0,0,m_row,m_col))
        else:
            if self.mapped[-1][3] + m_col < self.size[1]:
                self.mapped.append((self.mapped[-1][0], self.mapped[-1][3], m_row, m_col))
            else:
                if m_col > (self.size[0] - self.mapped[-1][2]):
                    raise ValueError("Matrix with {} rows does not fit on crossbar with {} free rows".format(m_col, self.size[0] - self.mapped[-1][2]))    
                self.mapped.append((self.mapped[-1][2], 0, m_row, m_col))
                
        #self.mapped.append((self.mapped[-1][0] + self.mapped[-1][2], self.mapped[-1][1] + self.mapped[-1][3], m_row, m_col))
        return self.mapped[-1][0], self.mapped[-1][1] 
    
    def clear(self):
        self.mapped = []
        self.tensors = []
        self.saved_tiles = {}
        self.W = torch.ones(self.size) * self.g_on
