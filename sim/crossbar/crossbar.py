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

from traceback import print_last
import torch
import numpy as np
import itertools
import time
import warnings
from . import util, adc

class crossbar:
    def __init__(self, device_params, deterministic=False):

        
        self.deterministic = deterministic # Useful for debugging
        self.V = device_params["Vdd"] # Power Supply Voltage
        self.r_wl = torch.Tensor((device_params["r_wl"],)) # Wordline Resistance 
        self.r_bl = torch.Tensor((device_params["r_bl"],)) # Bitline Resistance
        self.size = device_params["m"], device_params["n"]  # Number of crossbars, rows, columns
        self.r_on = device_params["r_on"] # Device on resistance
        self.r_off = device_params["r_off"] # Device off resistance

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
        
        # Input & Output resistances
        self.r_in = device_params["r_in"]
        self.r_out = device_params["r_out"]
        
        # Device Programming Error approximation
        # assumes ideal programming to any conductance but the end result is perturbed by gaussian noise with spread
        # equal to some percentage (the "viability") of the conductance.
            
        self.g_on = torch.ones(self.size) / device_params["r_on"]
        self.g_off =  torch.ones(self.size) / device_params["r_off"]
        #self.g_on = 1 / torch.normal(device_params["r_on"], device_params["r_on_stddev"], size=self.size)
        #self.g_off = 1 / torch.normal(device_params["r_off"], device_params["r_off_stddev"], size=self.size) 

        # Set up conductances around edges of crossbar
        self.g_s_wl_in = torch.ones(self.tile_rows) / self.r_in
        self.g_s_wl_out = torch.ones(self.tile_rows) * 1e-9 # floating
        self.g_s_bl_in = torch.ones(self.tile_cols) * 1e-9 # floating
        self.g_s_bl_out = torch.ones(self.tile_cols) / self.r_out

        # WL & BL voltages that are not the signal, assume bl_in, wl_out are tied low and bl_out is tied to 1 V. 
        self.v_bl_in = torch.zeros(self.size[1])
        self.v_bl_out = torch.zeros(self.size[1])
        self.v_wl_out = torch.zeros(self.size[0])

        # Set up E matrix
        E_B = torch.cat([torch.cat(((-self.v_bl_in[i] * self.g_s_bl_in[i]).view(1), torch.zeros(self.tile_cols-2), (-self.v_bl_in[i] * self.g_s_bl_out[i]).view(1))).unsqueeze(1) for i in range(self.tile_rows)])
        E_W = torch.cat([torch.cat(((torch.zeros(1) * self.g_s_wl_in[i]).view(1), torch.zeros(self.tile_cols-2), (self.v_wl_out[i].view(1) * self.g_s_wl_out[i]).view(1))) for i in range(self.tile_rows)]).unsqueeze(1)
        self.E = torch.cat((E_B, E_W))

        # Programming Viability
        self.viability = device_params["viability"]

        # Conductance Matrix; initialize each memristor at the on resistance
        self.W = torch.ones(self.size) * self.g_on #+ torch.randn(self.size) * self.g_on * 0.1

        # Stuck-on & stuck-on device nonideality 
        self.p_stuck_on = device_params["p_stuck_on"]
        self.p_stuck_off = device_params["p_stuck_off"]
        state_dist = torch.distributions.categorical.Categorical(probs=torch.Tensor([self.p_stuck_on, self.p_stuck_off, 1 - self.p_stuck_on - self.p_stuck_off]))
        self.state_mask = state_dist.sample(self.size)

        # Storage for all mapped tensors and their positions. Used to get data off the crossbar after simulation. 
        self.mapped = []
        self.tensors = [] #original data of all mapped weights
        self.tickets = []
        self.saved_tiles = {}

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

    # Iterates through the tiles and solves each and then adds their outputs together. 
    def solve(self, voltage):
        p_tiles = self.programmed_tiles()
        output = torch.zeros((voltage.size(1), self.size[1]))
        for i, j in p_tiles:
            coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_cols))
            if str(coords) not in self.saved_tiles.keys():
                self.make_M(coords) # Lazy hash

        # This part would be super easy to parallelize.
        Es_all = [None] * (self.size[0] // self.tile_rows)
        for i in set(j for j, _ in p_tiles):
            Es_all[i] = self.make_Es(voltage[i*self.tile_rows:(i+1)*self.tile_rows,:])
        
        for i, j in p_tiles:
            coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_cols))    
            M = self.saved_tiles[str(coords)]
            V = torch.transpose(torch.sub(*torch.chunk(torch.matmul(M, Es_all[i]), 2, dim=0)), 0, 1).view(-1, self.tile_rows, self.tile_cols)
            I_all = V * self.W[coords]
            I = torch.sum(I_all, axis=1)
            output[:, j*self.tile_cols:(j+1)*self.tile_cols] += I 

        return output
    
    # Constructs the M matrix in MV = E. 
    def make_M(self, coords):
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
                d[i, j] = self.g_bl
                   
            i = m - 1
            d[i, n*(i-1) + j] = self.g_bl
            d[i, n*i + j] = -self.g_s_bl_out[j] - g[i,j] - self.g_bl
                
            return d
        
        A = torch.block_diag(*tuple(makea(i) for i in range(m)))
        B = torch.block_diag(*tuple(-torch.diag(g[i,:]) for i in range(m)))
        C = torch.cat([makec(j) for j in range(n)],dim=0)
        D = torch.cat([maked(j) for j in range(n)], dim=0)

        M = torch.cat((torch.cat((A,B),dim=1), torch.cat((C,D),dim=1)), dim=0)
        M = torch.inverse(M)

        self.saved_tiles[str(coords)] = M

        return M

    # Handles programming for the crossbar instance.
    @torch.no_grad()
    def register_linear(self, matrix):
        assert len(self.tensors) == len(self.mapped), "tensors and indices out of sync"            
        index = len(self.tensors)
        self.tensors.append(matrix)
        row, col = self.find_space(matrix.size(0), matrix.size(1)*2)

        mat_scale_factor = torch.max(torch.abs(matrix)) / (torch.max(self.g_on) - torch.min(self.g_off)) * 3
        if mat_scale_factor == 0.0: mat_scale_factor = 1.0
        scaled_matrix = matrix / mat_scale_factor
        midpoint = 1/self.r_on - 1/self.r_off
        right_state, left_state = midpoint + scaled_matrix / 2,  midpoint - scaled_matrix / 2
        right_state = right_state + torch.randn(right_state.size()) * right_state * self.viability
        left_state = left_state + torch.randn(left_state.size()) * left_state * self.viability
        out = torch.stack((left_state, right_state), dim=2).view(right_state.size(0), right_state.size(1) * 2)

        new_W = self.W.clone().detach()
        new_W[row:row+scaled_matrix.size(0), col:col + scaled_matrix.size(1) * 2] = out
        self.W = new_W

        if not self.deterministic: self.apply_stuck()
        self.tickets.append(Ticket(index, row, col // 2, matrix.size(0), matrix.size(1), matrix, self, differential=True, scale=mat_scale_factor))
        return self.tickets[-1]
    
    @torch.no_grad()
    def register_bit_matrix(self, bit_matrix):
        n_bits = bit_matrix.size(0)
        matrix = torch.sum(2**(torch.flip(torch.arange(n_bits), dims=(0,))).reshape(-1, 1, 1) * bit_matrix, axis=0)
        assert len(self.tensors) == len(self.mapped), "tensors and indices out of sync"            
        index = len(self.tensors)
        self.tensors.append(matrix)
        row, col = self.find_space(matrix.size(0), matrix.size(1))

        g_on, g_off = 1 / self.r_on, 1 / self.r_off
        states = g_off + (g_on - g_off) * matrix / (2**n_bits - 1)
        self.W[row:row+matrix.size(0), col:col+matrix.size(1)] = states + torch.randn(matrix.size()) * states * self.viability
        if not self.deterministic: self.apply_stuck()
        self.tickets.append(Ticket(index, row, col, matrix.size(0), matrix.size(1), matrix, self, n_bits, differential=False))
        return self.tickets[-1]


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
                    raise ValueError("Matrix with {} rows does not fit on crossbar with {} free rows".format(m_col, self.size[0] - self.mapped[-1][2]))
                    
                self.mapped.append((self.mapped[-1][2], 0, m_row, m_col))
                
        #self.mapped.append((self.mapped[-1][0] + self.mapped[-1][2], self.mapped[-1][1] + self.mapped[-1][3], m_row, m_col))
        return self.mapped[-1][0], self.mapped[-1][1] 

    def make_V(self, vector, v_bits, row, m_rows):
        """vect_min = torch.min(vector) - 1.0 if vector.numel() > 1 else -1.0
        vector = vector - vect_min
        vect_scale_factor = torch.max(vector) / (2**v_bits - 1)
        vector = torch.round(vector / vect_scale_factor if vect_scale_factor != 0.0 else vector).type(torch.long)
        if v_bits > 1 and vector.numel() > 1: bit_vector = torch.flip(vector.unsqueeze(-1).bitwise_and(2**torch.arange(v_bits)).ne(0).byte().squeeze(), (1,)).type(torch.float) * self.V
        else: bit_vector = torch.ones(vector.size()).type(torch.float) * self.V"""
        bit_vector, _, vect_min, vect_scale_factor = util.bit_slice(vector, v_bits, n=1)
        pad_vector = torch.zeros(self.size[0], v_bits)
        pad_vector[row:row + m_rows,:] = bit_vector[0].T * self.V
        return pad_vector, vect_scale_factor, vect_min

    def vmm(self, ticket, vector, v_bits):
        print("vmm \n --------------------------------")
        Vs, vect_scale_factor, vect_min = self.make_V(vector, v_bits, ticket.row, ticket.m_rows)
        print("Vs", Vs)
        output = self.solve(Vs)
        if ticket.differential: output = output.view(v_bits, -1, 2)[:, :, 0] - output.view(v_bits, -1, 2)[:, :, 1]
        else: adc_output = self.adc.read_currents(output)
        output = adc_output * (self.size[0] * (2**ticket.n_bits-1))
        for b in range(v_bits):
            print(Vs[:, b])
            print(adc_output[b])
            print(output[b])
            print(self.W)
            print("\n")
        for i in range(output.size(0)): output[i] *= 2**(v_bits - i - 1)
        print("vmm outputs", output)
        print(vect_scale_factor)
        output = torch.sum(output, axis=0)[ticket.col:ticket.col + ticket.m_cols] * vect_scale_factor
        output = output.view(-1, 1)
        offset = (vect_min * torch.ones(1, 3).mm(ticket.matrix)).view(-1, 1)
        output += offset
        print("-----------------------------------------")
        return output
        
    def clear(self):
        self.mapped = []
        self.tensors = []
        self.saved_tiles = {}
        self.W = torch.ones(self.size) * self.g_on

    def attach_adc(self, resolution):
        assert len(self.tensors) == 0
        self.adc = adc.adc(self, resolution)


class Ticket:
    def __init__(self, index, row, col, m_rows, m_cols, matrix, crossbar, n_bits, differential=False, scale=1.0):
        self.row, self.col = row, col
        self.m_rows, self.m_cols = m_rows, m_cols
        self.crossbar = crossbar
        self.matrix = matrix
        self.index = index
        self.row_sum = torch.sum(self.matrix, axis=0)
        self.differential = differential
        self.mat_scale_factor = scale
        self.n_bits = n_bits

    def free(self):
        self.crossbar.tensors = self.crossbar.tensors[:self.index] + self.crossbar.tensors[self.index+1:]
        self.crossbar.mapped = self.crossbar.mapped[:self.index] + self.crossbar.mapped[self.index+1:]
        self.crossbar.tickets = self.crossbar.tickets[:self.index] + self.crossbar.tickets[self.index+1:]
