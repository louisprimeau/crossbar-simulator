import torch
from memory_profiler import profile

class ticket:
    def __init__(self, index, row, col, m_rows, m_cols, matrix, mat_scale_factor, v_bits, crossbar):
        self.row, self.col = row, col
        self.m_rows, self.m_cols = m_rows, m_cols
        self.crossbar = crossbar
        self.mat_scale_factor = mat_scale_factor
        self.matrix = matrix
        self.index = index
        self.row_sum = torch.sum(self.matrix, axis=0)
        self.v_bits = v_bits

    def free(self):
        self.crossbar.tensors = self.crossbar.tensors[:self.index] + self.crossbar.tensors[self.index+1:]
        self.crossbar.mapped = self.crossbar.mapped[:self.index] + self.crossbar.mapped[self.index+1:]
        self.crossbar.tickets = self.crossbar.tickets[:self.index] + self.crossbar.tickets[self.index+1:]

        

    
