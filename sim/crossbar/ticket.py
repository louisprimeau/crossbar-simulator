import torch
from memory_profiler import profile

class ticket:
    def __init__(self, index, row, col, m_rows, m_cols, matrix, mat_scale_factor, crossbar):
        self.row, self.col = row, col
        self.m_rows, self.m_cols = m_rows, m_cols
        self.crossbar = crossbar
        self.mat_scale_factor = mat_scale_factor
        self.matrix = matrix
        self.index = index
        self.row_sum = torch.sum(self.matrix, axis=0)
        
    def remap(self, new_matrix, bypass=False):
        assert new_matrix.size() == self.matrix.size(), "new matrix is not the same size as the old one!"
        new_ticket = self.crossbar.register_linear(new_matrix, index=self.index, bypass=bypass)
        
        self.mat_scale_factor = new_ticket.mat_scale_factor
        self.matrix = new_ticket.matrix
        self.row_sum = torch.sum(self.matrix, axis=0)
        
        assert self.row == new_ticket.row
        assert self.col == new_ticket.col
        assert self.m_rows == new_ticket.m_rows
        assert self.m_cols == new_ticket.m_cols
        assert self.index == new_ticket.index
        
    def prep_vector(self, vector, v_bits):

        # Scale vector to [0, 2^v_bits]
        vect_min = torch.min(vector) if vector.numel() > 1 else 0.0
        vector = vector - vect_min

        vect_scale_factor = torch.max(vector) / (2**v_bits - 1)
        
        vector = torch.round(vector / vect_scale_factor if vect_scale_factor != 0.0 else vector).type(torch.long)
        if v_bits > 1 and vector.numel() > 1:
            bit_vector = torch.flip(vector.unsqueeze(-1).bitwise_and(2**torch.arange(v_bits)).ne(0).byte().squeeze(), (1,)).type(torch.float) * self.crossbar.V
        else:
            bit_vector = torch.ones(vector.size()).type(torch.float) * self.crossbar.V

    
        # Pad bit vector with unselected voltages
        pad_vector = torch.zeros(self.crossbar.size[0], v_bits)
        pad_vector[self.row:self.row + self.m_rows,:] = bit_vector

        return pad_vector, vect_scale_factor, vect_min
    
    def vmm(self, vector, v_bits=16):
        assert vector.size(1) == 1, "vector wrong shape"

        crossbar = self.crossbar
        
        # Rescale vector and convert to bits.
        pad_vector, vect_scale_factor, vect_min = self.prep_vector(vector, v_bits)

        # Solve crossbar circuit
        output = crossbar.solve(pad_vector)

        # Get relevant output columns and add binary outputs
        output = output.view(v_bits, -1, 2)[:, :, 0] - output.view(v_bits, -1, 2)[:, :, 1]

        for i in range(output.size(0)):
            output[i] *= 2**(v_bits - i - 1)

        output = torch.sum(output, axis=0)[self.col:self.col + self.m_cols]

        output = (output / crossbar.V * vect_scale_factor * self.mat_scale_factor) + vect_min * self.row_sum
        return output.view(-1, 1)

class ticket_direct:
    def __init__(self, index, row, col, m_rows, m_cols, conductance_matrix, crossbar):
        self.index = index
        self.row, self.col = row, col
        self.m_rows, self.m_cols = m_rows, m_cols
        self.conductance_matrix = conductance_matrix
        self.crossbar = crossbar
        
    def remap(self, new_matrix):
        assert new_matrix.size() == self.conductance_matrix.size(), "new matrix is not the same size as the old one!"
        new_ticket = self.crossbar.register_direct(new_matrix, index=self.index)
        self.conductance_matrix = new_ticket.conductance_matrix

        assert self.index == new_ticket.index
        assert self.row == new_ticket.row
        assert self.col == new_ticket.col
        assert self.m_rows == new_ticket.m_rows
        assert self.m_cols == new_ticket.m_cols
        assert self.index == new_ticket.index
            
    def apply_voltage(self, voltage):
        assert len(voltage.size()) == 1 and voltage.size(0) == self.m_rows
        assert torch.all(voltage == 1 or voltage == 0), "Input voltage not a binary vector"
        pad_vector = torch.nn.functional.pad(voltage, (self.row, self.crossbar.size[0] - self.row - self.m_rows)).unsqueeze(1) * self.crossbar.V
        out = self.crossbar.solve(pad_vector) / self.crossbar.V
        return out.squeeze()[self.col:self.col+self.m_cols].view(-1)
