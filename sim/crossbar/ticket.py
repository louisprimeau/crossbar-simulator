import torch

class ticket:
    def __init__(self, row, col, m_rows, m_cols, matrix, mat_scale_factor, crossbar):
        self.row, self.col = row, col
        self.m_rows, self.m_cols = m_rows, m_cols
        self.crossbar = crossbar
        self.mat_scale_factor = mat_scale_factor
        self.matrix = matrix
        
        
    def prep_vector(self, vector, v_bits):

        # Scale vector to [0, 2^v_bits]
        vect_min = torch.min(vector)
        vector = vector - vect_min        
        vect_scale_factor = torch.max(vector) / (2**v_bits - 1)
        vector = vector / vect_scale_factor if vect_scale_factor != 0.0 else vector

        # decompose vector by bit
        bit_vector = torch.zeros(vector.size(0),v_bits)
        bin2s = lambda x : "".join(reversed( [str((int(x) >> i) & 1) for i in range(v_bits)] ) )
        for j in range(vector.size(0)):
            bit_vector[j,:] = torch.Tensor([float(i) for i in list(bin2s(vector[j]))])
        bit_vector *= self.crossbar.V

        # Pad bit vector with unselected voltages
        pad_vector = torch.zeros(self.crossbar.size[0], v_bits)        
        pad_vector[self.row:self.row + self.m_rows,:] = bit_vector

        return pad_vector, vect_scale_factor, vect_min
    
    def vmm(self, vector, v_bits=4):
        assert vector.size(1) == 1, "vector wrong shape"

        crossbar = self.crossbar
        
        # Rescale vector and convert to bits.
        pad_vector, vect_scale_factor, vect_min = self.prep_vector(vector, v_bits)
        
        # Solve crossbar circuit
        output = crossbar.solve(pad_vector)
        
        # Get relevant output columns and add binary outputs        
        output = output.view(v_bits, -1, 2)[:,:,0] - output.view(v_bits, -1, 2)[:,:,1]
    
        for i in range(output.size(0)):
            output[i] *= 2**(v_bits - i - 1)
        output = torch.sum(output, axis=0)[self.col:self.col + self.m_cols] 
        
        # Rescale output
        magic_number = 1 # can use to compensate for resistive losses in the lines. Recommend multiplying a bunch of 8x8 integer matrices to find this.
        
        output = (output / crossbar.V * vect_scale_factor * self.mat_scale_factor) / magic_number + torch.sum(vect_min * self.matrix, axis=0)
        
        return output.view(-1, 1)
