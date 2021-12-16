
from . import crossbar, adc


class Interface:

    def __init__(self, matrix, crossbars, bits):
        self.bits = bits
        self.matrix = matrix
        self.crossbars = crossbars
        self.weights, matrices, self.m_min, self.m_range = self.chop_matrix(matrix)
        self.tickets = [crossbar.register_bit_matrix(m) for m, crossbar in zip(matrices, crossbars)]
        
    def vmm(self, v):
        return torch.sum(w*cb.vmm(t, v) for w, cb, t in zip(self.weights, self.crossbars, self.tickets), axis=0) * self.m_range + \
               torch.ones_like(matrix).dot(v)*self.m_min
               

    def chop_matrix(self, matrix):
        bpd = self.bits / len(self.crossbars)
        mm = torch.min(matrix)
        r = (torch.max(matrix) - mm)
        
        if r = 0:
            return [1] + [0]*(len(self.crossbars)-1),
                   [torch.cat(torch.ones(1, *matrix.size()), torch.zeros(bpd - 1, *matrix.size()), axis=0)] + \
                   [torch.zeros(bpd, *matrix.size()) for i in range(len(self.crossbars)-1)],
                   mm,
                   r
        else: matrix = (matrix - mm) * 2**self.bits / r

        bit_matrix = torch.flip(vector.unsqueeze(-1).bitwise_and(2**torch.arange(self.bits)).ne(0).byte().squeeze(), (1,)).type(torch.float)
        weights = [sum(2**b for b in range(vb1, vb1+bits_per_device)) for vb1 in range(0, self.bits, bpd)]
        matrices = [bit_matrix[i:i+bpd] for i in range(0, self.bits, bpd)]
        
        return weights, matrices, mm, r

        
            
