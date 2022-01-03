import torch
from . import util
from . import crossbar


class System:

    def __init__(self, n_crossbars, bits, device_params):
        self.bits = bits
        self.crossbars = [crossbar.crossbar(device_params) for _ in range(n_crossbars)]

    def program(self, matrix):
        return Interface(matrix, self.crossbars, self.bits)

class Interface:

    def __init__(self, matrix, crossbars, bits):
        self.bits = bits
        self.matrix = matrix
        self.crossbars = crossbars
        matrices, self.weights, self.m_min, self.m_range = util.bit_slice(matrix, self.bits, len(self.crossbars))
        self.tickets = [crossbar.register_bit_matrix(m) for m, crossbar in zip(matrices, crossbars)]
        
    def vmm(self, vector):
        return util.bit_join([cb.vmm(ticket, vector) for cb, ticket in zip(self.crossbars, self.tickets)], self.weights, 0.0, self.m_range) + torch.ones_like(self.matrix.T).mm(vector) * self.m_min

    def free(self):
        for ticket in self.tickets:
            ticket.free()
