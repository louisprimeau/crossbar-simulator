import torch
from . import util
from . import crossbar


class System:

    def __init__(self, n_crossbars, device_params):
        self.dac_bits = device_params["dac_resolution"]
        self.memristor_bits = device_params["device_resolution"]
        self.adc_bits = device_params["adc_resolution"]
        
        self.crossbars = [crossbar.crossbar(device_params) for _ in range(n_crossbars)]
        for cb in self.crossbars: cb.attach_adc(self.adc_bits)

    def program(self, matrix):
        return Interface(matrix, self.crossbars, self.dac_bits, self.memristor_bits, self.adc_bits)

class Interface:

    def __init__(self, matrix, crossbars, dac_bits, memristor_bits, adc_bits):
        self.dac_bits = dac_bits
        self.memristor_bits = memristor_bits
        self.adc_bits = adc_bits
        self.matrix = matrix
        self.crossbars = crossbars
        matrices, self.weights, self.m_min, self.m_range = util.bit_slice(matrix, self.memristor_bits, len(self.crossbars))
        self.tickets = [crossbar.register_bit_matrix(m) for m, crossbar in zip(matrices, crossbars)] # least significant part first
        
    def vmm(self, vector):
        vs = [cb.vmm(ticket, vector, self.dac_bits) for cb, ticket in zip(self.crossbars, self.tickets)]
        offset = torch.ones_like(self.matrix.T).mm(vector) * self.m_min
        bjoin = util.bit_join(vs, self.weights, 0.0, self.m_range)
        out = bjoin + offset
        return out

    def free(self):
        for ticket in self.tickets:
            ticket.free()
