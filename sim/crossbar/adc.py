
import torch
class adc:

    def __init__(self, params):
        
        self.resolution = params['resolution']
        self.V_ref_Hi = params['V_ref_Hi']
        self.V_ref_Lo = params['V_ref_Lo']
        self.TI = params['TI']
        
    def V_read(self, V):
        assert len(V.size()) == 1
        lo, hi = self.V_ref_Lo, self.V_ref_Hi
        V = torch.clip(V, min=lo, max=hi)
        V = (V - lo) / (hi - lo)
        return torch.flip(vector.unsqueeze(-1).bitwise_and(2**torch.arange(self.resolution)).ne(0).byte().squeeze(), (1,)).type(torch.float)

    def I_read(self, I):
        return V_read(I * self.TI)

