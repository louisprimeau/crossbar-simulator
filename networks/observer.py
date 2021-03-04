import torch
class observer():
    def __init__(self):
        self.history = [None, None]
        self.on = False
    def append(self, tensor1, tensor2):
        if not self.on: return None
        
        if self.history[0] is None: self.history[0] = tensor1
        else: self.history[0] = torch.cat((self.history[0], tensor1), axis=0)
        
        if self.history[1] is None: self.history[1] = tensor2
        else: self.history[1] = torch.cat((self.history[1], tensor2), axis=0)


