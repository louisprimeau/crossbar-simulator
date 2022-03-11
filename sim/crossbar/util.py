import torch
import numpy as np

#Utility function, should be moved
def print_mapping(tensors, mapping, crossbar_size):
    cb = torch.zeros(*crossbar_size)
    for t, m in zip(tensors, mapping):
        cb[m[0]:m[0]+m[2], m[1]:m[1]+m[3]] = t
    rows = torch.nonzero(cb, as_tuple=True)[0].tolist()
    cols = torch.nonzero(cb, as_tuple=True)[1].tolist()
    values = cb[torch.nonzero(cb, as_tuple=True)].tolist()
    for val in zip(rows,cols,values):
        print(val[0], val[1], val[2], sep=", ")


# Implements scipy's minmax scaler except just between 0 and 1 for torch Tensors.
# Taken from a ptrblck post on the PyTorch forums.
class MinMaxScaler(object):
    def __call__(self, tensor):
        self.scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        self.min = tensor.min(dim=1, keepdim=True)[0]
        tensor.sub_(self.min).mul_(self.scale)
        return tensor
    def inverse_transform(self, tensor):
        tensor.div_(self.scale).add_(self.min)
        return tensor

@torch.no_grad()
def bit_slice(tensor, bits, n=1):
    assert bits % n  == 0
    shape = tensor.size()
    tensor = tensor.view(-1)
    bpc = bits // n
    low = torch.min(tensor) #to prevent infinite orders of magnitude
    r = (torch.max(tensor) - low) / (2**bits - 1) # SCALING HERE !!!!!!!
    if r == 0:
        tensors = [torch.cat((torch.ones(1, *shape), torch.zeros(bpc - 1, *shape)), axis=0)] + [torch.zeros(bpc, *shape) for i in range(n-1)]
        return tensors, [1] + [0] * (n - 1), low, r
    else: 
        tensor = (tensor - low) / r
        tensor = torch.round(tensor).type(torch.int32)
    if bits == 1: return [(tensor > 0.5).type(torch.float).reshape(1, *shape)], [1], low, r
    bit_tensor = torch.flip(tensor.unsqueeze(-1).bitwise_and(2**torch.arange(bits)).ne(0).byte().squeeze(), (1,)).type(torch.float)
    lsc = 2**bpc - 1
    weights = list(reversed([sum(2**b for b in range(vb1, vb1+bpc)) / lsc for vb1 in range(0, bits, bpc)])) # actually just powers of 2
    tensors = [bit_tensor[:, i:i+bpc].transpose(0, 1).reshape(bpc, *shape) for i in range(0, bits, bpc)]
    return tensors, weights, low, r

@torch.no_grad()
def bit_join(tensors, weights, low, r):
    out = 0
    for tensor, weight in zip(tensors, reversed(weights)):
        out += tensor * weight
    return sum(tensor * weight for tensor, weight in zip(tensors, weights)) * r + low

def bittensor2string(tensor):
    sz = tensor.size()
    str_arr = tensor.numpy().astype(int).astype(str)
    str_arr = str_arr.reshape(sz[0], -1).T
    new_arr = np.chararray(str_arr.shape[0], itemsize=str_arr.shape[1])
    for i, col in enumerate(str_arr): new_arr[i] = ''.join(col)
    return np.array2string(new_arr.reshape(sz[1:]))

def round_to(tensor, levels):
    size = tensor.size()
    return torch.argmin(torch.abs(tensor.view(-1) - levels.view(-1, 1).expand(-1, tensor.numel())), axis=0).reshape(size)
