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
