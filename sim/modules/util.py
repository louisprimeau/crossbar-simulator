

import torch

def kernel_to_matrix(kernel):
    return kernel.view(kernel.size(0), kernel.size(1) * kernel.size(2) * kernel.size(3))

def identity_kernel(input_size, kernel_size, dtype=torch.float):
    kernel = torch.zeros(input_size, input_size, kernel_size, kernel_size, dtype=dtype)
    torch.nn.init.dirac_(kernel, groups=1)
    as_matrix = kernel_to_matrix(kernel)
    return kernel, as_matrix

def kaiming_kernel(input_size, output_size, kernel_size, dtype=torch.float, param=False):
    kernel = torch.zeros(output_size, input_size, kernel_size, kernel_size, dtype=dtype)
    torch.nn.init.kaiming_normal_(kernel, nonlinearity='relu')
    as_matrix = kernel_to_matrix(kernel)
    if param: kernel = torch.nn.Parameter(kernel)
    return kernel, as_matrix

def normal_kernel(output_size, dtype=torch.float, param=False):
    kernel = torch.zeros(output_size, dtype=dtype)
    torch.nn.init.normal_(kernel)
    as_matrix = kernel.reshape(output_size, 1)
    if param: kernel = torch.nn.Parameter(kernel)
    return kernel, as_matrix

def init_linear(input_size, output_size, dtype=torch.float, param=False):
    matrix = torch.zeros(output_size, input_size, dtype=dtype)
    torch.nn.init.kaiming_uniform_(matrix, a=5**0.5)
    if param: matrix = torch.nn.Parameter(matrix)
    return matrix
