import torch

def triangle(V_high, V_low, N1, N2, N3):
    return torch.cat((torch.linspace(0, V_low, N1), 
                      torch.linspace(V_low, V_high, N2), 
                      torch.linspace(V_high, 0, N3)))

def sine(V, N):
    return V * torch.sin(torch.linspace(0, 2 * 3.14159265359, N))

def square(V_high, V_low, N_rise, N_flat):
    return torch.cat((torch.linspace(0, V_high, N_rise), 
                      torch.ones(N_flat) * V_high, 
                      torch.linspace(V_high, V_low, N_rise * 2),
                      torch.ones(N_flat) * V_low))    

def half_square(V_high, N_rise, N_flat):
    return torch.cat((torch.linspace(0, V_high, N_rise), 
                      torch.ones(N_flat) * V_high, 
                      torch.linspace(V_high, 0, N_rise)))
