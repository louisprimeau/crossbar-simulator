from numpy import record
import torch
from . import waveform

def set(memristor_model, w, target_conductance, record_history=False):

    if record_history: history = []
    write_waveform = waveform.half_square(memristor_model.v_off * 5, 1, 5).reshape(-1, 1)
    dt = 0.001
    error = 1e12

    for i in range(5000):
        for i in range(write_waveform.size(0)):
            w = memristor_model.euler_step(write_waveform[i], w, dt)
            if record_history: history.append(w)
        new_error = torch.abs(memristor_model.conductance(w) - target_conductance)
        if new_error >= error:
            if record_history: return w, torch.cat(history)
            return w
        error = new_error

    assert False, "failed"


def reset(memristor_model, w, target_conductance, record_history=False):

    if record_history: history = []
    write_waveform = waveform.half_square(memristor_model.v_on * 5, 1, 5).reshape(-1, 1)
    dt = 0.001
    error = 1e12

    for i in range(5000):
        for i in range(write_waveform.size(0)):
            w = memristor_model.euler_step(write_waveform[i], w, dt)
            if record_history: history.append(w)
        new_error = torch.abs(memristor_model.conductance(w) - target_conductance)
        if new_error >= error:
            if record_history: return w, torch.cat(history)
            return w
        error = new_error

    assert False, "failed"