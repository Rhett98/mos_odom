import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotationError(pose_error):
    a = pose_error[:,0,0]
    b = pose_error[:,1,1]
    c = pose_error[:,2,2]
    d = 0.5*(a+b+c-1.0)
    return torch.arccos(max(min(d,1.0),-1.0))

def translationError(pose_error):
    dx = pose_error[:,0,3]
    dy = pose_error[:,1,3]
    dz = pose_error[:,2,3]
    return torch.sqrt(dx**2+dy**2+dz**2)

def RPE(input_m, target_m):
    pose_error = torch.bmm(torch.inverse(input_m), target_m)
    r_err = rotationError(pose_error)
    t_err = translationError(pose_error)
    return t_err + r_err

def valid_RPE(input_m, target_m):
    pose_error = torch.bmm(torch.inverse(input_m), target_m)
    r_err = rotationError(pose_error)
    t_err = translationError(pose_error)
    return t_err, r_err

def multiscaleEPE(network_output, target, weights=None):
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * RPE(output, target)
    return loss

def get_all_err(network_output, target):
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    t_err_list = []
    r_err_list = []
    for i in range(len(network_output)):
        t_err , r_err = valid_RPE(network_output[i], target)
        t_err_list.append(t_err)
        r_err_list.append(r_err)
    return t_err_list, r_err_list


