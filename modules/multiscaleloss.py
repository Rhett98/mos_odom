import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import T2quat_tran

def rotationError(pose_error):
    a = pose_error[:,0,0]
    b = pose_error[:,1,1]
    c = pose_error[:,2,2]
    d = 0.5*(a+b+c-1.0)
    return torch.arccos(torch.clamp(d, -1.0, 1.0))

def translationError(pose_error):
    dx = pose_error[:,0,3]
    dy = pose_error[:,1,3]
    dz = pose_error[:,2,3]
    return torch.sqrt(dx**2+dy**2+dz**2)

def RPE(input_m, target_m):
    pose_error = torch.bmm(torch.inverse(input_m), target_m)
    # err = torch.norm(pose_error-torch.eye(4))
    err = torch.norm(pose_error - torch.eye(4).unsqueeze(0).expand_as(pose_error).cuda(), dim=(1, 2))
    return torch.mean(err) 

# def RPE(input_m, target_m):
#     pose_error = torch.bmm(torch.inverse(input_m), target_m)
#     r_err = rotationError(pose_error)
#     t_err = translationError(pose_error)
#     return t_err + r_err

def valid_RPE(input_m, target_m):
    pose_error = torch.bmm(torch.inverse(input_m), target_m)
    r_err = rotationError(pose_error)
    t_err = translationError(pose_error)
    return torch.mean(r_err), torch.mean(t_err)

def multiscaleRPE(network_output, target, weights=None):
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
        t_err_list.append(torch.mean(t_err))
        r_err_list.append(torch.mean(r_err))
    return t_err_list, r_err_list

class scaleHWSLoss(nn.Module):
    """ Geometric loss function from PoseNet paper """
    def __init__(self, sx=0.0, sq=-2.5, eps=1e-6, weight=None):
        super(scaleHWSLoss, self).__init__()
        self.sx = nn.Parameter(torch.Tensor([sx])).cuda()
        self.sq = nn.Parameter(torch.Tensor([sq])).cuda()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.weight = weight
        if self.weight is None:
            self.weight = [1.6, 0.8, 0.4, 0.2, 0.1]  # as in original article
              
    def forward(self, network_output, target): 
        assert(len(self.weight) == len(network_output)) 
        if type(network_output) not in [tuple, list]:
            network_output = [network_output] 
        loss = 0 
        for i in range(len(network_output)):
            q ,t = T2quat_tran(network_output[i])
            q_gt ,t_gt = T2quat_tran(target)
            loss_q = self.l2_loss(q, q_gt)
            loss_t = self.l1_loss(t, t_gt)   
            loss += self.weight[i]*(torch.exp(-self.sx)*loss_t + self.sx \
                + torch.exp(-self.sq)*loss_q + self.sq)
        
        return loss

if __name__ == '__main__':
    x = torch.Tensor([[[ 0.9996, -0.0287,  0.0053,  0.1721],
                        [ 0.0285,  0.9990,  0.0338,  0.0229],
                        [-0.0063, -0.0337,  0.9994,  0.0738],
                        [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    y = torch.Tensor([[[ 0.9993, -0.0352, -0.0078,  0.2248],
                        [ 0.0356,  0.9980,  0.0527,  0.0250],
                        [ 0.0060, -0.0530,  0.9986,  0.0130],
                        [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    z = RPE(x, y)
    # z = valid_RPE(x, y)
    print(z)

