import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    err = torch.norm(pose_error - torch.eye(4).unsqueeze(0).expand_as(pose_error), dim=(1, 2))
    return err 

# def RPE(input_m, target_m):
#     pose_error = torch.bmm(torch.inverse(input_m), target_m)
#     r_err = rotationError(pose_error)
#     t_err = translationError(pose_error)
#     return t_err + r_err

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

def rotation_error(R_gt, R_est):
    # 将旋转矩阵或旋转向量转换为四元数
    q_gt = F.normalize(torch.quaternion.from_rotation_matrix(R_gt), dim=-1)
    q_est = F.normalize(torch.quaternion.from_rotation_matrix(R_est), dim=-1)

    # 计算四元数之间的角度差
    q_diff = q_gt.inverse() * q_est
    angle_diff = 2 * torch.acos(torch.clamp(q_diff.real, -1, 1))

    # 将角度差转换为度量单位（如弧度或角度）
    # 这里使用弧度作为度量单位
    error = angle_diff.mean()

    return error

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

