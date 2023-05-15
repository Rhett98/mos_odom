import yaml
import torch
from tqdm import tqdm
from utility.dataset.kitti.parser_multiscan import Parser

# ARCH = yaml.safe_load(open('config/arch/mos-test.yml', 'r'))
# DATA = yaml.safe_load(open('config/data/local-test.yaml', 'r'))
# data = '../dataset'
# epsilon_w = ARCH["train"]["epsilon_w"]
# parser = Parser(root=data,
#                         train_sequences=DATA["split"]["train"],
#                         valid_sequences=DATA["split"]["valid"],
#                         test_sequences=None,
#                         split='train',
#                         labels=DATA["labels"],
#                         color_map=DATA["color_map"],
#                         learning_map=DATA["learning_map"],
#                         learning_map_inv=DATA["learning_map_inv"],
#                         sensor=ARCH["dataset"]["sensor"],
#                         max_points=ARCH["dataset"]["max_points"],
#                         batch_size=ARCH["train"]["batch_size"],
#                         workers=ARCH["train"]["workers"],
#                         gt=True,
#                         shuffle_train=True)
# loader = parser.get_train_set()
# assert len(loader) > 0
# # for i, (proj_in, proj_mask,proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,trans,rot) in enumerate(loader):
# #     print(trans[0].data)
# #     print(rot[0].data)
# #     print("*******")
# content = torch.zeros(parser.get_n_classes(), dtype=torch.float)
# for cl, freq in DATA["content"].items():
#     x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
#     content[x_cl] += freq
# loss_w = 1 / (content + epsilon_w)  # get weights
# for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
#     if DATA["learning_ignore"][x_cl]:
#         # don't weigh
#         loss_w[x_cl] = 0
# print("Loss weights from content: ", loss_w.data)

# # set train and valid evaluator
# ignore_class = []
# for i, w in enumerate(loss_w):
#     if w < 1e-10:
#         ignore_class.append(i)
#         print("Ignoring class ", i, " in IoU evaluation")
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

def get_x_q(pose: torch.Tensor):
    """ Get x, q vectors from pose matrix 
    Args:
        pose (Bx4x4 array): relative pose
    Returns:
        x (Bx3x1 array): translation 
        q (Bx4x1 array): quarternion
    """
    x = pose[:, :-1, -1]
    rot = pose[:, :-1, :-1] 
    r = R.from_matrix(rot.detach().numpy())
    q = torch.from_numpy(r.as_quat())
    
    return x.float(), q.float()

def get_pose(x, q):
    """ Get 4x4 pose from x and q numpy vectors
    Args:
        x (3x1 array): translation 
        q (4x1 array): quarternion
    Returns:
        pose (4x4 array): transformation pose
    """
    pose = np.identity(4)
    r = R.from_quat(q)
    rot = r.as_matrix()
    pose[:-1, :-1] = rot
    pose[:-1, -1] = x
    
    return pose

def rotation_error(pose_error):
    """ Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """ Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

if __name__ == '__main__':
    loss = nn.L1Loss()
    tran = torch.tensor([0.6632,0.0047,0.0086])
    tran_l = torch.tensor([0.6755,0.0033,0.0138])
    rot = torch.tensor([ 0.4232, -0.0019,  0.0005,  0.0033])
    rot_l = torch.tensor([ 1, -2.4459e-04,  7.2978e-04,  1.9661e-03])
    rot_norm = rot/torch.norm(rot)
    pose = get_pose(tran, rot)
    pose_l = get_pose(tran_l, rot_l)
    print(pose)
    print(pose_l)
    err = np.linalg.inv(pose_l) @ pose
    print("rot err:",rotation_error(err), loss(rot_l, rot))
    print("tran err:",translation_error(err), loss(tran_l, tran))