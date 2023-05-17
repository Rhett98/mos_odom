#!/usr/bin/env python3
# Parts of the code taken from pytorch3d (https://pytorch3d.readthedocs.io/)
from __future__ import division
import os
import sys
path = os.getcwd()
sys.path.append(path)
import torch
import kornia
import numpy as np

class CircularPad(torch.nn.Module):
    def __init__(self, padding=(1, 1, 0, 0)):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, input):
        return torch.nn.functional.pad(input=input, pad=self.padding, mode='circular')

def get_quaternion_from_transformation_matrix(matrix, device='cpu'):
    rotation_matrix = torch.tensor(matrix[:3, :3], device=device)
    return kornia.geometry.conversions.rotation_matrix_to_quaternion(rotation_matrix,order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ)

def get_translation_from_transformation_matrix(matrix, device='cpu'):
    return torch.tensor(matrix[:3, -1], device=device)

def quaternion_to_rot_matrix(quaternion):
    return kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ)

def euler_to_rot_matrix(input):
    return kornia.geometry.conversions.angle_axis_to_rotation_matrix(input)

def get_euler_from_transformation_matrix(matrix):
    rotation_matrix = torch.tensor(matrix[:3, :3])
    return kornia.geometry.conversions.rotation_matrix_to_angle_axis(rotation_matrix)
    
def get_transformation_matrix_quaternion(translation, quaternion, device='cpu'):
        rotation_matrix = quaternion_to_rot_matrix(quaternion)
        transformation_matrix = torch.zeros((rotation_matrix.shape[0], 4, 4), device=device)
        transformation_matrix[:, :3, :3] = rotation_matrix
        transformation_matrix[:, 3, 3] = 1
        transformation_matrix[:, :3, 3] = translation
        return transformation_matrix

def get_transformation_matrix_euler(translation, euler, device='cpu'):
        rotation_matrix = euler_to_rot_matrix(euler)
        transformation_matrix = torch.zeros((rotation_matrix.shape[0], 4, 4), device=device)
        transformation_matrix[:, :3, :3] = rotation_matrix
        transformation_matrix[:, 3, 3] = 1
        transformation_matrix[:, :3, 3] = translation
        return transformation_matrix

def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)
