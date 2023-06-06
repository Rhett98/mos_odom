import torch

def transformPC(PC, matrix):
    """use matrix to transform current pc

    Args:
        PC (tensor): (B, 3, H, W)
        matrix (tensor): (B, 4, 4)
    """
    B, C, H, W = PC.shape
    PC = PC.view(B, -1, C)
    padding_column = torch.ones(B, H*W, 1, dtype=PC.dtype, device=PC.device)
    PC = torch.cat((PC, padding_column), dim=2).permute(0, 2, 1)
    output = torch.bmm(matrix, PC)
    return output[:, :3, :].view(B, C, H, W)

def mul_q_point(q_a, q_b):
    batch_size = q_a.shape[0]
    q_a = q_a.view(batch_size, 1, 4)
    q_b = q_b.view(batch_size, 1, 4)

    q_result_0 = (q_a[:, :, 0] * q_b[:, :, 0]) - (q_a[:, :, 1] * q_b[:, :, 1]) - (q_a[:, :, 2] * q_b[:, :, 2]) - (q_a[:, :, 3] * q_b[:, :, 3])
    q_result_0 = q_result_0.view(batch_size, -1, 1)

    q_result_1 = (q_a[:, :, 0] * q_b[:, :, 1]) + (q_a[:, :, 1] * q_b[:, :, 0]) + (q_a[:, :, 2] * q_b[:, :, 3]) - (q_a[:, :, 3] * q_b[:, :, 2])
    q_result_1 = q_result_1.view(batch_size, -1, 1)

    q_result_2 = (q_a[:, :, 0] * q_b[:, :, 2]) - (q_a[:, :, 1] * q_b[:, :, 3]) + (q_a[:, :, 2] * q_b[:, :, 0]) + (q_a[:, :, 3] * q_b[:, :, 1])
    q_result_2 = q_result_2.view(batch_size, -1, 1)

    q_result_3 = (q_a[:, :, 0] * q_b[:, :, 3]) + (q_a[:, :, 1] * q_b[:, :, 2]) - (q_a[:, :, 2] * q_b[:, :, 1]) + (q_a[:, :, 3] * q_b[:, :, 0])
    q_result_3 = q_result_3.view(batch_size, -1, 1)

    q_result = torch.cat([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1).view(batch_size, -1)

    return q_result   ##  B 4

def inv_q(q):
    q_2 = torch.sum(q * q, dim=-1, keepdim=True) + 1e-10
    q_ = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
    q_inv = q_ / q_2

    return q_inv

def quatt2T(t, q):
    t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    
    c1 = torch.tensor([1.0], device=q.device)
    add = torch.tensor([0.0, 0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], -1)
    T = torch.cat([(c1 - (yY + zZ)).unsqueeze(1),
                   (xY - wZ).unsqueeze(1),
                   (xZ + wY).unsqueeze(1),
                   t0.unsqueeze(1)], dim=1)

    T = torch.cat([T,
                   (xY + wZ).unsqueeze(1),
                   (c1 - (xX + zZ)).unsqueeze(1),
                   (yZ - wX).unsqueeze(1),
                   t1.unsqueeze(1)], dim=1)

    T = torch.cat([T,
                   (xZ - wY).unsqueeze(1),
                   (yZ + wX).unsqueeze(1),
                   (c1 - (xX + yY)).unsqueeze(1),
                   t2.unsqueeze(1)], dim=1)

    T = torch.cat([T.view(T.shape[0],3,4), add.unsqueeze(1)], dim=1)

    return T

def T2quat_tran(R):
    # 计算四元数的分量
    w = torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)

    # 归一化四元数
    quaternion = torch.stack([w, x, y, z], dim=1)
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True)
    tran = R[:, :3, 3]
    return quaternion, tran

if __name__=='__main__':
    q = torch.tensor([[0.7071, 0.7071, 0.0000, 0.0000],
                  [0.8660, 0.0000, 0.5000, 0.0000],
                  [0.8660, 0.0000, 0.5000, 0.0000]])
    t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [4.0, 5.0, 6.0]])

    q_a = torch.tensor([[0.7071, 0.7071, 0.0000, 0.0000],
                    [0.8660, 0.0000, 0.5000, 0.0000]])
    q_b = torch.tensor([[0.5000, 0.5000, 0.5000, 0.5000],
                        [0.0000, 0.7071, 0.7071, 0.0000]])

    x = torch.tensor([[[ 1.0000,  0.0000,  0.0000,  1.0000],
                    [ 0.0000,  0.0000, -1.0000,  2.0000],
                    [ 0.0000,  1.0000,  0.0000,  3.0000],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]],

                    [[ 0.5000,  0.0000,  0.8660,  4.0000],
                    [ 0.0000,  1.0000,  0.0000,  5.0000],
                    [-0.8660,  0.0000,  0.5000,  6.0000],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]],

                    [[ 0.5000,  0.0000,  0.8660,  4.0000],
                    [ 0.0000,  1.0000,  0.0000,  5.0000],
                    [-0.8660,  0.0000,  0.5000,  6.0000],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    print(quatt2T(t, q))
    print(T2quat_tran(x))