import torch
import torch.nn as nn

def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-3,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    batch_size, N, C = src_points.size()
    
    # init weights
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)
    
    # one batch
    batch_transformation = torch.Tensor()
    for i in range(0, batch_size):
        weight = weights[i]
        src_centroid = torch.sum(src_points[i] * weight, dim=0, keepdim=True)  # (1, 3)
        ref_centroid = torch.sum(ref_points[i]* weight, dim=0, keepdim=True)  # (1, 3)
        mask = torch.all(src_points[i] == 0, dim=1)
        valid_mask = ~ mask
        valid_src_points = src_points[i][valid_mask]
        valid_ref_points = ref_points[i][valid_mask]
        weight = weight[valid_mask]
        src_points_centered = valid_src_points - src_centroid  # (N, 3)
        ref_points_centered = valid_ref_points - ref_centroid  # (N, 3)

        # H = USV^T
        H = src_points_centered.permute(1, 0) @ (weight * ref_points_centered)

        U, _, V = torch.svd(H.cpu())  
        Ut, V = U.transpose(0, 1).cuda(), V.cuda()
        eye = torch.eye(3).cuda()
        eye[-1, -1] = torch.sign(torch.det(V @ Ut))
        R = V @ eye @ Ut
        t = ref_centroid.permute(1, 0).cuda() - R @ src_centroid.permute(1, 0).cuda()
        t = t.squeeze(1)
        
        transform = torch.eye(4).cuda()
        transform[:3, :3] = R
        transform[:3, 3] = t
        if i == 0:
            batch_transformation = transform.unsqueeze(0).clone()
        else:
            batch_transformation = torch.cat([batch_transformation.clone(), transform.unsqueeze(0).clone()], dim=0)

    return batch_transformation


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.0, eps=1e-5):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        # self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
        )
        
if __name__ == '__main__':

    # x = torch.Tensor([[[0, 0, 0], [0, 1, 2],[0, 3, 4],[1, 7, 9]],[[0, 5, 0], [0, 0, 0],[6, 7, 8],[0, 0, 0]]])  # 原始张量
    # B, N, C = x.size()
    # full_y = torch.Tensor()
    # for i in range(0,B):
    #     mask = torch.all(x[i] == 0, dim=1)
    #     valid_mask = ~ mask
    #     y = x[i][valid_mask]
    #     if i == 0:
    #         full_y = y.unsqueeze(0).clone()
    #     else:
    #         full_y = torch.cat([full_y, y.unsqueeze(0).clone()], dim=0)
    x = torch.randn(4,1000,3) 
    y = torch.randn(4,1000,3)
    mm = WeightedProcrustes()
    a = mm(x, y)
    print(a)