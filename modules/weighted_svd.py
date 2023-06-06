import torch
import torch.nn as nn

def weighted_procrustes(
    src_points,
    ref_points,
    # weights=None,
    # weight_thresh=0.0,
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
    
    # # init weights
    # if weights is None:
    #     weights = torch.ones_like(src_points[:, :, 0])
    # weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    # weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    # weights = weights.unsqueeze(2)  # (B, N, 1)
    
    # one batch
    # print(src_points)
    batch_transformation = torch.Tensor()
    for i in range(0, batch_size):
        # weight = weights[i]
        # src_centroid = torch.sum(src_points[i] * weight, dim=0, keepdim=True)  # (1, 3)
        # ref_centroid = torch.sum(ref_points[i]* weight, dim=0, keepdim=True)  # (1, 3)
        # delete zero point
        mask = torch.all(src_points[i] == 0, dim=1)
        valid_mask = ~ mask
        valid_src_points = src_points[i][valid_mask]
        valid_ref_points = ref_points[i][valid_mask]
        # print(valid_src_points.shape)
        # weight = weight[valid_mask]
        # downsample
        step = 5
        num_samples = int(valid_src_points.shape[0]/step) 
        # TODO: 增加match score， topK筛选
        # 均匀采样
        indices = torch.linspace(0, valid_src_points.size(0)-1, num_samples).long()
        # 使用索引进行随机采样
        # indices = torch.randperm(valid_src_points.size(0))[:num_samples].long()
        valid_src_points = valid_src_points[indices, :]
        valid_ref_points = valid_ref_points[indices, :]
        # print(valid_src_points.shape)
        # print("***")
        src_centroid = torch.sum(valid_src_points, dim=0, keepdim=True)/valid_src_points.shape[0]  # (1, 3)
        ref_centroid = torch.sum(valid_ref_points, dim=0, keepdim=True)/valid_ref_points.shape[0]  # (1, 3)
        # print(src_centroid)
        # print(valid_src_points, src_centroid)
        # print(valid_ref_points , ref_centroid)
        src_points_centered = valid_src_points - src_centroid  # (N, 3)
        ref_points_centered = valid_ref_points - ref_centroid  # (N, 3)
        # print(src_points_centered)
        # print(ref_points_centered)
        # H = USV^T
        # H = src_points_centered.permute(1, 0) @ (weight * ref_points_centered)
        H = src_points_centered.permute(1, 0) @ (ref_points_centered)
        # print(H)
        H[torch.isnan(H)] = 0.0
        # print(H)
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
            # weights=weights,
            # weight_thresh=self.weight_thresh,
            eps=self.eps,
        )
        
def transform(PC, matrix):
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
     
if __name__ == '__main__':

    import numpy as np

    # 指定KITTI数据集中点云二进制文件的路径
    file_path1 = "/home/yu/Resp/dataset/sequences/24/velodyne/000010.bin"
    file_path2 = "/home/yu/Resp/dataset/sequences/24/velodyne/000011.bin"
    # 使用NumPy加载二进制文件
    point_cloud_data1 = np.fromfile(file_path1, dtype=np.float32).reshape(1, -1, 4)
    point_cloud_data2 = np.fromfile(file_path1, dtype=np.float32).reshape(1, -1, 4)

    # 将点云数据转换为PyTorch张量
    x = torch.from_numpy(point_cloud_data1[:,:,:3])
    y = torch.from_numpy(point_cloud_data2[:,:,:3])
    xx = torch.randn(2,5,64,2048)
    # print(x.shape)
    gt = torch.tensor( [[[ 9.99759159e-01, -2.19067345e-02, -8.70411443e-04, 1.66291837e-01],
                        [ 2.19070716e-02,  9.99760511e-01, -2.03509981e-04, 1.36440622e-02],
                        [ 8.74661754e-04,  1.84397055e-04,  9.99999581e-01, 2.31188809e-03],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]],
                        [[ 9.99759159e-01, -2.19067345e-02, -8.70411443e-04, 1.66291837e-01],
                        [ 2.19070716e-02,  9.99760511e-01, -2.03509981e-04, 1.36440622e-02],
                        [ 8.74661754e-04,  1.84397055e-04,  9.99999581e-01, 2.31188809e-03],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]])
    # x= torch.randn(1,3,64,2048)
    # print(x.shape)
    # print(gt.shape)
    # # mm = WeightedProcrustes()
    # # a = mm(x, y)
    a = transform(xx, gt)
    print(a.shape)