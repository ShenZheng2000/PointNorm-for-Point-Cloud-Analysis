
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat


from pointnet2_ops import pointnet2_utils
from thop import profile
import numpy as np


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", 
                point_norm=False, reverse_point_norm=False, 
                local_mean=False, local_std=False, global_mean=False, global_std=False, 
                **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.point_norm = point_norm
        self.reverse_point_norm = reverse_point_norm

        self.local_mean = local_mean
        self.local_std = local_std
        self.global_mean = global_mean
        self.global_std = global_std

        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0

            self.affine_alpha_first = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta_first = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

            # self.affine_alpha_first_follow = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            # self.affine_beta_first_follow = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

            # hyperparams for second normalization
            self.affine_alpha_second = nn.Parameter(torch.ones([1,1,channel + add_channel]))
            self.affine_beta_second = nn.Parameter(torch.zeros([1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]

        # point normalization
        if self.normalize is not None:

            if self.point_norm is True:
                # normalize grouped_points wrt new_points 
                grouped_points, new_points = self.point_normalization_first(grouped_points, new_points, B)

            if self.reverse_point_norm is True:
                # normalize new_points wrt grouped_points (mean before)
                grouped_points, new_points = self.point_normalization_second_before(grouped_points, new_points, B)


        # NOTE: concat by last dimension to restore the channels reduced by index_points
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


    def point_normalization_first(self, grouped_points, new_points, B):
        # print("grouped_points shape", grouped_points.shape) # bs, points, k, channels
        # print("new_points shape", new_points.shape) # bs, points, channels

        if self.local_mean:
            mean = new_points.unsqueeze(dim=-2) # bs, points, 1, channels
        elif self.global_mean:
            mean = torch.mean(new_points.reshape(B,-1), dim = -1, keepdim = True).unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1, 1

        if self.local_std:
            std = torch.std((grouped_points-mean), dim = -2, keepdim=True) # bs, points, 1, channels
        elif self.global_std:
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1, 1

        grouped_points = (grouped_points-mean)/(std + 1e-5)
        grouped_points = self.affine_alpha_first*grouped_points + self.affine_beta_first

        return grouped_points, new_points


    def point_normalization_second_before(self, grouped_points, new_points, B):
        # NOTE: this is 'mean_before' version
        new_points = new_points.unsqueeze(dim=-2) # bs, points, 1, channels

        if self.local_mean: 
            mean = torch.mean(grouped_points, dim = -2).unsqueeze(dim = -2) # bs, points, 1, channels
        elif self.global_mean:
            mean = torch.mean(grouped_points.reshape(B,-1), dim = -1, keepdim = True).unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1, 1

        if self.local_std: 
            # std = torch.std((new_points-mean) + 1e-5, dim = -2, keepdim=True) # bs, points, 1, channels # NOTE: this will cause !!!
            std = torch.sqrt(torch.abs(new_points-mean) + 1e-5) # NOTE: Use 1e-5 to avoid torch.sqrt(0)

        elif self.global_std:
            std = torch.std((new_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1, 1

        new_points = (new_points - mean) / (std + 1e-5)
        # print("torch.min(new_points)", torch.min(new_points))
        new_points = new_points.squeeze(dim = -2)
        new_points = self.affine_alpha_second*new_points + self.affine_beta_second

        return grouped_points, new_points



    # def point_normalization_first_follow(self, grouped_points, new_points, B):
    #     mean = new_points.unsqueeze(dim=-2) # NOTE: suppose that use_xyz is false and that anchor is selected
    #     std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)

    #     grouped_points = (grouped_points-mean)/(std + 1e-5)
    #     grouped_points = self.affine_alpha_first_follow*grouped_points + self.affine_beta_first_follow

    #     return grouped_points, new_points


    # def point_normalization_second_after(self, grouped_points, new_points, B):
    #     # NOTE: this is 'mean_after' version
    #     mean = grouped_points
    #     new_points = new_points.unsqueeze(dim=-2)
    #     std = torch.std((new_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)

    #     new_points = (new_points - mean) / (std + 1e-5)
    #     new_points = torch.mean(new_points, dim = -2)
    #     new_points = self.affine_alpha_second*new_points + self.affine_beta_second

    #     return grouped_points, new_points
        


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=1, kernel_size=3, padding=1): 
        super(InvertedResidual, self).__init__()
        # print("kernel_size", kernel_size)
        # print("padding", padding)
        self.stride = stride
        # assert stride in [1, 2]
        assert stride == 1

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True, use_inv_res=False):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            if use_inv_res:
                operation.append(
                    InvertedResidual(out_channels, out_channels, expand_ratio=res_expansion)
                )
            else:
                operation.append(
                    ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                    bias=bias, activation=activation)
                )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu', 
                use_inv_res=False):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            if use_inv_res:
                operation.append(
                    InvertedResidual(channels, channels, expand_ratio=res_expansion)
                )
            else:
                operation.append(
                    ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
                )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], 
                 use_inv_res=False, 
                 point_norm=False, 
                 reverse_point_norm=False,
                 local_mean=False,
                 local_std=False,
                 global_mean=False,
                 global_std=False,
                 **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, 
                                            anchor_points, 
                                            kneighbor, 
                                            use_xyz, 
                                            normalize, 
                                            point_norm, 
                                            reverse_point_norm,
                                            local_mean,
                                            local_std,
                                            global_mean,
                                            global_std)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz, 
                                             use_inv_res=use_inv_res)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation, use_inv_res=use_inv_res)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x):
        # print("original x shape", x.shape)
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            # print("====================================>")
            # print("Before local, x shape {}, xyz shape {}".format(x.shape, xyz.shape))
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            # print("After local, x shape {}, xyz shape {}".format(x.shape, xyz.shape))
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        # print("x before pool ", x.shape)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        # print("x before class", x.shape)
        x = self.classifier(x)
        return x


def PointNorm_1_1(num_classes=40, embed_dim = 64, res_expansion = 1.0, 
                use_inv_res = False, 
                point_norm=False, 
                reverse_point_norm=False, 
                local_mean=False,
                local_std=False,
                global_mean=False,
                global_std=False,
                **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=embed_dim, groups=1, res_expansion=res_expansion,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[1, 1, 1, 1], pos_blocks=[1, 1, 1, 1],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], 
                   use_inv_res=use_inv_res, 
                   point_norm=point_norm, 
                   reverse_point_norm=reverse_point_norm,
                   local_mean=local_mean,
                   local_std=local_std,
                   global_mean=global_mean,
                   global_std=global_std,
                   **kwargs)


def PointNorm_2_2(num_classes=40, embed_dim = 64, res_expansion = 1.0, 
                use_inv_res = False, 
                point_norm=False, 
                reverse_point_norm=False, 
                local_mean=False,
                local_std=False,
                global_mean=False,
                global_std=False,
                **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=embed_dim, groups=1, res_expansion=res_expansion,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], 
                   use_inv_res=use_inv_res, 
                   point_norm=point_norm, 
                   reverse_point_norm=reverse_point_norm,
                   local_mean=local_mean,
                   local_std=local_std,
                   global_mean=global_mean,
                   global_std=global_std,
                   **kwargs)


def PointNorm_3_3(num_classes=40, embed_dim = 64, res_expansion = 1.0, 
                use_inv_res = False, 
                point_norm=False, 
                reverse_point_norm=False, 
                local_mean=False,
                local_std=False,
                global_mean=False,
                global_std=False,
                **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=embed_dim, groups=1, res_expansion=res_expansion,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[3, 3, 3, 3], pos_blocks=[3, 3, 3, 3],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], 
                   use_inv_res=use_inv_res, 
                   point_norm=point_norm, 
                   reverse_point_norm=reverse_point_norm,
                   local_mean=local_mean,
                   local_std=local_std,
                   global_mean=global_mean,
                   global_std=global_std,
                   **kwargs)


def PointNormTiny(num_classes=40, embed_dim = 32, res_expansion = 0.25, 
                    use_inv_res = False, 
                    point_norm=False, 
                    reverse_point_norm=False, 
                    local_mean=False,
                    local_std=False,
                    global_mean=False,
                    global_std=False,
                    **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=embed_dim, groups=1, res_expansion=res_expansion,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2], 
                   use_inv_res=use_inv_res, 
                   point_norm=point_norm, 
                   reverse_point_norm=reverse_point_norm,
                   local_mean=local_mean,
                   local_std=local_std,
                   global_mean=global_mean,
                   global_std=global_std,           
                   **kwargs)




if __name__ == '__main__':
    # Debug for whole model
    data = torch.rand(2, 3, 1024).cuda()
    print("===> testing PointNorm ...")
    model = PointNorm_2_2(use_inv_res=False, point_norm=True, reverse_point_norm=True,
                            local_mean=True, global_mean=False,
                            local_std=False, global_std=True).cuda() 
    out = model(data)
    print(out.shape)

