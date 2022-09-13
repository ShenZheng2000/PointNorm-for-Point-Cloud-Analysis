import torch.nn as nn
import torch.nn.functional as F
# from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
# from pointnet2_utils_ori import PointNetSetAbstraction,PointNetFeaturePropagation
import torch

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 9, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 64, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 128, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 256, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points



def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

class get_loss(nn.Module):
    def __init__(self, epsilon:float=0.1):
        super(get_loss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, target, trans_feat, weight):
        n = pred.size()[-1]
        loss = -pred.sum(dim=-1).mean()
        nll = F.nll_loss(pred, target, weight)
        return linear_combination(loss/n, nll, self.epsilon)


if __name__ == '__main__':

    # NOTE: code for debugging
    # model = get_model(13)
    # xyz = torch.rand(6, 9, 2048)
    # (model(xyz)

    # GLS = get_loss_smooth(epsilon=0.1).cuda()
    # GL = get_loss().cuda()
    # pred = torch.rand(65536, 13).cuda()
    # target = torch.rand(65536).long().cuda()
    # trans_feat = torch.rand(16, 512, 16).cuda()
    # weight = torch.rand(13).cuda()

    # print("GLS result is", GLS(pred, target, trans_feat, weight))
    # print("GL result is", GL(pred, target, trans_feat, weight))

    # NOTE: Codes for calculating flop and params
    import torch
    import fvcore.nn
    import fvcore.common
    from fvcore.nn import FlopCountAnalysis

    model = get_model(13)
    model.eval()
    # model = deit_tiny_patch16_224()

    inputs = (torch.randn((1,9,4096)))
    k = 1024.0
    flops = FlopCountAnalysis(model, inputs).total()
    print(f"Flops : {flops}")
    flops = flops/(k**3)
    print(f"Flops : {flops:.3f}G")
    params = fvcore.nn.parameter_count(model)[""]
    print(f"Params : {params}")
    params = params/(k**2)
    print(f"Params : {params:.3f}M")