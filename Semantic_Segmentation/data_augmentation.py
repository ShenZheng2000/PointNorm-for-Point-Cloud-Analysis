import torch
import random
import numpy as np
import collections
from scipy.linalg import expm, norm

# color dropping

class ChromaticDropGPU(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.color_drop = color_drop

    def __call__(self, data):
        colors_drop = torch.rand(1) < self.color_drop
        if colors_drop:
            data[:, :3] = 0
        return data

# color autocontrast

class ChromaticAutoContrast(object):
    def __init__(self, randomize_blend_factor=True, blend_factor=0.5, **kwargs):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, data, **kwargs):
        if random.random() < 0.2:
            # to avoid chromatic drop problems
            if data.mean() <= 0.1:
                return data
            lo = data.min(1, keepdims=True)[0]
            hi = data.max(1, keepdims=True)[0]
            scale = 255 / (hi - lo)
            contrast_feats = (data - lo) * scale
            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            data = (1 - blend_factor) * data + blend_factor * contrast_feats
        return data

# normalize

class ChromaticNormalize(object):
    def __init__(self,
                 color_mean=[0.5136457, 0.49523646, 0.44921124],
                 color_std=[0.18308958, 0.18415008, 0.19252081],
                 **kwargs):
        self.color_mean = torch.from_numpy(np.array(color_mean)).to(torch.float32)
        self.color_std = torch.from_numpy(np.array(color_std)).to(torch.float32)

    def __call__(self, data):

        device = data.device
        if data[:, :3,:].max() > 1:
            data[:, :3,:] /= 255.
        # print(data.size(),data[:,:3,:].size(),self.color_mean.size())
        # data[:, :3,:] = (data[:, :3,:] - self.color_mean.to(device)) / self.color_std.to(device)
        for i in range(3):
            data[:,i,:] = data[:,i,:] - self.color_mean.to(device)[i] / self.color_std.to(device)[i]
        return data

# rotation

class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        for i in data:
            i[:3,:] =  rot_mat.T @ i[:3,:]
        return data

if __name__ == "__main__":
    data = torch.rand(6, 9, 2048)

    # # color drop
    # color_drop = ChromaticDropGPU()
    # out = color_drop(data)

    # color autocontrast
    color_contrast = ChromaticAutoContrast()
    out = color_contrast(data)
    print("out size", out.shape)

    # # normalize
    # normalize = ChromaticNormalize()
    # out = normalize(data)

    # # rotation
    # rotation = PointCloudRotation()
    # out = rotation(data)

    print(out.size())

