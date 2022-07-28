"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def rotate_pointcloud(pointcloud, theta = None):
    if theta == None:
        theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):  
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz1 = torch.from_numpy(xyz1).float()
        pc = torch.from_numpy(pc).float()
        pc = torch.mul(pc, xyz1)
        return pc

class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training', use_translate = False, use_rotate = False):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.use_translate = use_translate
        self.use_rotate = use_rotate
        # self.pointscale = PointcloudScale(0.9, 1.1) # NOTE: [0.9, 1.1] is recommended by PointNext

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        # For training only 
        if self.partition == 'training':
            if self.use_translate:
                pointcloud = translate_pointcloud(pointcloud)
            if self.use_rotate:
                pointcloud = rotate_pointcloud(pointcloud)
            # if self.use_scale:
            #     pointcloud = self.pointscale(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    pass
    # pointcloud = np.zeros((1024, 3))
    # pointscale = PointcloudScale(0.9, 1.1)
    # pointcloud = pointscale(pointcloud)
    # print("pointcloud shape", pointcloud.shape)

    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
