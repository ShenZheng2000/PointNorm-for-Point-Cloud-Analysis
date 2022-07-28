import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import json
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# def load_data(partition):
#     all_data = []
#     all_label = []
#     for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
#         f = h5py.File(h5_name)
#         data = f['data'][:].astype('float32')
#         label = f['label'][:].astype('int64')
#         f.close()
#         all_data.append(data)
#         all_label.append(label)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


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


# def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
#     N, C = pointcloud.shape
#     pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
#     return pointcloud


# # =========== ModelNet40 =================
# class ModelNet40(Dataset):
#     def __init__(self, num_points, partition='train'):
#         self.data, self.label = load_data(partition)
#         self.num_points = num_points
#         self.partition = partition  # Here the new given partition will cover the 'train'

#     def __getitem__(self, item):  # indice of the pts or label
#         pointcloud = self.data[item][:self.num_points]
#         label = self.label[item]
#         if self.partition == 'train':
#             # pointcloud = pc_normalize(pointcloud)  # you can try to add it or not to train our model
#             pointcloud = translate_pointcloud(pointcloud)
#             np.random.shuffle(pointcloud)  # shuffle the order of pts
#         return pointcloud, label

#     def __len__(self):
#         return self.data.shape[0]


# =========== ShapeNet Part =================
class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False, use_translate = False, use_rotate = False):
        # NOTE: add some params
        self.split = split
        self.use_translate = use_translate
        self.use_rotate = use_rotate

        self.npoints = npoints
        # self.root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal' # NOTE: for debug only
        self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        # If train or trainval, use the following augmentation techniques
        if self.split == 'trainval' or self.split == 'train':

            # Add translation
            if self.use_translate:
                point_set = translate_pointcloud(point_set)

            # Add rotation
            if self.use_rotate:
                point_set = rotate_pointcloud(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    train = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    test = PartNormalDataset(npoints=2048, split='test', normalize=False)
    for data, label, _, _ in train:
        print(data.shape)
        print(label.shape)
