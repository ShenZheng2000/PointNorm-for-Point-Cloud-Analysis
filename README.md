# PointNorm-for-Point-Cloud-Analysis
This is the official Pytorch implementation of our paper "PointNorm: Normalization is All You Need for Point Cloud Analysis"

## Updates
- 2022/7/26: The code for classifcation and segmentation have been uploaded. 

- 2022/7/24: Basic instructions have been given. The complete code will be uploaded next week. 

- 2022/7/15: Arxiv Link is available at: https://arxiv.org/abs/2207.06324

- 2022/7/13: Stay tuned. Code will be uploaded soon.

## Abstract
Point cloud analysis is challenging due to the irregularity of the point cloud data structure. Existing works typically employ the ad-hoc sampling-grouping operation of PointNet++, followed by sophisticated local and/or global feature extractors for leveraging the 3D geometry of the point cloud. Unfortunately, those intricate hand-crafted model designs have led to poor inference latency and performance saturation in the last few years. In this paper, we point out that the classical sampling-grouping operations on the irregular point cloud cause learning difficulty for the subsequent MLP layers. To reduce the irregularity of the point cloud, we introduce a DualNorm module after the sampling-grouping operation. The DualNorm module consists of Point Normalization, which normalizes the grouped points to the sampled points, and Reverse Point Normalization, which normalizes the sampled points to the grouped points. The proposed PointNorm utilizes local mean and global standard deviation to benefit from both local and global features while maintaining a faithful inference speed. Experiments on point cloud classification show that we achieved state-of-the-art accuracy on ModelNet40 and ScanObjectNN datasets. We also generalize our model to point cloud part segmentation and demonstrate competitive performance on the ShapeNetPart dataset.

## Dependencies
- Python 3.8
- Pytorch 1.9
- CUDA 11
- torchvision 0.10

## Install
```
# clone this repository
git clone https://github.com/ShenZheng2000/PointNorm-for-Point-Cloud-Analysis.git
cd PointNorm-for-Point-Cloud-Analysis

# install required packages
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```

## Datasets
ModelNet40 and ScanObjectNN will be automatically downloaded with the training command. 

ShapeNetPart needs to be prepared in the following way.

```
cd part_segmentation
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

## Classification on ModelNet40
* The result at ModelNet40 are volatile. Running the same model with different seeds leads to significantly different results. To alleviate the randomness, you may consider training the model with different seeds for 2-4 times and report the average accuracy as your score. 

* We note that different methods use different voting strategies to promote their classification accuracy (see the paper for details). For fairness, we do not use any voting strategy in the experiments.

Run the following lines in terminal.
```
cd classification_ModelNet40

# For PointNorm (the full-sized model), run:
python main.py --model PointNorm_2_2

# For PointNorm-Tiny (the lightweight model), run:
python main.py --model PointNormTiny --embed_dim 32 --res_expansion 0.25
```

## Classification on ScanObjectNN

Run the following lines in terminal.

```
cd classification_ScanObjectNN

# For PointNorm (the full-sized model), run:
python main.py --model PointNorm_2_2 --point_norm True --reverse_point_norm True --local_mean True --global_std True

# For PointNorm-Tiny (the lightweight model), run:
python main.py --model PointNormTiny --point_norm True --reverse_point_norm True --local_mean True --global_std True --embed_dim 32 --res_expansion 0.25
```

## Part segmentation on ShapeNetPart

Run the following lines in terminal.

```
cd part_segmentation

# For PointNorm (the full-sized model), run:
python main.py --model PointNorm --point_norm True --reverse_point_norm True --local_mean True --global_std True

# For PointNorm-Tiny (the lightweight model), run:
python main.py --model PointNormTiny --point_norm True --reverse_point_norm True --local_mean True --global_std True --embed_dim 32 --res_expansion 0.25
```

## TODO-List
- [x] Upload readme with basic instructions
- [x] Upload files for classfication at ScanObjectNN
- [x] Upload files for segmentation at ShapeNetPart
- [x] Upload files for classfication at ModelNet40
- [ ] Upload files for visualization
- [ ] Upload pretrained checkpoints
- [ ] Polish readme

## BibTeX
Please cite our paper if you find this repository helpful.
```
@article{zheng2022pointnorm,
  title={PointNorm: Normalization is All You Need for Point Cloud Analysis},
  author={Zheng, Shen and Pan, Jinqian and Lu, Changjie and Gupta, Gaurav},
  journal={arXiv preprint arXiv:2207.06324},
  year={2022}
}
```

## Contact
shenzhen@andrew.cmu.edu

## Acknowledgment
This repository is heavily based upon [PointMLP](https://github.com/ma-xu/pointMLP-pytorch). We sincerely thank the PointMLP's authors for their excellent work and kind sharing.
