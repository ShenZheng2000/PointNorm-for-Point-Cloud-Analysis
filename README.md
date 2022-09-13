# PointNorm-for-Point-Cloud-Analysis
This is the official Pytorch implementation of our paper "PointNorm: Dual Normalization is All You Need for Point Cloud Analysis"

## Updates
- 2022/9/15: We have revised the paper and have updated the manuscript for arxiv. Here are an non-exhaust list for the changes. 
  - Change the format from CVF to IEEE style
  - Add the experiments for S3DIS semantic segmentation
  - Add Optimization Landscape Analysis
  - Merge the figure for shape classfication and semantic segmentaion.
  - Reformat the mathematical notations and deviations.
  - Update the instruction on Readme (DOING).

- 2022/8/5: Code for visualizing the loss landscape and the segmentation output has been uploaded. Detailed instructions will be included in this readme file next week. 

- 2022/7/28: Code for classifcation and segmentation have been uploaded. 

- 2022/7/24: Basic instructions have been given. The complete code will be uploaded next week. 

- 2022/7/15: Arxiv Link is available at: https://arxiv.org/abs/2207.06324

- 2022/7/13: Stay tuned. Code will be uploaded soon.

## Abstract
Point cloud analysis is challenging due to the irregularity of the point cloud data structure. Existing works typically employ the ad-hoc sampling-grouping operation of PointNet++, followed by sophisticated local and/or global feature extractors for leveraging the 3D geometry of the point cloud. Unfortunately, those intricate hand-crafted model designs have led to poor inference latency and performance saturation in the last few years. In this paper, we point out that the classical sampling-grouping operations on the irregular point cloud cause learning difficulty for the subsequent MLP layers. To reduce the irregularity of the point cloud, we introduce a DualNorm module after the sampling-grouping operation. The DualNorm module consists of Point Normalization, which normalizes the grouped points to the sampled points, and Reverse Point Normalization, which normalizes the sampled points to the grouped points. The proposed PointNorm utilizes local mean and global standard deviation to benefit from both local and global features while maintaining a faithful inference speed. Experiments on point cloud classification show that we achieved state-of-the-art accuracy on ModelNet40 and ScanObjectNN datasets. We also generalize our model to point cloud part segmentation and demonstrate competitive performance on the ShapeNetPart dataset.

## Intuition
![intuition](https://github.com/ShenZheng2000/PointNorm-for-Point-Cloud-Analysis/blob/main/Figures/Model_Head.png)

## Workflow
![workflow](https://github.com/ShenZheng2000/PointNorm-for-Point-Cloud-Analysis/blob/main/Figures/Model_WorkFlow.png)

## Dependencies
- Python 3.8
- Pytorch 1.9
- CUDA 11
- torchvision 0.10

## Install
  * Follow the step below to install the required packages. 
    ```
    # clone this repository
    git clone https://github.com/ShenZheng2000/PointNorm-for-Point-Cloud-Analysis.git
    cd PointNorm-for-Point-Cloud-Analysis

    # install required packages
    pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
    pip install pointnet2_ops_lib/.
    ```

## Datasets
* ModelNet40 and ScanObjectNN will be automatically downloaded with the training command. 

* ShapeNetPart needs to be prepared in the following way.
  ```
  cd part_segmentation
  mkdir data
  cd data
  wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
  unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
  ```

## Classification on ModelNet40
* The results at ModelNet40 are volatile. Running the same model with different seeds leads to significantly different scores. To alleviate the randomness, you may consider training the model with different seeds for 2-4 times and report the average accuracy as your score. 

* Different methods use different voting strategies to promote their classification accuracy (see the paper for details). For fairness, we do not use any voting strategy in our experiments.

* With all this in mind, you can run the following lines in terminal.
  ```
  cd classification_ModelNet40

  # For PointNorm (the full-sized model), run:
  python main.py --model PointNorm_2_2

  # For PointNorm-Tiny (the lightweight model), run:
  python main.py --model PointNormTiny --embed_dim 32 --res_expansion 0.25
  ```

## Classification on ScanObjectNN

* Run the following lines in terminal.
  ```
  cd classification_ScanObjectNN

  # For PointNorm (the full-sized model), run:
  python main.py --model PointNorm_2_2 --point_norm True --reverse_point_norm True --local_mean True --global_std True

  # For PointNorm-Tiny (the lightweight model), run:
  python main.py --model PointNormTiny --point_norm True --reverse_point_norm True --local_mean True --global_std True --embed_dim 32 --res_expansion 0.25
  ```

## Part segmentation on ShapeNetPart

* Run the following lines in terminal.
  ```
  cd part_segmentation

  # For PointNorm (the full-sized model), run:
  python main.py --model PointNorm --point_norm True --reverse_point_norm True --local_mean True --global_std True

  # For PointNorm-Tiny (the lightweight model), run:
  python main.py --model PointNormTiny --point_norm True --reverse_point_norm True --local_mean True --global_std True --embed_dim 32 --res_expansion 0.25
  ```
  
## Semantic Segmentation

* Due to a limited amount of GPUs, we only examine PointNet++ (w/ DualNorm) on semantic segmentation tasks. If we have more GPUs in the future, we will try PointNorm. 

* Clone the following repository
  ```
  git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
  ```
  
* Do the following changes with my updated version. 
  - [train_semseg.py](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_semseg.py) to train_semseg.py
  - [test_semseg.py](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/test_semseg.py) to test_semseg.py
  - [models/pointnet2_sem_seg.py](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_sem_seg.py) to pointnet2_sem_seg.py
  - [models/pointnet2_utils.py](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py) to pointnet2_utils.py
  - Add [data_augmentation.py]
  
* Download [S3DIS](http://buildingparser.stanford.edu/dataset.html) and save in `./data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`

* Process S3DIS using the following command (This process might takes 10-20 minutes, so please be patient.)
  ```
  cd data_utils
  python collect_indoor3d_data.py
  ```

* Find you data folder in `./data/stanford_indoor3d/`

* Go to `train_semseg.py` and `test_semseg.py`, change root to the path of the data folder.

* For training, run:
  ```
  python train_semseg.py --model pointnet2_sem_seg --log_dir {name_for_your_log_dir}
  ```
 
* For testing, run
  ```
  python test_semseg.py --log_dir {name_for_your_log_dir} --visual # with visualization
  ```

* Double check the`.obj` files in `./log/sem_seg/{name_for_your_log_dir}/visual/`. 

* Visualize the semantic segmentation output using [MeshLab](https://www.meshlab.net/).


## TODO-List
- [x] Upload readme with basic instructions
- [x] Upload files for classfication at ScanObjectNN
- [x] Upload files for segmentation at ShapeNetPart
- [x] Upload files for classfication at ModelNet40
- [x] Upload files for visualization
- [x] Update the code for semantic segmentation
- [ ] Update the code for Standard Deviation Analysis and Optimization Landscape Analysis. 
- [ ] Update readme

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
This repository is heavily based upon [PointMLP](https://github.com/ma-xu/pointMLP-pytorch) and [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) . We appreciate their excellent work and kind sharing.
