# Visual-odometry-tutorial

You could see the Video lecture on [Youtube](https://youtu.be/VOlYuK6AtAE).  

The original material is available in [bitbucket](https://bitbucket.org/castacks/visual_odometry_tutorial/src/master/).  

I just fixed some directory in code, provided dataset and requirement.txt!  
I tested this repository on Ubuntu 18.04.  

## Contents  
- [How to setup enviornmet](#how-to-setup-enviornmet)  
- [How to run this repository](#how-to-run-this-repository)  
- [Contact](#contact)

## How to setup enviornmet  

You should have [Anaconda](https://www.anaconda.com/).  

**1. Make virtual environment using Anaconda**  

First, just clone this repository.  
```
git clone https://github.com/Taeyoung96/Visual-odometry-tutorial.git  
cd Visual-odometry-tutorial
```
Make virtual environment with python version 3.8.  

```
conda create -n vo_tutorial python=3.8
conda activate
```

If you follow above command lines, you enter the virtual environment.  
Then install required python packages.  
```
pip install -r requirements.txt
```  

**2. Prepare the dataset**  

In this tutorial, there are using two famous Dataset.  
- [KITTI dataset - Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)  
- [TUM RGB-D dataset - RGB-D SLAM Dataset and Benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset)  

But these datasets are very large.  
So I have prepared the minimal dataset for this tutorial on Google drive.  
- [KITTI dataset - 09 sequence](https://drive.google.com/file/d/1n_ZOaHSQ2Geul4MDV-3ejT6Z44ODnbcr/view?usp=sharing)  
- [TUM RGB-D dataset - freiburg2_desk sequence](https://drive.google.com/file/d/1owc_uS_eKNrfVujWkyRE7VMATU5Q67l2/view?usp=sharing)  

**These datasets have their licences, so don't use these datasets for commercial uses!**  

If you want to match the default directory path, unzip it in the following path.  
In `data/` folder, there are two folders named `kitti-odom/` and `tum/`.  
`.txt` file exists in each folder, and you can unzip the dataset in the same path as `.txt` file.  

If you are finished, the dataset path is as follows.  
- KITTI dataset - 09 sequence : `~[YOUR_DIR]/Visual-odometry-tutorial/data/kitti-odom`.  
- TUM dataset - freiburg2_desk sequence : `~[YOUR_DIR]/Visual-odometry-tutorial/data/tum`.  


## How to run this repository  

There are 3 python files that you can run it.  

In `cv_basics/` folder there are `epipolar.py` and `feature_matching.py`.  

- `feature_matching.py` : Visualize the result of feature matching using two consecutive images.  
  We extracted keypoints using ORB features and did feature math with brute-force matching algorthm.  
  
- `epipolar.py` : Visualize the epipolar line using two consecutive images.  

When you want to execute `feature_matching.py` on default directory (`~/Visual-odometry-tutorial`)  
follow codes below.  
```
cd cv_basics  
python feature_matching.py
```

### `feature_matching.py` Result  

<p align="center"><img src="/result/feature_matching.png" width = "600" ></p>  

When you want to execute `epipolar.py` on default directory (`~/Visual-odometry-tutorial`)  
follow codes below.  
```
cd cv_basics  
python epipolar.py
```

### `epipolar.py` Result  

<p align="center"><img src="/result/epipolar.png" width = "600" ></p>  

In `visual-odometry/` folder there is `vo.py`  

When you want to execute `vo.py` on default directory (`~/Visual-odometry-tutorial`)  
follow codes below.  

When you finish the code, you could get **Trajectory.png and Trajectory.txt**!.  

```
cd visual-odometry 
```

There are 3 arguments, when run the code below.  

- `'--data_dir_root'` : Set the dataset path.  
  (When you follow **2. Prepare the dataset** you could use deafult path. default='../data/')  
- `'--dataset_type'` : Decide which dataset to use. (default='TUM') 
- `'--len_trajMap'` : Specifies the size of the trajectory visualized window. (default=700)  

If you want to run code with **KITTI dataset**,  
```
python vo.py --dataset_type='KITTI'
```
### `vo.py` Result (KITTI)  

<p align="center"><img src="/result/vo_kitti.png" width = "700" ></p>  

Or if you want to run code with **TUM dataset**,  
```
python vo.py --dataset_type='TUM'
```
### `vo.py` Result (TUM)  

<p align="center"><img src="/result/vo_tum.png" width = "700" ></p>  

**The results are not accurate. This is because we are not doing any optimizations, we are just estimating.**  

## Contact  

If you have any question, feel free to send an email.  

- **TaeYoung Kim** : tyoung96@yonsei.ac.kr   
