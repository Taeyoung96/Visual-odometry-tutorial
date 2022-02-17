# Visual-odometry-tutorial

You could see the Video lecture on [Youtube](https://youtu.be/VOlYuK6AtAE).  

The original code is available in [bitbucket](https://bitbucket.org/castacks/visual_odometry_tutorial/src/master/).  

I just fixed some directory in code, provided dataset and requirement.txt!  
I tested this repository on Ubuntu 18.04.  

## How to setup enviornmet  

You should have [Anaconda](https://www.anaconda.com/).  

**1. Make virtual environment using Anaconda**  

First, just clone this repository.  
```
git clone https://github.com/Taeyoung96/Visual-odometry-tutorial.git
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


