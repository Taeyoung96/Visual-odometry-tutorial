import os
import numpy as np
import time
import cv2
import argparse
from glob import glob
from pose_evaluation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_root", type=str, default='../data/',
                    help="dataset root")
parser.add_argument("--dataset_type", type=str, default='TUM', choices=['KITTI', 'TUM'],
                    help="datasetset type")
parser.add_argument("--len_trajMap", type=int, default=700,
                    help="size of the trajectory map")
args = parser.parse_args()

# dataset directory
if args.dataset_type == 'TUM':
    seq = 'freiburg2_desk' 
    img_data_dir = args.data_dir_root + 'tum/rgbd_dataset_' + seq + '/rgb/'

elif args.dataset_type == 'KITTI':
    seq = '09'
    img_data_dir = args.data_dir_root + 'kitti-odom/' + seq + '/image_2/'

# intrinsic parameters:
if args.dataset_type == 'TUM':
    width = 640
    height = 480
    fx, fy, cx, cy = [535.4, 539.2, 320.1, 247.6]

elif args.dataset_type == 'KITTI':
    width = 1241.0
    height = 376.0
    fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]


if __name__ == "__main__":
    # define the output pose file
    trajMap = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)
    if args.dataset_type == 'TUM':
        out_pose_file = './' + 'TUM' + '-' + seq + '-traj_est.txt'
    elif args.dataset_type == 'KITTI':
        out_pose_file = './' + 'KITTI' + '-' + seq + '-traj_est.txt'

    # get the image list in the directory
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)

    for i in range(num_frames):
        start_time = time.time()
        # Load image
        curr_img = cv2.imread(img_list[i], 0)

        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0])
        else:
            prev_img = cv2.imread(img_list[i-1], 0)

            #====================== Use ORB Feature to do feature matching =====================#
            # Create ORB features
            orb = cv2.ORB_create(nfeatures=6000)

            # Find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(prev_img, None)
            kp2, des2 = orb.detectAndCompute(curr_img, None)

            # use brute-force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match ORB descriptors
            matches = bf.match(des1, des2)

            # Sort the matched keypoints in the order of matching distance
            # so the best matches came to the front
            matches = sorted(matches, key=lambda x: x.distance)

            img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[0:100], None)
            height,width,_ = img_matching.shape
            img_matching = cv2.resize(img_matching,  (int(width/2),(int(height/2))))
            cv2.imshow('feature matching', img_matching)

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # compute essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

            # get camera motion
            R = R.transpose()
            t = -np.matmul(R, t)

            if i == 1:
                curr_R = R
                curr_t = t
            else:
                curr_R = np.matmul(prev_R, R)
                curr_t = np.matmul(prev_R, t) + prev_t

            # draw the current image with keypoints
            curr_img_kp = cv2.drawKeypoints(curr_img, kp2, None, color=(0, 255, 0), flags=0)
            cv2.imshow('keypoints from current image', curr_img_kp)

        # save current pose
        [tx, ty, tz] = [curr_t[0], curr_t[1], curr_t[2]]
        qw, qx, qy, qz = rot2quat(curr_R)

        end_time = time.time()
        timestamp = end_time - start_time

        with open(out_pose_file, 'a') as f:
            f.write('%f %f %f %f %f %f %f %f\n' % (timestamp, tx, ty, tz, qx, qy, qz, qw))

        prev_R = curr_R
        prev_t = curr_t

        # draw estimated trajectory (blue)
        offset_draw = (int(args.len_trajMap/2))
        offset_draw_small = (int(args.len_trajMap/8))

        # Different starting points depending on the dataset
        if args.dataset_type == 'TUM':
            cv2.circle(trajMap, (int(curr_t[0] + offset_draw) , int(curr_t[2]) + offset_draw), 1, (255, 0, 0), 2)
        elif args.dataset_type == 'KITTI':
            cv2.circle(trajMap, (int(curr_t[0] + offset_draw) , int(curr_t[2])+ offset_draw_small ), 1, (255, 0, 0), 2)
        cv2.imshow('Trajectory', trajMap)
        cv2.waitKey(1)

    # Save the trajectory result depending on the dataset
    if args.dataset_type == 'TUM':
        cv2.imwrite('TUM' + '-' + seq +'_trajMap.png', trajMap)
    elif args.dataset_type == 'KITTI':
        cv2.imwrite('KITTI' + '-' + seq +'_trajMap.png', trajMap)
