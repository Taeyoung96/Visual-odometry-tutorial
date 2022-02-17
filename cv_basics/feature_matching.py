import cv2
import numpy as np
import matplotlib.pyplot as plt


def feature_detecting_harris(img):
    # parameters for cv2.cornerHarris():
    # img - Input image, it should be grayscale and float32 type.
    # blockSize - It is the size of the small patch considered for corner detection
    # ksize - Aperture parameter of Sobel derivative used.
    # k - Harris detector free parameter in the equation.

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harrisDetector = cv2.cornerHarris(img_gray, 2, 3, 0.04)

    # Threshold for an optimal value, it may vary depending on the image.
    img[harrisDetector > 0.01*harrisDetector.max()] = [0, 0, 255]
    cv2.imshow('dst', img)
    cv2.waitKey(0)


def feature_matching_orb(img1, img2):
    # create ORB features
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # use brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match ORB descriptors
    matches = bf.match(des1, des2)

    # Sort the matched keypoints in the order of matching distance
    # so the best matches came to the front
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 100 matches.
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:100], None)

    img_match_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
    plt.imshow(img_match_rgb)
    plt.show()


if __name__ == "__main__":
    # read grayscale images
    img1 = cv2.imread('../data/kitti_l.png')
    img2 = cv2.imread('../data/kitti_l_1.png')

    # feature_detecting_harris(img1)
    feature_matching_orb(img1, img2)







