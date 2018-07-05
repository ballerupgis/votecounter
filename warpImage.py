import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from IAMLTools import getHomographyFromMouse

def warpImage(warpimage, image):
    # Get the homography of both images.
    Htm, _ = getHomographyFromMouse(warpimage, image)
    
    # Get the image resolution.
    h, w = image.shape[:2]

    # Apply a perspective transformation to warp image.
    warped = cv2.warpPerspective(warpimage, Htm, (w, h))

    return warped

def autoWarpImage(empty, filled):
    # Load the images in gray scale
    img1 = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape
    img2 = cv2.resize(img2, (w, h))

    # Detect the SIFT key points and compute the descriptors for the two images
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    # Match the descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Select the good matches using the ratio test
    goodMatches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    # Apply the homography transformation if we have enough good matches 
    MIN_MATCH_COUNT = 10

    if len(goodMatches) > MIN_MATCH_COUNT:
        # Get the good key points positions
        sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
        destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
        
        # Obtain the homography matrix
        M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        matchesMask = mask.ravel().tolist()
        
    else:
        print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
        matchesMask = None
        
    warped = cv2.warpPerspective(img2, M, (w, h), flags=cv2.WARP_INVERSE_MAP)

    return warped
    
if __name__ == '__main__':
    empty_path = "images/stemboks/stem_back.jpg"
    filled_path = "images/stemboks/stem_3.jpg"

    empty = cv2.imread(empty_path)
    filled = cv2.imread(filled_path)

    warped = warpImage(filled, empty)
    warped2 = autoWarpImage(empty, filled)

    # # Show the final result.
    cv2.imshow("warped", warped)
    cv2.imshow("AUTO warp", warped2)
    cv2.imshow("Background", empty)

    cv2.waitKey(0)
    cv2.destroyAllWindows()