import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from IAMLTools import getHomographyFromMouse

# def getHomographyFromMouse(image1, image2, N=4):
#     """
#     getHomographyFromMouse(image1, image2, N=4) -> homography, mousePoints

#     Calculates the homography from a plane in image "image1" to a plane in image "image2" by using the mouse to define corresponding points
#     Returns: 3x3 homography matrix and a set of corresponding points used to define the homography
#     Parameters: N >= 4 is the number of expected mouse points in each image,
#                 when N < 0: then the corners of image "image1" will be used as input and thus only 4 mouse clicks are needed in image "image2".

#     Usage: Use left click to select a point and right click to remove the most recently selected point.
#     """
#     # Vector with all input images.
#     images = []
#     images.append(cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB))
#     images.append(cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2RGB))

#     # Vector with the points selected in the input images.
#     mousePoints = []

#     # Control the number of processed images.
#     firstImage = 0

#     # When N < 0, then the corners of image "image1" will be used as input.
#     if N < 0:

#         # Force 4 points to be selected.
#         N = 4
#         firstImage = 1
#         m, n = image1.shape[:2]

#         # Define corner points from image "image1".
#         mousePoints.append([(0, 0), (n, 0), (n, m), (0, m)])

#     # Check if there is the minimum number of needed points to estimate the homography.
#     if math.fabs(N) < 4:
#         N = 4
#         print("At least 4 points are needed!!!")

#     # Make a pylab figure window.
#     fig = plt.figure(1)

#     # Get the correspoding points from the input images.
#     for i in range(firstImage, 2):
#         # Setup the pylab subplot.
#         plt.subplot(1, 2, i+1)
#         plt.imshow(images[i])
#         plt.axis("image")
#         plt.title("Click " + str(N) + " times in this image.")
#         fig.canvas.draw()

#         # Get mouse inputs.
#         mousePoints.append(fig.ginput(N, -1))

#         # Draw selected points in the processed image.
#         for point in mousePoints[i]:
#             cv2.circle(images[i], (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
#         plt.imshow(images[i])
#         fig.canvas.draw()

#     # Close the pylab figure window.
#     plt.close(fig)

#     # Convert to OpenCV format.
#     points1 = np.array([[x, y] for (x, y) in mousePoints[0]])
#     points2 = np.array([[x, y] for (x, y) in mousePoints[1]])

#     # Calculate the homography.
#     homography, mask = cv2.findHomography(points1, points2)
#     return homography, mousePoints

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
    filled_path = "images/stemboks/stem_1.jpg"

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