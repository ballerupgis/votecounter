import numpy as np
import cv2
from glob import glob
from warpImage import warpImage, autoWarpImage
from collections import Counter

def binaryROIImage(image, rectangles, threshold=10):
    """
    Creates a binary image with black background and white
    BLOBS within each ROI indicating marks on the ballot.
    """
    
    h, w = image.shape[:2]

    # Grayscale image.
    grayscale = image.copy()
    if len(grayscale.shape) == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    # Create negative imgage for thresholding
    inv = cv2.bitwise_not(grayscale)
    
    # Create a binary image.
    _, thres = cv2.threshold(inv, threshold, 255, cv2.THRESH_BINARY)
    
    zeroed = np.zeros((h,w), dtype=np.uint8)

    for points in rectangles:
        x,y,w,h = cv2.boundingRect(np.array(points, dtype=np.int32))
        w-=1
        h-=1

        roi = thres[y:y+h, x:x+w]
        zeroed[y:y+h, x:x+w] = roi
            
    return zeroed

def detectMark(image, rectangles, threshold=10):
    """
    Finding largest BLOB area and returns the centroid.
    Threshold is for creation og binary ROI image.
    """
    areas = []
    centroids = []

    image = binaryROIImage(image, rectangles, threshold)

    # Find blobs in the input image.
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        for contour in contours:
            area = cv2.contourArea(contour) 
            centroid = calcCentroid(contour)
            areas.append(area)
            centroids.append(centroid) 
        idx = areas.index(max(areas))
        point = centroids[idx]

        return point
    else:
        print("Didn't find any marks")

def pointWithinRectangle(point, rectangle):
    """
    Return true if point is within a rectangle. rectangles is 
    defined as two diagonal corner points e.g. bottom left, top right 
    """
    px = point[1]
    py = point[0]
    x_vals = [x for (y, x) in rectangle]
    y_vals = [y for (y, x) in rectangle]
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)    
    if (x_min < px < x_max and y_min < py < y_max):
        return True
    else:
        return False

def calcCentroid(contour):
    """
    Calculates the centroid of the contour. Moments up to the third
    order of a polygon or rasterized shape.
    """
    moments = cv2.moments(contour)

    centroid = (-1, -1)
    if moments["m00"] != 0:
        centroid = (int(round(moments["m10"] / moments["m00"])),
                    int(round(moments["m01"] / moments["m00"])))

    return centroid
    
def getVote(point, data):
    """
    Looping though ROIs on a ballot and checking if the point is within on of the rectangles. 
    Returns corrosponding ROI name (politician) if condition is meet.
    """
    for i in data:
        if pointWithinRectangle(point, i["rect"]):
            return i["name"]

def loadImagesFromPath(glob_path):
    """
    Loads images from folder into list. Glob patterns can be used to select 
    specicific files e.g. path/to/*.jpg. This function might not be appropriate 
    when loading thousands of images since it is stored in memory.
    """
    images = []
    paths = glob(glob_path)
    for path in paths:
        img = cv2.imread(path)
        images.append(img)
    return images

def voteCounter(ballots, empty_ballot, data):
    """
    Counting votes on ballots and returning list of votes and 
    dictionary of politicians and corrosponding counts. Inputs 
    is list of images of filled ballots, image of empty ballot 
    and data about ROIs and corrosponding names
    """
    votes = []
    rects = [i['rect'] for i in data]

    for ballot in ballots:
        warped = autoWarpImage(empty_ballot, ballot)
        point = detectMark(warped, rects)
        vote = getVote(point, data)
        print(vote)
        votes.append(vote)

    counts = dict(Counter(votes))
    return votes, counts

if __name__ == '__main__':

    empty_path = "images/stemboks/stem_back.jpg"
    empty = cv2.imread(empty_path)
    data = np.load('data/stemboks_corrected.npy')

    ballots = loadImagesFromPath('images/stemboks/*.jpg')

    votes, counts = voteCounter(ballots,empty, data)
    print(votes, counts)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()