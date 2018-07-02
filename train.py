import cv2
import numpy as np
from IAMLTools import getContourProperties
from voteCounter import detectMark
from glob import glob


data = np.load('data/stemboks.npy')
rects = [i['rect'] for i in data]
warped = np.load('temp/warped.npy')

paths = glob('temp/cn*.npy')

contours = []

for path in paths:
    cnt = np.load(path)
    contours.append(cnt)



print(len(contours[6]))

point = detectMark(warped, rects)
print(point)