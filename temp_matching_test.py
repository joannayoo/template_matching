import cv2 as cv
import numpy as np
import imutils
import os

# from matplotlib import pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
image = cv.imread(dir_path + '/picture.jpg')
template = cv.imread(dir_path + '/template.jpg')
img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
temp = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
# temp = cv.Canny(temp, 50, 200)

(tH, tW) = temp.shape[:2]
(iH, iW) = img.shape[:2]


methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for m in methods:
    count = 1
    for scale in np.linspace(1.0, 3.0, 21)[::-1]:
        dim = (int(iW * scale), int(iH * scale))
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        method = eval(m)
        result = cv.matchTemplate(resized, temp, method)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        clone = np.dstack([resized, resized, resized])
        cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
                     (maxLoc[0] + tW, maxLoc[1] + tH), (0,0,200), 5)
        writeon = dir_path +'/' + m + '_' + str(scale) + '.jpg'
        cv.imwrite(writeon, clone)
        count += 1
