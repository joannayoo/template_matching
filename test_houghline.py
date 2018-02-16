import cv2 as cv
import numpy as np

img = cv.imread('/home/joannayoo/venv/template_matching/images/sudoku.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 200, apertureSize = 3)

lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, 30, 10)
for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv.line(img, (x1, y1), (x2, y2), (0,0,255), 2)

cv.imwrite('/home/joannayoo/venv/template_matching/sudoku_result.jpg', img)
