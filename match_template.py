#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import imutils
import os
import time
import argparse
from sklearn.cluster import KMeans

# from matplotlib import pyplot as plt
AVAILABLE_METHODS =  ['cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.realpath('__file__')), "templates")
DEFAULT_SCALE = 3.0
USE_ARGMIN = False
tW = 1526
tH = 648

class BillTemplate:
    def __init__(self, image, template_dir, method):
        self.templates = {}
        temp_list = [f for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))]
        for i in range(len(temp_list)):
            MAX_TEMP_H = MAX_TEMP_W = 0
            temp_image = cv.imread(os.path.join(template_dir, temp_list[i]))
            if not temp_image.data:
                print("Invalid template file: " + temp_list[i])
            else:
                self.templates[temp_list[i]] = cv.cvtColor(temp_image, cv.COLOR_BGR2GRAY)

        image = cv.imread(image)
        if not image.data:
            print("Invalid image: " + image)
        ## TODO: check the largest size of the template and reject( or do something with) the input image if it is smaller than the template.
        else:
            self.image = image
            #self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            #(iH, iW) = self.image.shape[:2]
            #if (iH < tH) or (iW < tW):
            #    self.image = cv.resize(self.image, (tW, tH), interpolation=cv.INTER_AREA)


        method = 'cv.TM_' + method
        if method not in AVAILABLE_METHODS:
            print("Invalid method: " + method)
        elif method in ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']:
            USE_ARGMIN = True
            self.method = method
        else:
            self.method = method

    def extract_and_match(self, disp_match):
        start = time.time()
        img = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        hue, sat, val = cv.split(img)
        retval, thresh = cv.threshold(sat, 50, 225, cv.THRESH_BINARY_INV)
        thresh_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, (7,7))
        thresh_close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, (7,7))
        thresh_edge = cv.Canny(thresh_close, 15, 150)

        cnts = cv.findContours(thresh_edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]

        im_copy = self.image
        perimeter = 0
        our_c = None
        
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02*peri, True)
            if approx.all() > perimeter:
                our_c = approx
        cv.drawContours(im_copy, [our_c], -1, (0,255,0), 5)
        
        # for i,c in enumerate(cnts):
            #print(c)
        #    c[i] = cv.convexHull(c[i])
        #for c in cnts:
        #    peri = cv.arcLength(c, True)
        #    if peri > perimeter:
        #        our_c = c 
        
        (iH, iW) = thresh_edge.shape[:2]
        black_image = np.zeros((iH, iW, 3), np.uint8)
        
        draw_cont = cv.drawContours(black_image, [our_c], -1, (255,255,255) , 1)
        cont_gray = cv.cvtColor(black_image, cv.COLOR_BGR2GRAY)
        
        ## Harris corner detection
        gray = np.float32(cont_gray)
        dst = cv.cornerHarris(gray, 10,3,0.04)
        dst_thresh = 0.1 * dst.max()
        new_dst = dst > dst_thresh
        corners = np.transpose(np.nonzero(new_dst))

        ref_1 = [0, 0]
        ref_2 = [0, iW]
        dist_1 = np.sum((corners - ref_1) ** 2, axis = 1)
        dist_2 = np.sum((corners - ref_2) ** 2, axis = 1)
        tl = corners[np.argmin(dist_1)]
        tr = corners[np.argmin(dist_2)]
        bl = corners[np.argmax(dist_2)]
        br = corners[np.argmax(dist_1)]

        cv.circle(draw_cont, (tl[1], tl[0]), 3, (0,0,255), -1)
        cv.circle(draw_cont, (tr[1], tr[0]), 3, (0,0,255), -1)
        cv.circle(draw_cont, (bl[1], bl[0]), 3, (0,0,255), -1)
        cv.circle(draw_cont, (br[1], br[0]), 3, (0,0,255), -1)
        # cv.imwrite('/home/joannayoo/venv/template_matching/step_10.jpg', draw_cont) #TODO
        rect = np.array([tl, tr, br, bl], dtype = "float32")
        rect[:,[0,1]] = rect[:,[1,0]]

        dst_size = np.array([[0,0], [tW -1, 0], [tW - 1, tH -1], [0, tH - 1]], dtype="float32")
        M = cv.getPerspectiveTransform(rect, dst_size)

        warp = cv.warpPerspective(im_copy, M, (tW, tH))
        # cv.imwrite('/home/joannayoo/venv/template_matching/step_11.jpg', warp) #TODO
        warp_border = cv.copyMakeBorder(warp, top=10, bottom=10, left=15, right=15, borderType=cv.BORDER_CONSTANT, value=[0,255,0])
        #warp_blur = cv.GaussianBlur(warp, (5,5), 0)
        warp_gray = cv.cvtColor(warp_border, cv.COLOR_BGR2GRAY)

        cv.imwrite('/home/joannayoo/venv/template_matching/step_12.jpg', warp_gray) #TODO

        best_match = ""
        match_values = {}
        for bill, template in self.templates.items():
            method = eval(self.method)
            result = cv.matchTemplate(warp_gray, template, method)
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
            print(bill)
            if USE_ARGMIN:
                match_values[bill] = minVal    
                print(minVal)
                print("===================")

            else:
                match_values[bill] = maxVal
                print(maxVal)
                print("===================")

        if USE_ARGMIN:
            best_match = min(match_values, key = match_values.get)
        else:
            best_match = max(match_values, key = match_values.get)
        end = time.time()
        elapsed_time = end - start
        
        print("The image is matched to " + best_match +  ". It took " + str(elapsed_time) + " seconds.")

        return best_match, elapsed_time
        '''
        lines = cv.HoughLines(cont_gray, 1, np.pi/180, 200)
        #print(lines)
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 200*(-b))
                y1 = int(x0 + 200*(a))
                x2 = int(x0 - 200*(-b))
                y2 = int(x0 - 200*(a))
                cv.line(draw_cont, (x1,y1),(x2,y2), (0,0,255), 5)
        cv.imwrite('/home/joannayoo/venv/template_matching/step_8.jpg', draw_cont) #TODO

        #blur = cv.GaussianBlur(draw_cont, (3,3), 0)
        

        
        rect = cv.minAreaRect(our_c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(im_copy, [box], 0, (0,0, 255) , 5)
        
        L = tuple(our_c[our_c[:, :, 0].argmin()][0])
        print(L) 
        R = tuple(our_c[our_c[:, :, 0].argmax()][0])
        U = tuple(our_c[our_c[:, :, 1].argmin()][0])
        D = tuple(our_c[our_c[:, :, 1].argmax()][0])
        pts = [L, R, U, D]
        cv.circle(im_copy, L, 10, (0,0,255),-1)
        cv.circle(im_copy, R, 10, (0,0,255),-1)
        cv.circle(im_copy, U, 10, (0,0,255),-1)
        cv.circle(im_copy, D, 10, (0,0,255),-1)
        
    
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(im_copy, (x,y), (x + w, y + h), (0,255,0), 5)
        
        c = max(cnts, key = cv.arcLength)
        cv.drawContours(im_copy, [c], -1, (0,255,0),5)
        
        
        #extraction method 1 : contour detection
        
        #kernel=cv.getStructuringElement(cv.MORPH_RECT, (15,15))
        #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        cv.imwrite('/home/joannayoo/venv/template_matching/step_3.jpg', img) #TODO
        img2, contours, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(self.image, contours, -1, (0,255,0), 10)
       
        for c in contours:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                cv.drawContours(img, [approx], -1, (0,255,0), 5)
    
        cv.imwrite('/home/joannayoo/venv/template_matching/step_4.jpg', img) #TODO
        
        
        #extraction metod 2 : Hough Transform
        img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        cv.imwrite('/home/joannayoo/venv/template_matching/step_1.jpg', img) #TODO
        img = cv.GaussianBlur(img, (3,3), 0)
        img = cv.Canny(img, 50,250)
        cv.imwrite('/home/joannayoo/venv/template_matching/step_2.jpg', img) #TODO

        lines = cv.HoughLines(img, 1, np.pi/180, 200)
        print(lines[0])
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 200*(-b))
                y1 = int(x0 + 200*(a))
                x2 = int(x0 - 200*(-b))
                y2 = int(x0 - 200*(a))
                cv.line(self.image, (x1,y1),(x2,y2), (0,0,255), 5)
        cv.imwrite('/home/joannayoo/venv/template_matching/step_3.jpg', self.image) #TODO
        '''
    def scale_and_match(self, scale, disp_match):
        start = time.time()
        
        opt_value = 0
        location = (0,0)
        matched_template = ""
        opt_scale = 1.0

        (iH, iW) = self.image.shape[:2]
        matched_image = np.zeros((iH, iW, 3), np.uint8)

        method = eval(self.method)
        increment = int((scale - 1.0)*10 + 1)
        
        for t in list(self.templates.keys()):
            for s in np.linspace(1.0, scale, increment)[::1]:
                resized = cv.resize(self.image, None, fx = s, fy = s, interpolation=cv.INTER_AREA)
                method = eval(self.method)
                result = cv.matchTemplate(resized, self.templates[t], method)
                (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)

                if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    if minVal < opt_value:
                        opt_value = minVal
                        location = minLoc
                        matched_template = t
                        opt_scale = s
                        matched_image = resized
                    else:
                        pass
                else:
                    if maxVal > opt_value:
                        opt_value = maxVal
                        location = maxLoc
                        matched_template = t
                        opt_scale = s
                        matched_image = resized

        if (disp_match):
            (tH, tW) = self.templates[matched_template].shape[:2]
            cv.rectangle(matched_image, (location[0], location[1]),(location[0] + tW, location[1] + tH), (0,0,200), 5)
            cv.imshow("match result", matched_image)
            writeon = os.path.join(os.path.dirname(os.path.realpath('__file__')), "result.jpg")
            cv.imwrite(writeon, matched_image)

        end = time.time()
        elapsed_time = end - start
        
        print("The image matched to " + matched_template + " with scale " + str(opt_scale) + ". It took " + str(end - start) + " seconds. Accuracy is " + str(opt_value))
        return opt_value, location, matched_template, opt_scale, elapsed_time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image to be matched")
    parser.add_argument("--template_dir", default=DEFAULT_TEMPLATE_PATH, help="Directory for the templates")
    parser.add_argument("--method", default="CCORR", help="Choose from CCOEFF, CCOEFF_NORMED, CCORR, CCORR_NORMED, SQDIFF, and SQDIFF_NORMED")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Maximum scale value")
    parser.add_argument("--scale_and_match", dest="use_scale", action="store_true", help="scale the bill and match")
    parser.add_argument("--disp_match", dest="disp", action="store_true", help="Display matching result")
    parser.set_defaults(use_scale=False)
    parser.set_defaults(disp=False)
    args = parser.parse_args()

    billTemplate = BillTemplate(args.image, args.template_dir, args.method)
    if (args.disp):
        billTemplate.scale_and_match(args.scale, args.disp)
    else:
        billTemplate.extract_and_match(args.disp)

if __name__ == "__main__":

    main()
    
    
