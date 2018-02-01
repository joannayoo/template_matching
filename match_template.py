#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import imutils
import os
import time
import argparse

# from matplotlib import pyplot as plt
AVAILABLE_METHODS =  ['cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.realpath('__file__')), "templates")
DEFAULT_SCALE = 3.0

class BillTemplate:
    def __init__(self, image, template_dir, method):
        self.templates = {}
        temp_list = [f for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))]
        for i in range(len(temp_list)):
            MAX_TEMP_H = MAX_TEMP_W = 0
            temp_image = cv.imread(os.path.join(template_dir, temp_list[i]))
            if not temp_image.data:
                print("Invalid template file: " + temp_list[i])
                break
            else:
                self.templates[temp_list[i]] = cv.cvtColor(temp_image, cv.COLOR_BGR2GRAY)
                (tH, tW) = temp_image.shape[:2]
                if (tH > MAX_TEMP_H):
                    MAX_TEMP_H = tH
                if (tW > MAX_TEMP_W):
                    MAX_TEMP_W = tW

        image = cv.imread(image)
        if not image.data:
            print("Invalid image: " + image)
        ## TODO: check the largest size of the template and reject( or do something with) the input image if it is smaller than the template.
        else:
            self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            (iH, iW) = self.image.shape[:2]

            if (iH < MAX_TEMP_H) or (iW < MAX_TEMP_W):
                upsize = max(MAX_TEMP_H/iH, MAX_TEMP_W, iW)
                (iH, iW) = (iH, iW) * np.ceil(upsize)
                self.image = cv.resize(self.image, (iW, iH), interpolation=cv.INTER_AREA)
                

        method = 'cv.TM_' + method
        if method not in AVAILABLE_METHODS:
            print("Invalid method: " + method)
        else:
            self.method = method

    def match_template(self, scale, disp_match):
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
                if (s == 1.1):
                    cv.rectangle(resized, (maxLoc[0], maxLoc[1]), (maxLoc[0] + self.templates[t].shape[1], maxLoc[1] + self.templates[t].shape[0]), (0,0,200), 5)
                    writeon = os.path.join(os.path.dirname(os.path.realpath('__file__')), "1_1.jpg")
                    cv.imwrite(writeon, resized)
                    print(maxVal)
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
    parser.add_argument("--disp_match", dest="disp", action="store_true", help="Display matching result")
    parser.set_defaults(disp=False)
    args = parser.parse_args()

    billTemplate = BillTemplate(args.image, args.template_dir, args.method)
    billTemplate.match_template(args.scale, args.disp)

if __name__ == "__main__":

    main()
    
    
