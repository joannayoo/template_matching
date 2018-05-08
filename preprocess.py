#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import os
import argparse

DEFAULT_NUM_IMGS = 3

def crop_images(path_in, path_out, num_images):
    # EX. path_in - KRW - 1.bmp, 2.bmp .. 
    #             - USD - 1.bmp, 2.bmp .. 
    currency_list = os.listdir(path_in)
    for currency in currency_list:
        img_list = os.listdir(os.path.join(path_in, currency))
        j = 1

        for i in range(len(img_list)):
            img = cv.imread(os.path.join(path_in, currency, img_list[i]))
            img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hue, sat, val = cv.split(img2)
            cnts = cv.findContours(hue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
            cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:num_images]
        
            for c in cnts:
                mask = np.zeros_like(img)
                peri = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.02*peri, True)
                cv.drawContours(mask, [approx], -1, (255,255,255), -1)

                out = np.zeros_like(img)
                out[mask == 255] = img[mask == 255]

                (x, y, _) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
            
                out = out[topx:bottomx+1, topy:bottomy+1]
        
                filename = str(j) + '.jpg'
                cv.imwrite(os.path.join(path_out, currency, filename), out)

                print(currency + ": ", str(j))
                j += 1
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", required=True, help="Directory of the original image")
    parser.add_argument("--path_out", required=True, help="Directory for the output images")
    parser.add_argument("--num_images", default=DEFAULT_NUM_IMGS, help="Number of images in one page")
    args = parser.parse_args()

    crop_images(args.path_in, args.path_out, args.num_images)

if __name__ == "__main__":
    main()
