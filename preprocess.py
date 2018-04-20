#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import os


orig_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'samples')
new_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'samples')

currency_list = os.listdir(orig_path)

for currency in currency_list:

    img_list = os.listdir(os.path.join(orig_path, currency))
    j = 0
    for i in range(len(img_list)):
        img = cv.imread(os.path.join(orig_path, currency, img_list[i]))
        img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hue, sat, val = cv.split(img2)
        cnts = cv.findContours(hue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]
        
        for c in cnts:
            mask = np.zeros_like(img)
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02*peri, True)
            cv.drawContours(mask, [approx], -1, (255,255,255), -1)

            out = np.zeros_like(img)
            out[mask == 255] = img[mask == 255]
            print(currency)
            print(j)
            (x, y, _) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            
            out = out[topx:bottomx+1, topy:bottomy+1]
        
            filename = str(j) + '.jpg'
            cv.imwrite(os.path.join(new_path, currency, filename), out)
    
            j += 1

        #out = np.zeros_like(im_copy)
        #out[mask == 255] = img[mask == 255]
        
    
    #cv.imwrite(os.path.join(orig_path, currency, 'temp.jpg'), im_copy)
    
