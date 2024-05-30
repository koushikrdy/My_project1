# -*- coding: utf-8 -*-

import numpy as np
import cv2, matplotlib.pyplot as plt
# read image and show
def watershade(img):
    
    plt.axis('off')
    plt.imshow(img)
    # performing otsu's binarization
    # convert to gray scale first
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #pure black and pure white
    print("Threshold limit: " + str(ret))
    
    plt.axis('off')#hide axis values
    plt.imshow(thresh, cmap = 'gray')
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)#removes unwanted noises.2 types:erode,dilate
    
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)#increases the object area
    
    # sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    plt.imshow(dist_transform, cmap = 'gray')
    fig = plt.figure(figsize = (10,5)) # to change figsize
    plt.subplot(131)
    plt.imshow(sure_bg, cmap = 'gray')
    plt.title('Sure background, dilated')
    
    plt.subplot(132)
    plt.imshow(sure_fg, cmap = 'gray')
    plt.title('Sure foreground, eroded')
    
    plt.subplot(133)
    plt.imshow(unknown, cmap = 'gray')
    plt.title('Subtracted image, black - sure bg & fg')
    plt.tight_layout()
    
    # plt.subplots_adjust(wspace = 3)
    # fine tuning 
    # f.subplots_adjust(wspace=3)
    ret, markers = cv2.connectedComponents(sure_fg)
    
    markers = markers + 1
    
    markers[unknown==255] = 0
    
    fig = plt.figure(figsize = (10,5)) # to change figsize
    plt.subplot(121)
    plt.imshow(markers, cmap = 'gray')
    plt.subplot(122)
    plt.imshow(markers)
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 255,0]
    print("----Watershed----")
    

    cv2.imwrite('watershade.jpg',img)
    plt.imshow(img)
    return img
    
im = cv2.imread("p9.jpg")
img=watershade(im)
plt.imshow(img)
