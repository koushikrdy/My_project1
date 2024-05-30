1# -*- coding: utf-8 -*-

import numpy as np
#import tensorflow as tf
import keras.preprocessing.image
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble 
from skimage import measure
import cv2 
import matplotlib.pyplot as plt
import matplotlib.cm as cm #colour map
import watershed

im = cv2.imread('p9.jpg') 


blur = cv2.GaussianBlur(im,(5,5),0)
median = cv2.medianBlur(blur,5)


water=watershed.watershade(median)


gray_image = cv2.cvtColor(water, cv2.COLOR_BGR2GRAY)
cv2.imwrite("result2.jpg",gray_image)
res,thresh_img=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV) 
cv2.imwrite("result3.jpg",thresh_img)
thresh_img = cv2.erode(thresh_img, None, iterations=2)
cv2.imwrite("erode.png", thresh_img)
cv2.imshow('erode',thresh_img)
cv2.waitKey(0)
thresh_img = cv2.dilate(thresh_img, None, iterations=3)
cv2.imwrite("dilate.png", thresh_img)
cv2.imshow('dilate',thresh_img)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


    
count=0
areas=[]
original_image_copy=im.copy()    
for c in contours:
    area=cv2.contourArea(c)
    
    if area>10 and area<400:
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        areas.append(area)
        #img = cv2.circle(im,center,radius,(0,0,255),5)
        cv2.circle(im,center,radius,(0,0,255),5)
       
plt.show()
#plt.imshow(im)
#plt.show()
cv2.imwrite("result4.jpg",im)
im = cv2.imread('result4.jpg')
im = cv2.resize(im, (600,500), interpolation = cv2.INTER_AREA)
height=50
width=50
start_num=0
rows=np.vsplit(im,5)
cells=[]   
detections={}
i=0;
for row in rows:
    row_cells=np.hsplit(row,6)
    for cell in row_cells:
        lower = np.array([17, 15, 100], dtype = "uint8")
        upper = np.array([50, 56, 200], dtype = "uint8")
        
        mask = cv2.inRange(cell, lower, upper)
        detect=cv2.countNonZero(mask)
        if detect>0:
            detections.update({i:detect})
        i+=1
        plt.subplot(121),plt.imshow(cell),plt.title('Cancer Range:%s'%detect)#row,column,plot number
        plt.xticks([]), plt.yticks([])#a list of positions at which ticks should be placed
        plt.subplot(122),plt.imshow(mask),plt.title('Cancer Mask')
        plt.xticks([]), plt.yticks([])
        plt.title('Cancer Range:%s'%detect)
        plt.show()
        cell=cell.flatten()
        cells.append(cell)


if len(areas)>0: 
    print("Cancer Detected Cells")
    print("CellId\tRange")
    for de,da in detections.items():
        print("%s\t%s"%(de,da))
    print("No of nodes detected %s"%len(areas))
    print("Maximum area speaded %s"%max(areas))
    print("Totel detected Area Sizes")
    i=1
    for a in areas:
        print("Area %s size is %s"%(i,a))
        i+=1
        
    if max(areas) < 100:
        if len(areas)==1:
            print("Cancer Stage 1")
        else:
            print("Cancer Stage 2")
    
    else:
        if len(areas)==1:
            print("Cancer Stage 3")
        else:
            print("Cancer Stage 4")
    
        
    cv2.imshow('Result',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No Cancer Detected ")
