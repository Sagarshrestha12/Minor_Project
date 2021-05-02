import cv2
import numpy as np
import argparse
import csv
from keras.models import model_from_json

import sqlite3
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
def skew_correction(thresh):
    #image =cv2.imread(filename)
    # This is done by averaging the three chanel value
    #gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # This is don by using simple techinque 255-value 
    #gray=cv2.bitwise_not(image)
    # binairze the image using thresholding 
   # gray=cv2.bitwise_not(gray)
   # thresh = cv2.threshold(gray, 0, 255,
      #  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle <-45:
        angle= (90 +angle)
    else:
        angle = -angle
    # now applying affine transformation
    (h,w)=thresh.shape[:2]
    centre=(w//2,h//2)
    M=cv2.getRotationMatrix2D(centre,angle,1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def rescaleFrame(frame,scale =0.75):
    width=int(frame.shape[0]*scale)
    height =int(frame.shape[1]*scale)
    dimensions =(height,width)
    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)
def color_detection(img):
    ret3,img= cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    mask= mask1+mask2
    output = cv2.bitwise_and(img, img, mask = mask)
    output=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    output = cv2.threshold(output, 250, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return output

def noise_reduction(img):
    kernel=np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing
def segmentation(img):
    Img = cv2.imread('sample.png',cv2.IMREAD_GRAYSCALE)
    #Invert
    Img = 255 - Img
    #Calculate horizontal projection
    proj = np.sum(Img,1)
    #create output image same height as text,500 px wide
    m= np.max(proj)
    w=500
    result = np.zeros((proj.shape[0],500))
    print(result.shape)
    proj=(proj)*w/m
    c=max(proj)
    r=np.where(proj==c)
    upper = Img[0:r[0][0]-10,:]
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    upper = cv2.morphologyEx(upper, cv2.MORPH_DILATE, kernel3)
    cont_upper,_ = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lower = Img[r[0][0]+5:,:]
    cont_lower, _  = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def det(img):
    row,cols=img.shape[0],img.shape[1]
    for i in range(row):
        for j in range(cols):
            k=img[i,j]
            if k[2]>=255:
                img[i,j]=[0,0,255]
            else:
                img[i,j]=[0,0,0]
    return img

'''img=cv2.imread("123.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('img2',img)
cv2.waitKey(0)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('d',th3)
cv2.waitKey(0)
kernel=np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('d',closing)
cv2.waitKey(0)
#img=cv2.imread("my.jpeg")
#cv2.imshow("skew_image",img)
#img1=skew_correction("my.jpeg")
#cv2.imshow("skew_corrected_image",img1)
#cv2.waitKey(0)
#rel=cv2.imread("saj.jpg")
#blank=np.ones(rel.shape,dtype='uint8')*255
#blank[:]=0,0,255
##rel1= rescaleFrame(rel,scale=0.2)
#img=rel[:,:,2]
#mg2=cv2.imread("ocen.jpeg")
#img2=rescaleFrame(img2,scale=2)
#cv2.imshow("blur",img2)
#img2= cv2.GaussianBlur(img2,(5,5),cv2.BORDER_DEFAULT)
#cv2.imshow("blur2",img2) 
#
#img=cv2.bitwise_and(rel,blank)
#img =rescaleFrame(img,scale=0.2)
#cv2.imshow('Blank',img)
#cv2.imshow('Blank2',rel1)
#height=rel1.shape[0]
#widht=rel1.shape[1]
#cropped=rel1[0:height//2,0:widht//2]
#cropped=cv2.resize(cropped,(cropped.shape[1],cropped.shape[0]),interpolation=cv2.INTER_CUBIC)
#cv2.imshow("croop",cropped)
#cv2.waitKey(0)
#i#mg3=cv2.imread("numbers3.png")
#im#g3= cv2.resize(img3,(24,24),interpolation=cv2.INTER_AREA)
#cv2.imshow("fff",img3)
#cv2.waitKey(0)
'''


