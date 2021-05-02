import cv2
import numpy as np
from keras.models import model_from_json

img = cv2.imread('98b.jpg',cv2.IMREAD_GRAYSCALE)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(img)
#img=color_detection(img)
#equ = cv2.equalizeHist(img)
#res = np.hstack((img,equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)
#cv2.imshow('ff',cl1)
#cv2.waitKey(0)
cv2.imshow('j',img)
cv2.waitKey(0)
img=cv2.medianBlur(img,5)
img=255-img
ret3,img= cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img=cv2.medianBlur(img,5)
#img=skew_correction(img)
#kernel = np.ones((5,5),np.uint8)
#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
proj = np.sum(img,1)
m=0
h=0
for i in range(len(proj)):
    if proj[i]>700:
        m=i
        break
for i in range(len(proj)-1,1,-1):
    if proj[i]>700:
        h=i
        break
height=h-m
hori_img=img[m:h,0:img.shape[1]]
cv2.imshow('d',hori_img)
cv2.waitKey(0)

#performing word segmentation
proj = np.sum(hori_img,0)
for i in range(len(proj)):
    if proj[i]>700:
        m=i
        break
for i in range(len(proj)-1,1,-1):
    if proj[i]>700:
        h=i
        break
print(hori_img.shape)
hori_img=hori_img[0:hori_img.shape[0],m:h]
cv2.imshow('d2',hori_img)
cv2.waitKey(0)
#character segmentation
proj = np.sum(hori_img,0)
print(proj)
width=[0]
i=0;
while(i!=len(proj)):
    if proj[i]<700:
        width.append(i)
        for k in range(i,len(proj)):
            if proj[k]>700:
                width.append(k)
                i=k
                break;
    i=i+1
width.append(hori_img.shape[1])
mark=''
loop=len(width)/2;
for i in range(0,len(width),2):
    res=cv2.resize(hori_img[0:hori_img.shape[0],width[i]:width[i+1]],(28,28),cv2.INTER_AREA)
    cv2.imshow('d3',res)
    cv2.waitKey(0)
    json_file = open('model.json', 'r')
    res = cv2.blur(res,(7,7))
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json) 
    loaded_model.load_weights("model.h5")
    res=np.expand_dims(res,2)
    res=np.expand_dims(res,0)
    y=np.argmax(loaded_model.predict(res), axis=-1)
    mark=mark+str(y[0])   
print(mark)
