from PIL import Image
import cv2
import sys
import os
sys.path.append('/home/sagar/.local/bin')
import pytesseract
img=cv2.imread('123.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img=255-img
#et3,gray= cv2.threshold(img,150,255,cv2.THRESH_BINARY)
cv2.imwrite('fil.png',gray)
text=pytesseract.image_to_string(Image.open('fil.png'))
os.remove('fil.png')
print('The pridicted number is {}'.format(text))
cv2.imshow('Image',img)
cv2.imshow('o',gray)
cv2.waitKey(0)