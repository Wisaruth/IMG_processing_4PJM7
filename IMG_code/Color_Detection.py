#https://medium.com/dolab/blog-3-python-with-opencv-for-color-detection-and-find-corner-detection-4a4b4c77590b
##https://stackoverflow.com/questions/38877102/how-to-detect-red-color-in-opencv-python
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "D:/Studio/ProjectModule7/IMG4Test/"
window_name = 'Color Detection'
img = cv2.imread(path+"Color_B.jpg")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_img = cv2.medianBlur(hsv_img, 5)


def mouse_click(event, x ,y ,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
# Convert

def color_masking (img_,color,sat,val):          # color : [low,up] degree , sat % and val %   
    sat[0],sat[1] = sat[0]*255/100,sat[1]*255/100
    val[0],val[1] = val[0]*255/100,val[1]*255/100 
    lower_color = np.array([color[0],sat[0],val[0]],dtype = np.uint8)
    upper_color = np.array([color[1],sat[1],val[1]],dtype = np.uint8)
    mask = cv2.inRange(img_,lower_color,upper_color)
    return mask

def red_color_masking (img_,hue1,hue2,sat,val): 
    mask1 = color_masking(img_,hue1,sat,val)
    mask2 = color_masking(img_,hue2,sat,val)
    cv2.imshow(window_name+"1",mask1)
    cv2.imshow(window_name+"2",mask2)
    mask = cv2.bitwise_or(mask1,mask2)
    return mask

#mask = red_color_masking (hsv_img,[0,10],[50,179],[10,100],[10,100])

mask1 = color_masking(hsv_img,[135,179],[10,100],[5,100])
mask2 = color_masking(hsv_img,[0,10],[10,100],[5,100])
mask = cv2.bitwise_or(mask1,mask2)

_,contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)

for contour in contours :
    area = cv2.contourArea(contour)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if area > 500:
        cv2.drawContours(img, [approx], -1, (255), 1)
        for x in range(len(approx)):
            cv2.circle(img, (approx[x][0][0], approx[x][0][1]), 3, (0,0,255), -1)

    
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name,mouse_click)

while True:
    key = cv2.waitKey(5)
    cv2.imshow(window_name,mask)
    if key == ord('q') :
        break
 


