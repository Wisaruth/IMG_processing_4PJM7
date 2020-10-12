#https://medium.com/dolab/blog-3-python-with-opencv-for-color-detection-and-find-corner-detection-4a4b4c77590b
##https://stackoverflow.com/questions/38877102/how-to-detect-red-color-in-opencv-python
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "D:/Studio/ProjectModule7/IMG4Test/"
window_name = 'Color Detection'
img = cv2.imread(path+"Color_B.jpg")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def mouse_click(event, x ,y ,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
# Convert

def single_hue_masking (clr_img2mask,range_clr,range_sat,range_val):            # Make Mask ( one range color) : [low,up] degree , sat % and val %   
    lower_color = np.array([range_clr[0],range_sat[0]*255/100,range_val[0]*255/100],dtype = np.uint8)
    upper_color = np.array([range_clr[1],range_sat[1]*255/100,range_val[1]*255/100],dtype = np.uint8)
    return cv2.inRange(clr_img2mask,lower_color,upper_color)

def due_hue_masking (clr_img2redmask,hue1,hue2,range_sat,range_val):            # Make Mask ( two range color) : [low,up] degree,[low,up] degree, sat % and val % 
    mask1 = single_hue_masking(clr_img2redmask,hue1,range_sat,range_val)        # For detect red
    mask2 = single_hue_masking(clr_img2redmask,hue2,range_sat,range_val)
    return cv2.bitwise_or(mask1,mask2)

def color_detection (img_,hsv_img_,single_mode,hue_,sat_,val_,thrshold_area):   # Detect color
    clr_det_contours =[]
    if single_mode :                                                            # find the color mask
        mask_ = single_hue_masking (hsv_img_,hue_,sat_,val_)
    else :
        mask_ = due_hue_masking (hsv_img_,hue_[0],hue_[1],sat_,val_)               
    _,all_contour,_ = cv2.findContours(mask_,cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)
    if all_contour is None :                                                    # check that found contour
        return False,False
    mask_ = np.zeros(mask_.shape, np.uint8)                                     # make the blank mask  
    for contour in all_contour :                                                # check area 
        area = cv2.contourArea(contour)
        if area > thrshold_area:
            clr_det_contours.append(contour)
            cv2.drawContours(mask_,[contour], -1, (255), -1)
            #for x in range(len(contour)):
            #    cv2.circle(img, (contour[x][0][0], contour[x][0][1]), 3, (0,0,255), 2)
    if clr_det_contours is None :                                                    
        return False,False
    crop_clrs_img = cv2.bitwise_or(img_,img_,mask=mask_)
    crop_clrs_img = cv2.medianBlur(crop_clrs_img, 5)
    return clr_det_contours,crop_clrs_img

hsv_red = [[0,10],[135,179]]
hsv_blue = [90,135] 
sat = [15,100]
val = [5,100]
contours,clrs_img = color_detection(img,hsv_img,True,hsv_blue,sat,val,500)
if contours is not False :  
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name,mouse_click)
    cv2.drawContours(img , contours, -1, (255), 2)
    while True:
        key = cv2.waitKey(5)
        cv2.imshow(window_name,clrs_img )
        cv2.imshow(window_name+"1",img )
        if key == ord('q') :
            break
else :
    print("Not found")
 


