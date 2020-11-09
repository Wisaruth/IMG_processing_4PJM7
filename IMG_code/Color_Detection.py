#https://medium.com/dolab/blog-3-python-with-opencv-for-color-detection-and-find-corner-detection-4a4b4c77590b
##https://stackoverflow.com/questions/38877102/how-to-detect-red-color-in-opencv-python
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/"
window_name = 'Color Detection'
img = cv2.imread(path+"real_map_B1.jpg")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cood = [0,0]
ck = False

class Frame :
    def __init__(self, shape):
        self.table = np.zeros((shape[0],shape[1]), dtype=np.uint8)

def mouse_click(event, x ,y ,flags, param):
    global cood,ck
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coodrinate : {} {}".format(x,y))
        cood[1],cood[0] =x,y
        ck = True
        
# Convert

def single_hue_masking (clr_img2mask,range_clr,range_sat,range_val):            # Make Mask ( one range color) : [low,up] degree , sat % and val %   
    lower_color = np.array([range_clr[0],range_sat[0]*255/100,range_val[0]*255/100],dtype = np.uint8)
    upper_color = np.array([range_clr[1],range_sat[1]*255/100,range_val[1]*255/100],dtype = np.uint8)
    return cv2.inRange(clr_img2mask,lower_color,upper_color)

def due_hue_masking (clr_img2redmask,hue1,hue2,range_sat,range_val):            # Make Mask ( two range color) : [low,up] degree,[low,up] degree, sat % and val % 
    mask1 = single_hue_masking(clr_img2redmask,hue1,range_sat,range_val)        # For detect red
    mask2 = single_hue_masking(clr_img2redmask,hue2,range_sat,range_val)
    return cv2.bitwise_or(mask1,mask2)

def level_clr_map (mask_clr_img,clr_frame):
    for i in range(mask_clr_img.shape[0]):
        for j in range(mask_clr_img.shape[1]):
            if mask_clr_img[i][j][0] != 0 and mask_clr_img[i][j][1] != 0 and mask_clr_img[i][j][2] != 0:   
                clr_frame.table[i][j] = 100*(255-mask_clr_img[i][j][1])/255
                
    return clr_frame


def color_detection (img_,hsv_img_,single_mode,hue_,sat_,val_,thrshold_area):   # Detect color
    clr_det_contours =[]
    if single_mode :                                                            # find the color mask
        mask_ = single_hue_masking (hsv_img_,hue_,sat_,val_)
    else :
        mask_ = due_hue_masking (hsv_img_,hue_[0],hue_[1],sat_,val_)               
    _,all_contour,_ = cv2.findContours(mask_,cv2.RETR_TREE,
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

#hsv_red = [[0,10],[135,179]]
hsv_ord = [0,20]
hsv_blue = [90,120]
hvs_all = [0,180] 
sat = [25,100]
val = [60,100]
blue_frame = Frame(img.shape)
contours,clrs_img = color_detection(img,hsv_img,True,hvs_all,sat,val,500)
blue_frame = level_clr_map (clrs_img ,blue_frame)

if contours is not False :  
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name,mouse_click)
    while True:
        key = cv2.waitKey(5)
        cv2.imshow(window_name,clrs_img)
        if key == ord('q') :
            break
        """if ck :
            img = cv2.circle(img, (cood[1],cood[0]), 2, (0, 255, 0), 3)
            print(" sat : {}".format(hsv_img[cood[0]][cood[1]]))
            print(" Level : {}".format(blue_frame.table[cood[0]][cood[1]]))
            ck = False
        """
else :
    print("Not found")
 


