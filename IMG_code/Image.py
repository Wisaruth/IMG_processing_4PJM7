import numpy as np
import cv2
from skimage.morphology import skeletonize


class Image :
#  
    def __init__(self,camera_index):
        self.cap = cv2.VideoCapture(camera_index+cv2.CAP_DSHOW)
        self.image = []
        self.bin_image = None
    
    def setting (self):
        codec = 0x47504A4D  # MJPG
        self.cap.set(cv2.CAP_PROP_FPS, 60.0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off

    def update_img(self,img):
        kernel = np.ones((5,5),np.uint8)
        self.image = img.copy() 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _,self.bin_image = cv2.threshold(gray , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    def saveImg(self,name,path):
        if self.image != []:
            cv2.imwrite(path+name+".jpg",self.image) 
            print("Save Image :"+name)
            return True
        else:
            return False

    def canny(self,low_thres):
        kernel_size = 3
        ratio = 2
        blur_img = cv2.GaussianBlur(self.bin_image,(5,5),0)
        boots_img = cv2.addWeighted(self.bin_image, 2, blur_img, -1, 0, self.bin_image) 
        result = cv2.Canny(boots_img, low_thres, low_thres*ratio,kernel_size)
        return result

    def skelton_mask (self,cntset):
        kernel = np.ones((5,5),np.uint8)
        binary_img=cv2.bitwise_not(self.bin_image)
        for cnt in cntset:
            cv2.drawContours(binary_img, [cnt], 0, 255, -1)
        erosion = cv2.erode(binary_img,kernel,iterations = 1)
        mask = skeletonize(erosion, method='lee')
        return  mask


    
def hue_masking (clr_img2mask,range_clr,range_sat,range_val):            # Make Mask ( one range color) : [low,up] degree , sat % and val %   
    lower_color = np.array([range_clr[0],range_sat[0]*255/100,range_val[0]*255/100],dtype = np.uint8)
    upper_color = np.array([range_clr[1],range_sat[1]*255/100,range_val[1]*255/100],dtype = np.uint8)
    return cv2.inRange(clr_img2mask,lower_color,upper_color)


def color_detection (img_,hsv_img_,hue_,sat_,val_,thrshold_area):   # Detect color
    clr_det_contours =[]
    mask_ = hue_masking (hsv_img_,hue_,sat_,val_)
    _,all_contour,_ = cv2.findContours(mask_,cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_NONE)
    if all_contour is None :                                                    # check that found contour
        return False
    mask_ = np.zeros(mask_.shape, np.uint8)                                     # make the blank mask  
    for contour in all_contour :                                                # check area 
        area = cv2.contourArea(contour)
        if area > thrshold_area:
            clr_det_contours.append(contour)
            cv2.drawContours(mask_,[contour], -1, (255), -1)
    if clr_det_contours is None :                                                    
        return False
    crop_clrs_img = cv2.bitwise_or(img_,img_,mask=mask_)
    crop_clrs_img = cv2.medianBlur(crop_clrs_img, 5)
    return crop_clrs_img





#          Example
"""
hsv_blue = [90,120]
sat = [25,100]
val = [60,100]
clr_img,cnt = image.color_detection(single_mode = True,
                                        hue=hsv_blue,
                                        sat=sat,
                                        val=val,
                                        thrshold_area= 500)
"""
#------------------------

