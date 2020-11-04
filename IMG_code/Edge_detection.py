#https://answers.opencv.org/question/172906/how-to-turn-contours-into-vectors/
#https://stackoverflow.com/questions/50274063/find-coordinates-of-a-canny-edge-image-opencv-python
#https://stackoverflow.com/questions/59525032/how-to-sum-areas-of-contours-after-sorted-them

from __future__ import print_function
import cv2 
import argparse
import numpy as np

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'

ratio = 3
kernel_size = 3
xy_mosue= [0,0]
contours = []
thepoint = 0
clicked = 0
check = 0 
path = "C:/Users/ASUS/OneDrive/My work/Project_module7/IMG_test/"

def  img_capture  (camera,name,mode,path):
    imgs =  []
    print ("Capture :")
    window_name= "Capture"
    count = 1
    while True:
        ret,frame = camera.read()
        cv2.imshow(window_name,frame)
        key = cv2.waitKey(5) & 0xFF # delay & get input from keyboard
        if  ret is False :
            break
        if key == ord('q') or key == ord('Q') :
            cv2.destroyWindow(window_name)
            if mode:
                return imgs
            else :
                break
        elif key == ord('g') :
            if mode :
                imgs.append(frame)
            else  :
                print ("Save Image :{}".format(count))
                cv2.imwrite(path+name+str(count)+".jpg", frame)
                count+=1

def mouse_click(event, x ,y ,flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click : {},{}".format(x,y))
        xy_mosue[0],xy_mosue[1] = x,y
        clicked = 1

def distance_cal (p1,x,y):       # 2 parameters are list 
    return np.sqrt(((p1[0]-x)**2)+((p1[1]-y)**2))
# val = 10
def CannyThreshold(val):
    low_threshold = val
    #img_blur = cv2.medianBlur(src_gray,5)
    img_blur = cv2.GaussianBlur(src_gray,(5,5),0)
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    #mask = detected_edges != 0
    _,contours_,_= cv2.findContours(detected_edges,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index in range(len(contours_)):
        area = cv2.contourArea(contours_[index])
        if area > 200:
            cv2.drawContours(src, contours_[index], -1, (0, 255, 0), 2)
            contours.append(contours_[index])
            #for index in range(0,len(contour),int((len(contour)-1)/2)):
            """for index in range(0,50,10):
                cv2.circle(src, (contour[index][0][0],contour[index][0][1]), i, (255, 0, 0), 2)
                i+=5
                print(contour[index][0][0],contour[index][0][1])   #0:width 1:high
            """

    
    

src = cv2.imread(path+"Map_A.jpg")
if src is None:
    print('Could not open or find the image: ')
    exit(0)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#cv2.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(10)
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name,mouse_click)

while True:
    key = cv2.waitKey(1)
    cv2.imshow(window_name, src)
    if clicked == 1:
        for index in range(len(contours)):
            i = True
            min_distace = 0
            for point in contours[index]:
                distace = distance_cal (xy_mosue,point[0][0],point[0][1])
                if i :
                    min_distace = distace
                    i = False
                if distace < min_distace and i == False:
                    min_distace = distace
                    thepoint = point[0]
            if thepoint is not None :
                cv2.circle(src, (thepoint[0],thepoint[1]), 10, (0, 255, 0), 2)
                print("{} : Distance {:.2f} pixcels ({})".format(index,min_distace,thepoint))    
        clicked = 0
        print("Done")
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break

"""
#cap = cv2.VideoCapture(0)
#img_capture(cap,"Edge_img",0,path)
src = cv2.imread(path+"Edge_img2.jpg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gaussian_3 = cv2.GaussianBlur(src_gray, (0, 0), 3.0)
unsharp_image = cv2.addWeighted(src_gray, 1.5, gaussian_3, -0.5, 0, src_gray)
cv2.imshow(window_name, unsharp_image)
cv2.waitKey()
"""