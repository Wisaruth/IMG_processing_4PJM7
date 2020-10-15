import cv2
import numpy as np

from matplotlib import pyplot as plt

def nothing(x):
    pass

ratio = 3
kernel_size = 3
contours = []
max_lowThreshold = 100

window_name = "Edge Detection"
bar_name = 'Thrshold'
path = "C:/Users/ASUS/OneDrive/My work/Project_module7/IMG_test/"
img = cv2.imread(path+"Map_A.jpg")



def CannyThreshold(low_threshold,img_):
    gray_img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img,5)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    #unsharp_image = cv2.addWeighted(gray_img, 1.5, blur_img, -0.5, 0, gray_img)
    detected_edges = cv2.Canny(blur_img, low_threshold, low_threshold*ratio, kernel_size)
    return detected_edges

def check_macthing(detected_edges,img_):
    _,contours_,_= cv2.findContours(detected_edges,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_:
        area = cv2.contourArea(cnt)
        if area > 500:
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            epsilon = 0.08 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            #num = len(approx)
            #if num == 3:
            #    crop_img = img_[y:y+h,x:x+w]
            #crop_img =[]
            crop_img = cv2.rectangle(img_, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(len(approx))
            cv2.drawContours( crop_img, [cnt], -1, (0, 255, 0), 2)
            cv2.imshow(window_name+"1", crop_img)
            cv2.waitKey()
            contours.append(cnt)
    

#cv2.namedWindow(window_name)
edge_img = CannyThreshold(27,img)
check_macthing(edge_img,img)
#cv2.createTrackbar(bar_name,window_name,0,max_lowThreshold,nothing)

while True:
    key = cv2.waitKey(10)
    """
    val = cv2.getTrackbarPos(bar_name,window_name)
    edge_img = CannyThreshold(val,img)"""
    cv2.imshow(window_name, edge_img)
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break
"""
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(draw_img, [approx], 0, (0), 2)
    cv2.imshow(window_name,draw_img)
    if len(approx) == 3:
        print("Triangle")
    elif len(approx) == 4:
        print("Rectangle")
    elif len(approx) == 5:
        print("Pentagon")
    elif 6 < len(approx) < 15:
        print("Ellipse")
    else:
        print("Circle")
    cv2.waitKey()
"""
