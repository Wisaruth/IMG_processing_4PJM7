import cv2
import numpy as np

from matplotlib import pyplot as plt

ratio = 3
kernel_size = 8
contours = []
max_lowThreshold = 100
title_trackbar = 'Min Threshold:'
window_name = 'Shape Detection'
path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/"
img = cv2.imread(path+"Map_A.jpg")



def CannyThreshold(val,img_):
    low_threshold = val
    gray_img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img,5)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    #unsharp_image = cv2.addWeighted(gray_img, 1.5, blur_img, -0.5, 0, gray_img)
    detected_edges = cv2.Canny(blur_img, low_threshold, low_threshold*ratio, kernel_size)
    _,contours_,_= cv2.findContours(detected_edges,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_:
        area = cv2.contourArea(cnt)
        if area > 100:
            epsilon = 0.08 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            #cv2.drawContours(img_, cnt, -1, (0, 255, 0), 3)
            for index in range(len(approx)):
                cv2.circle(img_, (approx[index][0][0],approx[index][0][1]), 1, (255, 0, 0), 2)
            cv2.imshow(window_name+"1", img_)
            cv2.waitKey()
            contours.append(cnt)
    return img_,detected_edges





canny_img,edge_img = CannyThreshold(10,img)
while True:
    key = cv2.waitKey(1)
    cv2.imshow(window_name+"1", edge_img)
    cv2.imshow(window_name+"2", canny_img)
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
