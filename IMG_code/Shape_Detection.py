import cv2
import numpy as np

from matplotlib import pyplot as plt

ratio = 3
kernel_size = 3
contours = []
max_lowThreshold = 100
title_trackbar = 'Min Threshold:'
window_name = 'Shape Detection'
path = "D:/Studio/ProjectModule7/IMG4Test/"
img = cv2.imread(path+"Color_B.jpg")



def CannyThreshold(val):
    low_threshold = val
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img,5)
    gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    detected_edges = cv2.Canny(gray_img, low_threshold, low_threshold*ratio, kernel_size)
    cv2.imshow(window_name, detected_edges)
    #mask = detected_edges != 0
    """_,contours_,_= cv2.findContours(detected_edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for index in range(len(contours_)):
        area = cv2.contourArea(contours_[index])
        if area > 200:
            contours.append(contours_[index])"""



cv2.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
#CannyThreshold(0)
while True:
    key = cv2.waitKey(1)
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
