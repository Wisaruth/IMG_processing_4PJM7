import numpy as np
import cv2
import os

gray_img_set =[]
img_set =[]
window_name = "Camera_No.1"
path ="C:/Users/ASUS/Downloads/IMG4Calibra/"
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

shoot = False
cap = cv2.VideoCapture(0)
print ("Program has started :")
while True:
    ret,frame = cap.read()
    cv2.imshow(window_name,frame)
    if  ret is False :
        break
    key = cv2.waitKey(5) & 0xFF # delay & get input from keyboard
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break
    elif key == ord('g') :
        shoot = True
        print ("Searching")

    if shoot :
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        if ret:
            cv2.imwrite(path+"img_"+str(len(img_set))+".jpg", frame) 
            print("Save Image No. {}".format(len(img_set)))
            img_set.append(frame)
            gray_img_set.append(gray)
            shoot = False


for index in range(len(img_set)):
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray_img_set[index], (7,6),None)
    # If found, add object points, image points (after refining them)
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray_img_set[index],corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
        # Draw and display the corners
    draw_img = img_set[index]
    draw_img = cv2.drawChessboardCorners(draw_img, (7,6), corners2,ret)
    #cv2.imwrite(path+"Cali_img_"+str(index)+".jpg", draw_img) 
    while True:
        cv2.imshow(window_name,draw_img)
        key = cv2.waitKey(5) & 0xFF # delay & get input from keyboard
        if key == ord('q') or key == ord('Q') :
            cv2.destroyWindow(window_name)
            break

img = cv2.imread(path+"img_0.jpg")
h, w = img.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h),None,None)
newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)


dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(path+"Final_img.jpg",dst)
while True:
    cv2.imshow(window_name,dst)
    key = cv2.waitKey(5) & 0xFF # delay & get input from keyboard
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break
