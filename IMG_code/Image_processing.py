import numpy as np
import cv2
from cv2 import aruco

class Image :

    def __init__(self,cap_index):
        self.cap = cv2.VideoCapture(cap_index)
        self.image = None
        self.gray_image = None
    
    def cap_img(self):
        ret,img = self.cap.read()
        if ret :
            return img
        else :
            return False
            
    def update_img(self,img):
        self.image = img 
        self.gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    def saveImg(self,name,path):
        if self.image != None:
            cv2.imwrite(path+name+".jpg",self.image) 
            print("Save Image :"+name)
            return True
        else:
            return False
    
#------------------------

#   Color Detection
    def clr_masking (self,hue_,sat_,val_):            # Make Mask ( one range color) : [low,up] degree , sat % and val %   
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_color = np.array([hue_[0],sat_[0]*255/100,val_[0]*255/100],dtype = np.uint8)
        upper_color = np.array([hue_[1],sat_[1]*255/100,val_[1]*255/100],dtype = np.uint8)
        return cv2.inRange(img,lower_color,upper_color)

    def color_detection (self,single_mode,hue,sat,val,thrshold_area):   # Detect color
        clr_det_contours =[]
        if single_mode :                                                            # find the color mask
            mask = self.clr_masking(hue,sat,val)
        else :
            mask1 = self.clr_masking (hue[0],sat,val)
            mask2 = self.clr_masking (hue[1],sat,val,)
            mask = mask1+mask2               
        _,all_contour,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if all_contour is None :                                                    # check that found contour
            return False,False
        mask = np.zeros(mask.shape, np.uint8)                                     # make the blank mask  
        for contour in all_contour :                                                # check area 
            area = cv2.contourArea(contour)
            if area > thrshold_area:
                clr_det_contours.append(contour)
                cv2.drawContours(mask,[contour], -1, (255), -1)
        if clr_det_contours is None :                                                    
            return False,False
        crop_clrs_img = cv2.bitwise_or(self.image,self.image,mask=mask)
        crop_clrs_img = cv2.medianBlur(crop_clrs_img, 3)
        return clr_det_contours,crop_clrs_img



class Calibration :
#   have2test!!!
    def __init__(self):
        self.mpax=None
        self.mpay=None
        self.roi=None

    def show_chessboard(self,img):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        if ret:
            draw_img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            return draw_img
        else:
            return False

    def calib_img(self,img):
        # crop the image
        dst = cv2.remap(img,self.mapx,self.mapy,cv2.INTER_LINEAR)
        x,y,w,h = self.roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def find_matrix4Calib(self,imgs,path):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        for index in range(len(imgs)):
            gray = cv2.cvtColor(imgs[index],cv2.COLOR_BGR2GRAY)
            find, corners = cv2.findChessboardCorners(gray, (7,6),None)
            # If found, add object points, image points (after refining them)
            if find:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)        
        shapes=gray.shape[::-1]
        ret, mtx, dist,_,_= cv2.calibrateCamera(objpoints,imgpoints,shapes,None,None)
        if ret:
            newcameramtx, self.roi =cv2.getOptimalNewCameraMatrix(mtx,dist,shapes,1,shapes)
            self.mpax,self.mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,shapes,5)
            self.save_Calib(path)
        else :
            return ret
    
    def save_Calib(self,path):
        # Save the camera matrix and the distortion coefficients to given path/file. """
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("mapx", self.mapx)
        cv_file.write("mapy", self.mapy)
        cv_file.write("roi", self.roi)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_Calib(self,path):
        # Loads camera matrix and distortion coefficients. """
        #FILE_STORAGE_READ
        try:
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            # note we also have to specify the type to retrieve other wise we only get a
            # FileNode object back instead of a matrix
            self.mapx = cv_file.getNode("mapx").mat()
            self.mapy = cv_file.getNode("mapy").mat()
            self.roi = cv_file.getNode("roi").mat()
            cv_file.release()
        except:
            print("Error: Not find mapx/mapy or wrong path")




class Aruco :
    def __init__(self):
        self.maxHeight = None
        self.maxWidth = None
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters =  aruco.DetectorParameters_create()

    def order_points(self,pts):
	    rect = np.zeros((4, 2), dtype = "float32")
	    s = pts.sum(axis = 1)
	    rect[0] = pts[np.argmin(s)]
	    rect[2] = pts[np.argmax(s)]
	    diff = np.diff(pts, axis = 1)
	    rect[1] = pts[np.argmin(diff)]
	    rect[3] = pts[np.argmax(diff)]
        #(tl, tr, br, bl) = rect
        widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
	    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
	    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1] ) ** 2))
	    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        self.maxWidth = max(int(widthA), int(widthB))
	    self.maxHeight = max(int(heightA), int(heightB))
	    # return the ordered coordinates
	    return rect

    def prespective(self,img,pts):
        rect = self.order_points(pts)
        """if self.maxHeight == None :
	        rect = self.order_points(pts)
        else :
            # Index 0 Top left
            rect = np.zeros((4, 2), dtype = "float32")
            rect[0] = pts[0]
            rect[2][0],rect[2][1] = pts[0]+self.maxWidth,pts[1]+self.maxHeight
            rect[1][0],rect[1][1] = pts[0]+self.maxWidth,pts[1]
            rect[3][0],rect[3][1] = pts[0],pts[1]+self.maxHeight"""   
	    dst = np.array([
		    [0, 0],
		    [self.maxWidth - 1, 0],
		    [self.maxWidth - 1, self.maxHeight - 1],
		    [0, self.maxHeight - 1]], dtype = "float32")
	    M = cv2.getPerspectiveTransform(rect, dst)
	    warped = cv2.warpPerspective(img, M, (self.maxWidth, self.maxHeight))
	    # return the warped image
	    return warped

    def find_aruco(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray,self.aruco_dict, parameters= self.parameters)
        aruco_pts = [0,0,0,0]
        list_id = [0,1,2,3]
        try :
            all_id = ids.tolist()
            for index in range(len(all_id)):
                cor = corners[index][0]
                if all_id[index][0] in list_id:
                    if all_id[index][0] is 0:
                        aruco_pts[0] = cor[2]
                    elif all_id[index][0] is 1:
                        aruco_pts[1] = cor[3]
                    elif all_id[index][0] is 2:
                        aruco_pts[2] = cor[1]
                    elif all_id[index][0] is 3:
                        aruco_pts[3] = cor[0]
                    list_id.remove(all_id[index][0])
            if list_id == []:
                aruco_pts = np.asarray(aruco_pts,dtype = "float32")
                warped = prespective(img, aruco_pts)
                show_aruco = aruco.drawDetectedMarkers(img, corners, ids)
                return warped,show_aruco
        except :
            return False,False
          

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

