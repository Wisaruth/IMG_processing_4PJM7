import numpy as np
import cv2
from cv2 import aruco

class Image :

    def __init__(self,cap_index):
        self.cap = cv2.VideoCapture(cap_index)
        self.image = None
        self.gray_image = None
    
            
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
        self.imgs = []
        self.mpax=None
        self.mpay=None
        self.roi=None

    def show_chessboard(self,img):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        if ret:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            draw_img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            return draw_img
        else:
            return img

    def calib_img(self,img):
        # crop the image
        dst = cv2.remap(img,self.mapx,self.mapy,cv2.INTER_LINEAR)
        x,y,w,h = self.roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def find_matrix4Calib(self,path):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        for index in range(len(self.imgs)):
            find, corners = cv2.findChessboardCorners(self.imgs[index], (7,6),None)
            # If found, add object points, image points (after refining them)
            if find:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(self.imgs[index],corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)        
        shapes=self.imgs[index].shape[::-1]
        ret, mtx, dist,_,_= cv2.calibrateCamera(objpoints,imgpoints,shapes,None,None)
        if ret:
            newcameramtx, self.roi =cv2.getOptimalNewCameraMatrix(mtx,dist,shapes,1,shapes)
            self.mpax,self.mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,shapes,5)
            self.save_Calib(path)
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
# Can crop with one ARUCO mark
# have2test to make sure
    def __init__(self):
        self.Height = None
        self.Width = None
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters =  aruco.DetectorParameters_create()
    
    def get_rect_onePoint (self,pnts,id):
        if id == 0 :
            pnts[1][0],pnts[1][1] = pnts[0][0]+self.Width ,pnts[0][1]
            pnts[2][0],pnts[2][1] = pnts[0][0]+self.Width ,pnts[0][1]+self.Height
            pnts[3][0],pnts[3][1] = pnts[0][0],pnts[0][1]+self.Height
        elif id == 1 :
            pnts[0][0],pnts[0][1] = pnts[1][0]-self.Width ,pnts[1][1]
            pnts[2][0],pnts[2][1] = pnts[1][0],pnts[1][1]+self.Height
            pnts[3][0],pnts[3][1] = pnts[1][0]-self.Width ,pnts[1][1]+self.Height
        elif id == 2 :
            pnts[0][0],pnts[0][1] = pnts[3][0],pnts[3][1]-self.Height
            pnts[1][0],pnts[1][1] = pnts[3][0]+self.Width ,pnts[3][1]-self.Height
            pnts[2][0],pnts[2][1] = pnts[3][0]+self.Width ,pnts[3][1]
        else:
            pnts[0][0],pnts[0][1] = pnts[2][0]-self.Width ,pnts[2][1]-self.Height
            pnts[1][0],pnts[1][1] = pnts[2][0],pnts[2][1]-self.Height
            pnts[3][0],pnts[3][1] = pnts[2][0]-self.Width ,pnts[2][1]

    def maxWidth_Height (self,rect):
        (tl, tr, br, bl) = rect
        self.Width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        self.Height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        #widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        #maxWidth = max(int(widthA), int(widthB))
        #heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        #maxHeight = max(int(heightA), int(heightB))

    def perspective (self,img,rect):   
        dst = np.array([
		    [0, 0],
		    [self.Width - 1, 0],
		    [self.Width - 1, self.Height - 1],
		    [0, self.Height - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.Width, self.Height))

    def cropWith_aruco(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray,self.aruco_dict, parameters= self.parameters)
        aruco_pts = np.zeros((4, 2))
        list_id = [0,1,2,3]
        all_id = ids.tolist()
        check = False
        for index in range(len(all_id)):
            cor = corners[index][0]
            if all_id[index][0] in list_id:
                check = True
                if all_id[index][0] is 0:
                    aruco_pts[0]=cor[2]
                elif all_id[index][0] is 1:
                    aruco_pts[1]=cor[3]
                elif all_id[index][0] is 2:
                    aruco_pts[3]=cor[1]
                elif all_id[index][0] is 3:
                    aruco_pts[2]=cor[0]
                if self.Width is not None :
                    self.get_rect_onePoint(aruco_pts,all_id[index][0])
                    break
                list_id.remove(all_id[index][0])
        if list_id == [] and self.Width is None :
            self.maxWidth_Height(aruco_pts)
        if self.Width is not None and check:
            aruco_pts = np.asarray(aruco_pts,dtype = "float32")
            warped = self.perspective(img, aruco_pts)
            show_aruco = aruco.drawDetectedMarkers(img, corners, ids)
            return True,warped,show_aruco
        else :
            return False,None,None


    def save_Aruco(self,path):
        # Save the camera matrix and the distortion coefficients to given path/file. """
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("Height", self.Height)
        cv_file.write("Width", self.Width)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_Aruco(self,path):
        # Loads camera matrix and distortion coefficients. """
        #FILE_STORAGE_READ
        try:
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            # note we also have to specify the type to retrieve other wise we only get a
            # FileNode object back instead of a matrix
            self.Height = cv_file.getNode("Height").mat()
            self.Width = cv_file.getNode("Width").mat()
            cv_file.release()
        except:
            print("Error: Not find W&H or wrong path")
          

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

