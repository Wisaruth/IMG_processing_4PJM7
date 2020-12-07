import numpy as np
import cv2
from cv2 import aruco

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
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            #print(ret)
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
            #self.save_Calib(path)
        self.imgs = []
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



class BG_subtractor :
# Can crop with one ARUCO mark
# have2test to make sure
    def __init__(self):
        self.imgset = []
        self.Height = None
        self.Width = None
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters =  aruco.DetectorParameters_create()
    

    def maxWidth_Height (self,rect):
        (tl, tr, br, bl) = rect
        self.Width = round(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)))
        self.Height = round(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)))

    def perspective (self,img,rect):   
        dst = np.array([
		    [0, 0],
		    [self.Width - 1, 0],
		    [self.Width - 1, self.Height - 1],
		    [0, self.Height - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.Width, self.Height))

    def cropWith_aruco(self,img,show_mode):
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray,self.aruco_dict, parameters= self.parameters)
        aruco_pts = np.zeros((4, 2))
        list_id = [0,1,2,3]
        show_aruco = None
        warped = None
        check = False
        if ids is not None:
            all_id = ids.tolist()
            if show_mode :
                show_aruco = aruco.drawDetectedMarkers(img.copy(), corners, ids)
            for index in range(len(all_id)):
                cor = corners[index][0]
                if all_id[index][0] in list_id:
                    if all_id[index][0] is 0:
                        aruco_pts[0]=cor[2]
                        #if self.Width is not None :
                        #    check = self.get_rect_onePoint(aruco_pts,all_id[index][0],gray.shape)
                        #    break
                    elif all_id[index][0] is 1:
                        aruco_pts[1]=cor[3]
                    elif all_id[index][0] is 2:
                        aruco_pts[3]=cor[1]
                    elif all_id[index][0] is 3:
                        aruco_pts[2]=cor[0]
                    
                    list_id.remove(all_id[index][0])
            if list_id == [] :
                if self.Width is None :
                    self.maxWidth_Height(aruco_pts)
                check = True
            if self.Width is not None and check:
                aruco_pts = np.asarray(aruco_pts,dtype = "float32")
                warped = self.perspective(img, aruco_pts)
        return check,warped,show_aruco
        
    def add_imgset(self,img):
        ret,crop_img,_ = self.cropWith_aruco(img,False)
        if ret:
            self.imgset.append(crop_img.copy()) 

    def median2getBG(self):
        imgs = np.asarray(self.imgset)
        median_img = np.median(imgs,axis=0).astype(np.uint8)
        return median_img

    ### Useless and don't work out as what I expected
    def get_rect_onePoint (self,pnts,id,img_shape): 
        if id == 0 :
            pnts[2][0],pnts[2][1] = pnts[0][0]+self.Width ,pnts[0][1]+self.Height
            if pnts[2][0] <= img_shape[1] and pnts[2][1] <= img_shape[0]:
                pnts[1][0],pnts[1][1] = pnts[0][0]+self.Width ,pnts[0][1]
                pnts[3][0],pnts[3][1] = pnts[0][0],pnts[0][1]+self.Height
            else :
                return False
        elif id == 1 :
            pnts[3][0],pnts[3][1] = pnts[1][0]-self.Width ,pnts[1][1]+self.Height
            if pnts[3][0] >= 0 and pnts[3][1] <= img_shape[0] :
                pnts[0][0],pnts[0][1] = pnts[1][0]-self.Width ,pnts[1][1]
                pnts[2][0],pnts[2][1] = pnts[1][0],pnts[1][1]+self.Height
            else :
                return False
        elif id == 2 :
            pnts[1][0],pnts[1][1] = pnts[3][0]+self.Width ,pnts[3][1]-self.Height
            if pnts[1][0] <= img_shape[1] and pnts[1][1] >= 0 :
                pnts[0][0],pnts[0][1] = pnts[3][0],pnts[3][1]-self.Height
                pnts[2][0],pnts[2][1] = pnts[3][0]+self.Width ,pnts[3][1]
            else :
                return False
        else:
            pnts[0][0],pnts[0][1] = pnts[2][0]-self.Width ,pnts[2][1]-self.Height
            if pnts[0][0] >= 0 and pnts[0][1]  <= img_shape[0]:
                pnts[1][0],pnts[1][1] = pnts[2][0],pnts[2][1]-self.Height
                pnts[3][0],pnts[3][1] = pnts[2][0]-self.Width ,pnts[2][1]
            else :
                return False
        return True

    def save_Aruco(self,path): 
        # Save the camera matrix and the distortion coefficients to given path/file. """
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("Height", self.Height)
        cv_file.write("Width", self.Width)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_Aruco(self,path):
        #FILE_STORAGE_READ
        try:
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            self.Height = cv_file.getNode("Height").mat()
            self.Width = cv_file.getNode("Width").mat()
            cv_file.release()
        except:
            print("Error: Not find W&H or wrong path")
    ### 

