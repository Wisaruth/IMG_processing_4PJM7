import numpy as np
import cv2


class Image :

    def __init__(self,cap_index,path):
        self.cap = cv2.VideoCapture(cap_index)
        self.path=path
        self.frame=None
        self.mpax=None
        self.mpay=None
        self.roi=None
    
    def update(self,raw_mode=True):
        ret,frame = self.cap.read()
        if ret:
            if raw_mode:
                self.frame=frame
            else:
                dst = cv2.remap(frame,self.mapx,self.mapy,cv2.INTER_LINEAR)
                # crop the image
                x,y,w,h = self.roi
                dst = dst[y:y+h, x:x+w]
                self.frame=dst
        else:
            print("Error: Not find any image")
        
    def saveImg(self,name):
        if self.frame != None:
            cv2.imwrite(self.path+name+".jpg",self.frame) 
            print("Save Image :"+name)
            return True
        else:
            return False

#   Calibration
#       have2test!!!
    def find_matrix4Calib(self,showchess_mode=False):
        window_name ="Calibration"
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        shoot = False
        while True:
            ret,frame = self.cap.read()
            cv2.imshow(window_name,frame)
            key = cv2.waitKey(10) & 0xFF 
            if  ret is False or key == ord('q') :
                break
            elif key == ord('g') :
                shoot = True
                print ("    capture now")
            if shoot :
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                find, corners = cv2.findChessboardCorners(gray, (7,6),None)
                # If found, add object points, image points (after refining them)
                if find:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)
                    # Draw and display the corners
                    if showchess_mode:
                        draw_img = cv2.drawChessboardCorners(frame, (7,6), corners2,ret)
                        cv2.imshow(window_name,draw_img)
                        cv2.waitKey()
                    shoot = False
        print ("Number of image:{}".format(len(imgpoints)))
        shapes=gray.shape[::-1]
        ret, mtx, dist,_,_= cv2.calibrateCamera(objpoints,imgpoints,shapes,None,None)
        newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,shapes,1,shapes)
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,shapes,5)
        self.save_coef_Calib(mapx,mapy,roi)
        return [mapx,mapy]
    
    def save_coef_Calib(self,mapx,mapy,roi):
        # Save the camera matrix and the distortion coefficients to given path/file. """
        cv_file = cv2.FileStorage(self.path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("mapx", mapx)
        cv_file.write("mapy", mapy)
        cv_file.write("roi", roi)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_coef_Calib(self):
        # Loads camera matrix and distortion coefficients. """
        #FILE_STORAGE_READ
        try:
            cv_file = cv2.FileStorage(self.path, cv2.FILE_STORAGE_READ)
            # note we also have to specify the type to retrieve other wise we only get a
            # FileNode object back instead of a matrix
            self.mapx = cv_file.getNode("mapx").mat()
            self.mapy = cv_file.getNode("mapy").mat()
            self.roi = cv_file.getNode("roi").mat()
            cv_file.release()
        except:
            print("Error: Not find mapx/mapy or wrong path")
    
#------------------------


#   Color Detection
    def clr_masking (self,hue_,sat_,val_):            # Make Mask ( one range color) : [low,up] degree , sat % and val %   
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([hue_[0],sat_[0]*255/100,val_[0]*255/100],dtype = np.uint8)
        upper_color = np.array([hue_[1],sat_[1]*255/100,val_[1]*255/100],dtype = np.uint8)
        return cv2.inRange(img,lower_color,upper_color)

    def color_detection (self,single_mode,hue,sat,val,thrshold_area):   # Detect color
        clr_det_contours =[]
        if single_mode :                                                            # find the color mask
            mask = self.clr_masking (hue,sat,val)
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
        crop_clrs_img = cv2.bitwise_or(self.frame,self.frame,mask=mask)
        crop_clrs_img = cv2.medianBlur(crop_clrs_img, 3)
        return clr_det_contours,crop_clrs_img

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

        
            
            
        

        


