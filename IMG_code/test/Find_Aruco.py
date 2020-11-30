import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl

path = "C:/Users/wisar/OneDrive/My work/Project_module7/Aruo_4x4/"
frame = cv2.imread(path+"test.jpg")
window_name = 'Aruco'

"""
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
    return rect
"""
def get_rect_onePoint (pnts,id,w,h):
    if id == 0 :
        pnt = pnts[0]
        pnts[1][0],pnts[1][1] = pnt[0]+w,pnt[1]
        pnts[2][0],pnts[2][1] = pnt[0]+w,pnt[1]+h
        pnts[3][0],pnts[3][1] = pnt[0],pnt[1]+h
    elif id == 1 :
        pnt = pnts[1]
        pnts[0][0],pnts[0][1] = pnt[0]-w,pnt[1]
        pnts[2][0],pnts[2][1] = pnt[0],pnt[1]+h
        pnts[3][0],pnts[3][1] = pnt[0]-w,pnt[1]+h
    elif id == 2 :
        pnt = pnts[3]
        pnts[0][0],pnts[0][1] = pnt[0],pnt[1]-h
        pnts[1][0],pnts[1][1] = pnt[0]+w,pnt[1]-h
        pnts[2][0],pnts[2][1] = pnt[0]+w,pnt[1]
    else:
        pnt = pnts[2]
        pnts[0][0],pnts[0][1] = pnt[0]-w,pnt[1]-h
        pnts[1][0],pnts[1][1] = pnt[0],pnt[1]-h
        pnts[3][0],pnts[3][1] = pnt[0]-w,pnt[1]

def find_wh (rect):
    (tl, tr, br, bl) = rect
    print(rect)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    return maxWidth,maxHeight

def perspective(image, rect,maxWidth,maxHeight):
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
    return warped    


cap = cv2.VideoCapture(2)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
"""
ARUCO : corner          0-------1       3-------0
                        |    ^  |       |       |
                        |    |  |       |  ->   |
                        3-------2       2-------1
        numbers are the index in corner !!!

Battlefield : Position's ARUCO marker
                        0-------1       
                        |       |       
                        |       |       
                        2-------3       

"""
w = None
h = None
while (1):
    ret,frame = cap.read()
    key = cv2.waitKey(5)
    if ret is False or key == ord('q') :
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #rect = np.zeros((4, 2), dtype = "float32")
    aruco_pts = np.zeros((4, 2))
    list_id = [0,1,2,3]
    warped = None
    try :
        all_id = ids.tolist()
        for index in range(len(all_id)):
            cor = corners[index][0]
            if all_id[index][0] in list_id:
                if all_id[index][0] is 0:
                    aruco_pts[0]=cor[2]
                elif all_id[index][0] is 1:
                    aruco_pts[1]=cor[3]
                elif all_id[index][0] is 2:
                    aruco_pts[3]=cor[1]
                elif all_id[index][0] is 3:
                    aruco_pts[2]=cor[0]
                if w is not None :
                    get_rect_onePoint(aruco_pts,all_id[index][0],w,h)
                    break
                list_id.remove(all_id[index][0])
        if list_id == [] and w is None :
            w,h = find_wh (aruco_pts)
        if w is not None:
            aruco_pts = np.asarray(aruco_pts,dtype = "float32")
            warped = perspective(frame, aruco_pts,w,h)
            cv2.imshow("Prespective",warped)
    except :
        pass
    frame = aruco.drawDetectedMarkers(frame, corners, ids)    
    cv2.imshow(window_name,frame)
    



"""

    for index in range(len(all_id)):
        if all_id[index] in [0,1,2,3] and all_id[index] not in aruco_ids :
            cor = np.asarray(all_corner[0][index],dtype = np.int)
            lu_cor = cor[0]
            list_w = []
            list_h = []
            for i in cor[1:] :
                if lu_cor[0] > i[0]  :
                    lu_cor[0] = i[0]
                if lu_cor[1] > i[1] :
                    lu_cor[1] = i[1]
                w = abs(cor[0][0]-i[0])
                h = abs(cor[0][1]-i[1])
            if w != 0 :
                list_w.append(w)
            if h != 0 :
                list_h.append(h)
            aruco_box[all_id[index]] =[lu_cor[0],lu_cor[1],min(list_w),min(list_h)]
            aruco_ids.append(all_id[index][0])
    if len(aruco_ids) == 4:
        cor = aruco_box[0]
        lu_cor = [0,0]
        for index in range(len(1,aruco_ids)):
            if cor[0] > aruco_box[index][0] or cor[1] > aruco_box[index][1] :
                cor = aruco_box[index]
        for index in range(len(1,aruco_ids)):
    
"""
"""
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
all_corner, all_id, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#frame_markers = aruco.drawDetectedMarkers(frame.copy(), all_corner, all_id)

aruco_pts = []
list_id = [0,1,2,3]
warped = frame
all_id = all_id.tolist()
for index in range(len(all_id)):
    cor = all_corner[index][0]
    if all_id[index][0] in list_id:
        if all_id[index][0] is 0:
            aruco_pts.append(cor[2])
        elif all_id[index][0] is 1:
            aruco_pts.append(cor[3])
        elif all_id[index][0] is 2:
            aruco_pts.append(cor[1])
        elif all_id[index][0] is 3:
            aruco_pts.append(cor[0])
        #list_id.remove(all_id[index][0])

if len(aruco_pts) == 4:
    aruco_pts = np.asarray(aruco_pts,dtype = "float32")
    warped = four_point_transform(frame, aruco_pts)
      

plt.figure()
plt.imshow(warped)
plt.show()
"""
