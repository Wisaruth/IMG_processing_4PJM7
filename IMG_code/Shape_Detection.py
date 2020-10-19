import cv2
import numpy as np

from matplotlib import pyplot as plt

def nothing(x):
    pass


max_lowThreshold = 100
contours = []
sign_imgs = []
sign_name = [0,2,8]
sign_name2 = ["Star","2Triangle","2Triangle"]


window_name = "Edge Detection"
bar_name = 'Thrshold'
path = "C:/Users/ASUS/OneDrive/My work/Project_module7/IMG_test/"
map_img = cv2.imread(path+"Map_A.jpg")
for j in sign_name :
    sign_img = cv2.imread(path+"img_test"+str(j)+".jpg",0)
    sign_imgs.append(sign_img)


def CannyThreshold(low_thres_cannay,img_,approx_mode,approx_thres,area_thres): 
    # cv2.CHAIN_APPROX_SIMPLE cv2.CHAIN_APPROX_NONE
    # approx_thres = 0.08
    kernel_size_cannay = 3
    ratio_ = 2
    pass_cnts = []
    pass_boxcoord =[]
    pass_approx = []
    gray_img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img,5)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    unsharp_image = cv2.addWeighted(gray_img, 2, blur_img, -1, 0, gray_img)
    detected_edges = cv2.Canny(unsharp_image, low_thres_cannay, low_thres_cannay*ratio_,kernel_size_cannay)
    if approx_mode is True:
        mode_ = cv2.CHAIN_APPROX_SIMPLE
    else :
        mode_ = cv2.CHAIN_APPROX_NONE
    _,cnts_,_= cv2.findContours(detected_edges,cv2.RETR_TREE,mode_)
    for cnt in cnts_:
        area = cv2.contourArea(cnt)
        if area > area_thres:
            pass_cnts.append(cnt)
            if approx_mode :
                rect = cv2.boundingRect(cnt)
                pass_boxcoord.append(rect)
                epsilon = approx_thres * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pass_approx.append(approx)

    return detected_edges,pass_cnts,pass_boxcoord,pass_approx

def check_matching(img_,temp_,match_thres,cnt_,boxcoord_):
    
    #count=0
    x,y,w,h = boxcoord_
    crop_img = img_[y:y+h,x:x+w]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.resize(crop_img, (temp_.shape[1], temp_.shape[0]),
                            interpolation = cv2.INTER_LINEAR)
    res = cv2.matchTemplate(crop_img,temp_,cv2.TM_CCOEFF_NORMED)
    _, max_val,_,_ = cv2.minMaxLoc(res)
    if max_val >= match_thres:
        return True
    else :
        return False

    #cv2.rectangle(img_ , (x, y), (x+w, y+h), (0, 255, 0), 2)
    
cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name)    
font = cv2.FONT_HERSHEY_SIMPLEX 
  


fontScale = 1
thickness = 2
   
# Using cv2.putText() method 
  

#edge_img = CannyThreshold(56,img)
edge_img,contours,coordinate,approx = CannyThreshold(65,map_img,True,0.08,100)
last_coord = [0,0,0]

for j in range(len(sign_name)):
    for i in range(len(contours)):
        check= check_matching(map_img,sign_imgs[j],0.6,contours[i],coordinate[i])
        if check :
            x,y,w,h = coordinate[i]
            #print(coordinate[i])
            mid_ = [int((x+w)/2),int((y+h)/2)]
            dis_chess = max([abs(last_coord[0]-mid_[0]),abs(last_coord[1]-mid_[1])])
            if last_coord[2]==0 or dis_chess>last_coord[2] :
                map_img = cv2.rectangle(map_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                map_img = cv2.putText(map_img,str(sign_name2[j]),(coordinate[i][0],coordinate[i][1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                print(sign_name[j]) 
                last_coord[0],last_coord[1] = mid_[0],mid_[1]
                last_coord[2] = max([w,h])
        
            
cv2.imshow(window_name, map_img)
while True:
    key = cv2.waitKey(10)
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break       

#cv2.createTrackbar(bar_name,window_name,0,max_lowThreshold,nothing)
"""
while True:
    key = cv2.waitKey(10)
    ret,frame = cap.read()
    edge_img,contours,coordinate,approx = CannyThreshold(65,frame,True,0.08,500)
    #val = cv2.getTrackbarPos(bar_name,window_name)
    #edge_img = CannyThreshold(val,img)
    cv2.imshow(window_name, edge_img)
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break
"""
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
