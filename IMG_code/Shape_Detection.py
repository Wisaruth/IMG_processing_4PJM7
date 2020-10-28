import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass


max_lowThreshold = 100
contours = []
sign_imgs = []
sign_name = [0,2,8]
sign_name2 = ["Star","Triangle","2Triangles"]


window_name = "Edge Detection"
bar_name = 'Thrshold'
path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/"
map_img = cv2.imread(path+"Map_A.jpg")
for j in sign_name :
    sign_img = cv2.imread(path+"img_test"+str(j)+".jpg",0)
    sign_imgs.append(sign_img)

def unsharp_image (img_):
    blur_img = cv2.GaussianBlur(img_,(5,5),0)
    unsharp_image = cv2.addWeighted(img_, 2, blur_img, -1, 0, img_)
    return  unsharp_image

def find_canny(low_thres_cannay,img_):
    kernel_size_cannay = 3
    ratio_ = 2
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(img_,5)
    blur_img = cv2.GaussianBlur(blur_img,(5,5),0)
    edges = cv2.Canny(blur_img, low_thres_cannay, low_thres_cannay*ratio_,kernel_size_cannay)
    return edges

def find_contours(edges_,mode_,area_thres):
    approx_thres = 0.08
    pass_data ={"cnts":[],"boxcoords":[],"approxs":[]}
    _,cnts_,_= cv2.findContours(edges_,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_ in cnts_:
        area = cv2.contourArea(cnt_)
        
        if area > area_thres  :
            
            pass_data["cnts"].append(cnt_)
            rect = cv2.boundingRect(cnt_)
            pass_data["boxcoords"].append(rect)
            epsilon = approx_thres * cv2.arcLength(cnt_, True)
            approx_ = cv2.approxPolyDP(cnt_, epsilon, True)
            pass_data["approxs"].append(approx_)
    
    return pass_data

def check_matching(img_,temp_,match_thres,cnt_,boxcoord_):
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

    
    
#cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name)    
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
thickness = 2
edge_mask = find_canny(22,map_img)
"""
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
"""        

linesP=cv2.HoughLinesP(edge_mask,1,np.pi/180, 100, 500, 40)

last_line =[0,0,0,0]
dis_chess =[0,0]
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(edge_mask, (l[0], l[1]), (l[2], l[3]), 0,1)
        dis_chess[0],dis_chess[1] = max(abs(last_line[0]-l[0]),abs(last_line[1]-l[1])),max(abs(last_line[2]-l[2]),abs(last_line[3]-l[3]))
        #if dis_chess[0]>10 and dis_chess[1] > 10:
        #    cv2.line(map_img, (l[0], l[1]), (l[2], l[3]), (0),2)
        last_line=l

dataset =find_contours(edge_mask,True,50)

for index in range(len(dataset["cnts"])) :
    print(len(dataset["approxs"][index]))
    x,y,w,h = dataset["boxcoords"][index]
    cv2.rectangle(map_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow(window_name, map_img)
    cv2.waitKey()
"""
while True:
    key = cv2.waitKey(10)
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break       
"""
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
