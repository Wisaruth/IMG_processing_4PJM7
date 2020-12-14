import cv2
from skimage.morphology import skeletonize
import math
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass


class Symbol:
    def __init__(self, name,mid,box):
        self.name = name
        self.mid = mid
        self.box = box

def unsharp_image (img_):
    blur_img = cv2.GaussianBlur(img_,(5,5),0)
    result = cv2.addWeighted(img_, 2, blur_img, -1, 0, img_)
    return result

def find_canny(low_thres_cannay,gray_img):
    kernel_size_cannay = 3
    ratio_ = 2
    #gray_img = cv2.medianBlur(gray_img,5)
    
    #gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    
    result = cv2.Canny(gray_img, low_thres_cannay, low_thres_cannay*ratio_,kernel_size_cannay)
    return result

def find_contours(edges_,mode,area_thres,thick_=None):
    approx_thres = 0.08
    pass_sym ={"cnts":[],"boxcoords":[],"approxs":[],"area":[]}
    if thick_ is not None:
        kernel = np.ones((3,3), np.uint8) 
        edges_ = cv2.dilate(edges_, kernel, iterations=thick_)
    if mode :
        _,cnts_,_= cv2.findContours(edges_,cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    else :
        _,cnts_,_= cv2.findContours(edges_,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt_ in cnts_:
        area = cv2.contourArea(cnt_)
        if area > area_thres[0]   :
            if  mode is True and area < area_thres[1]:
                rect = cv2.boundingRect(cnt_)
                epsilon = approx_thres * cv2.arcLength(cnt_, True)
                approx_ = cv2.approxPolyDP(cnt_, epsilon, True)
                pass_sym["approxs"].append(approx_)
                pass_sym["boxcoords"].append(rect)
            pass_sym["area"].append(area)
            pass_sym["cnts"].append(cnt_)
    return pass_sym

def hough_transform (edges_,minpoint,maxgap):# 500, 20
    linesP=cv2.HoughLinesP(edges_,1,np.pi/180, 100, minpoint, maxgap)
    last_line =[0,0,0,0]
    dis_chess =[0,0]
    pass_line =[]
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(edges_, (l[0], l[1]), (l[2], l[3]), 0,2)
            dis_chess[0],dis_chess[1] = max(abs(last_line[0]-l[0]),abs(last_line[1]-l[1])),max(abs(last_line[2]-l[2]),abs(last_line[3]-l[3]))
            if dis_chess[0]>10 and dis_chess[1] > 10:
                pass_line.append(l)
            last_line=l
    return edges_,pass_line


def check_matching(img_,temp_,match_thres,cnt_,boxcoord_):
    x,y,w,h = boxcoord_
    crop_img = img_[y:y+h,x:x+w]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.resize(crop_img, (temp_.shape[1], temp_.shape[0]),
                            interpolation = cv2.INTER_LINEAR)
    _,crop_img = cv2.threshold(crop_img,127,255,cv2.THRESH_BINARY)
    res = cv2.matchTemplate(crop_img,temp_,cv2.TM_CCOEFF_NORMED)
    _, max_val,_,_ = cv2.minMaxLoc(res)
    if max_val >= match_thres:
        return True
    else :
        return False

def find_symWithCorner(cntset_,target_syms,img_,mode_show):
    pass_sym =[]
    last_coord =[0,0,0]
    result = None
    for index in range(len(cntset_["cnts"])) :
        for sym_ in target_syms :
            if len(cntset_["approxs"][index])==sym_[0]:
                x,y,w,h = cntset_["boxcoords"][index]
                mid_ = [int((x+w)/2),int((y+h)/2)]
                dis_chess = max([abs(last_coord[0]-mid_[0]),abs(last_coord[1]-mid_[1])])
                if dis_chess> last_coord[2] :
                    if mode_show:
                        result = cv2.rectangle(img_, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        result = cv2.putText(img_,sym_[1],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    last_coord = [mid_[0],mid_[1],max([w,h])]
                    new = Symbol(sym_[1],mid_ ,[x,y,w,h])
                    pass_sym.append(new)
    return pass_sym,result

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)    
    return result
              


all_target_sym =[[5,"Star"],[3,"Triangle"],[4,"Rectangle"]]
window_name = "Edge Detection"
path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/"
map_img = cv2.imread(path+"Map_2A.jpg")
for i in range(5,360,5):
    img = rotate_image(map_img,i)
    cv2.imshow("IMG",img)
    print(img.shape)
    cv2.waitKey()

"""
gray_map = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
_,threshold = cv2.threshold(gray_map , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
threshold=cv2.bitwise_not(threshold)
cntset =find_contours(threshold,False,[0,0],False)
print(len(cntset["cnts"]))
for index in range(len(cntset["cnts"])):
    cv2.drawContours(threshold, [cntset["cnts"][index]], 0, 255, -1)
erosion = cv2.erode(threshold,kernel,iterations = 1)
skeleton_lee = skeletonize(erosion, method='lee')
cv2.imshow(window_name,skeleton_lee)
cv2.waitKey()
"""       

"""
gray_map = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow(window_name) 
kernel = np.ones((3,3),np.uint8) 
gray_map = unsharp_image(gray_map)
edge_mask = find_canny(90,gray_map)
edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
edge_mask,_ =hough_transform(edge_mask,1000,10)
cntset =find_contours(edge_mask,True,[500,3000])
findsyms,img =find_symWithCorner(cntset,all_target_sym,map_img,True)
cv2.imshow(window_name,img)
cv2.waitKey()
"""

"""
cv2.namedWindow(window_name)
bar_name = 'Thrshold'
cv2.createTrackbar(bar_name,window_name,0,100,nothing)
kernel = np.ones((3,3),np.uint8)
while True:
    key = cv2.waitKey(10)
    #ret,frame = cap.read()
    val = cv2.getTrackbarPos(bar_name,window_name)
    
    gray_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    gray_img = unsharp_image(gray_img)
    detected_edges = cv2.Canny(gray_img, val, val*2,3)
    detected_edges = cv2.morphologyEx(detected_edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow(window_name, detected_edges)
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break
"""

"""
sign_imgs = []
sign_name = [0,2,8]
for j in sign_name :
    sign_img = cv2.imread(path+"img_test"+str(j)+".jpg",0)
    sign_imgs.append(sign_img)
sign_name = ["Star","Triangle","Rectangle"]
last_coord = [0,0,0]

gray_map = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
gray_map = unsharp_image(gray_map)
edge_mask = find_canny(22,gray_map)
cntset=find_contours(edge_mask,True,[500,3000])
for j in range(len(sign_name)):
    for i in range(len(contours)):
        check= check_matching(map_img,sign_imgs[j],0.5,contours[i],coordinate[i])
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