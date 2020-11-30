import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

path = "C:/Users/ASUS/OneDrive/My work/Project_module7/IMG_test/"
all_target_sym =[[5,"Star"],[3,"Triangle"],[4,"Rectangle"]]
window_name = "Edge Detection"
pre_map_img = cv2.imread(path+"Map_2A.jpg")
real_map_img = cv2.imread(path+"real_map_B1.jpg")


class Symbol:
    def __init__(self, name,mid,box):
        self.name = name
        self.mid = mid
        self.box = box

def unsharp_image (img_):
    blur_img = cv2.GaussianBlur(img_,(5,5),0)
    result = cv2.addWeighted(img_, 2, blur_img, -1, 0, img_)
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
            elif mode is False:
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


def find_symWithCorner(cntset_,target_syms,img_,mode_show):
    pass_sym =[]
    last_coord =[0,0,0]
    result = None
    for index in range(len(cntset_["cnts"])) :
        for sym_ in target_syms :
            if len(cntset_["approxs"][index])==sym_[0]:
                x,y,w,h = cntset_["boxcoords"][index]
                mid_ = [int(x+(w/2)),int(y+(h/2))]
                dis_chess = max([abs(last_coord[0]-mid_[0]),abs(last_coord[1]-mid_[1])])
                if dis_chess> last_coord[2] :
                    if mode_show:
                        result = cv2.rectangle(img_, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        result = cv2.putText(img_,sym_[1],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    last_coord = [mid_[0],mid_[1],max([w,h])]
                    new = Symbol(sym_[1],mid_ ,[x,y,w,h])
                    pass_sym.append(new)
    return pass_sym,result

low_thres_cannay = 90
kernel = np.ones((5,5),np.uint8)
scale_rect = 10

gray_map = cv2.cvtColor(pre_map_img, cv2.COLOR_BGR2GRAY)
_,binary_img = cv2.threshold(gray_map , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
gray_map = unsharp_image(gray_map)
edge_mask = cv2.Canny(gray_map, low_thres_cannay, low_thres_cannay*2,3)
edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
#edge_mask,_ =hough_transform(edge_mask,1000,10)
cntset =find_contours(edge_mask,True,[500,3000])
"""
print(cntset["approxs"])
for index in range(len(cntset["cnts"])):
    #if len(cntset["approxs"][index]) == 4:
    cv2.drawContours(pre_map_img, [cntset["cnts"][index]], 0,  (0,255,0), 2)
    cv2.imshow(window_name,pre_map_img)
    cv2.waitKey()
"""
syms,img =find_symWithCorner(cntset,all_target_sym,pre_map_img,True)


binary_img=cv2.bitwise_not(binary_img)
cntset =find_contours(binary_img,False,[0,0],False)
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)


for index in range(len(cntset["cnts"])):
    cv2.drawContours(binary_img, [cntset["cnts"][index]], 0, 255, -1)
erosion = cv2.erode(binary_img,kernel,iterations = 1)
skelton_mask = skeletonize(erosion, method='lee')


last_pt = [0,0]
# 150 133
w_real = 120
h_real = 120

for sym in syms :
    if sym.name == "Star":
        last_pt[0],last_pt[1] = sym.mid[0],sym.mid[1]
    x,y,w,h = sym.box
    #w,h = w+scale_rect,h+scale_rect
    w,h =  w_real,h_real
    x,y = sym.mid[0]-round(w/2),sym.mid[1]-round(h/2)
    skelton_mask[y:y+h,x:x+w] = 0
    #pre_map_img[y:y+h,x:x+w] = 100
    pts = np.argwhere(skelton_mask[y-1:y+h+1,x-1:x+w+1]==255)
    for index in range(len(pts[:2])) :
        delta_y = sym.mid[1]-(pts[index][0]+y)
        delta_x = pts[index][1]+x - sym.mid[0]
        #print(pts[index][1]+x,pts[index][0]+y)
        #print(sym.mid)
        if delta_x == 0 :
            if delta_y >= 0 :
                theta = 90.0
            else:
                theta = -90.0
        else :
            theta = np.arctan2(delta_y,delta_x) * 180 / np.pi
        #print(theta)
        if theta > 45.0 and theta <= 135.0 :
            pt = [sym.mid[0],y]
        elif theta > -45.0 and theta <= 45.0:
            pt = [x+w,sym.mid[1]]
        elif theta > -135.0 and theta <= -45.0 :
            pt = [sym.mid[0],y+h]
        else:
            pt = [x,sym.mid[1]]
        cv2.line(skelton_mask, (x-1+pts[index][1],y-1+pts[index][0]), (pt[0],pt[1]), 255,1)
        cv2.line(skelton_mask, (pt[0],pt[1]), (sym.mid[0],sym.mid[1]), 250,1)
        #cv2.imshow("Mask",skelton_mask)
        #cv2.imshow("IMG",pre_map_img)
        #cv2.waitKey()
    skelton_mask[sym.mid[1]][sym.mid[0]] = 250




#kernel = [[0,1,2],[0,2],[0,1,2]]
kernel = [[0,-1],[1,0],[0,1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]]
NB_check = False
deadRoad_check = False
list_path = []
list_path.append([])
list_path[0].append(last_pt.copy())
check_pnt = 0
index_path = 0
count = 0
while(1):
    #img = cv2.circle(skelton_mask.copy(),(pt[0],pt[1]) , 2, 255, 2)
    #cv2.imshow(window_name,img)
    for i in range(len(kernel)):
        x,y = last_pt[0]+kernel[i][0],last_pt[1]+kernel[i][1]
        if skelton_mask[y][x] >= 250 : 
            last_pt[0],last_pt[1] = x,y
            if skelton_mask[y][x] == 250 and check_pnt == 0 :
                list_path.append([])
                index_path += 1
                check_pnt = 1
                #print("Check point {}".format(list_path))
            if skelton_mask[y][x] != 250 :
                list_path[index_path].append(last_pt.copy())
                check_pnt = 0
            skelton_mask[y][x] -= 100
            NB_check = True
            break
        elif i == len(kernel)-1  :
            deadRoad_check = True
        if NB_check :
            NB_check = False
            break
    if deadRoad_check:
        #print ("Not find")
        break
print (len(list_path))
for skl in list_path:
    pnts_skel = np.array(skl)
    #pnts_skel = np.column_stack(np.where(skelton_mask.transpose()!=0))
    poly_pnts = cv2.approxPolyDP(pnts_skel,0.02*skelton_mask.shape[1],False)
    #print(poly_pnts[0][0])
    cv2.polylines(pre_map_img, [poly_pnts], False, (0,0,255), 1)
    print(len(pnts_skel))
    #print(pnts_skel)
    cv2.imshow("IMG",pre_map_img)
    cv2.waitKey()
#cv2.imshow("Mask",skelton_mask)

#for sym in syms :
#    print(sym.name)
#    print(skelton_mask[pnts_skel[-1][1]-3:pnts_skel[-1][1]-3+5,pnts_skel[-1][0]-3:pnts_skel[-1][0]-3+10])
    