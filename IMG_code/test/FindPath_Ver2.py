import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/"
all_target_sym =[[5,"Star"],[3,"Triangle"],[4,"Rectangle"]]
window_name = "Edge Detection"
pre_map_img = cv2.imread(path+"Map_2A.jpg")
real_map_img = cv2.imread(path+"real_map_B1.jpg")
hvs_map = cv2.cvtColor(pre_map_img,cv2.COLOR_BGR2HSV)
hvs_map = cv2.GaussianBlur(hvs_map,(5,5),0)

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

#cv2.imshow("Mask",skelton_mask)
#cv2.waitKey()

last_pt = [0,0]
w_real = 140
h_real = 140

for sym in syms :
    if sym.name == "Star":
        last_pt[0],last_pt[1] = sym.mid[0],sym.mid[1]
    x,y,w,h = sym.box
    w,h =  w_real,h_real
    x,y = sym.mid[0]-round(w/2),sym.mid[1]-round(h/2)
    skelton_mask[y:y+h,x:x+w] = 0
    pts = np.argwhere(skelton_mask[y-1:y+h+1,x-1:x+w+1]==255)
    for index in range(len(pts[:2])) :
        delta_y = sym.mid[1]-(pts[index][0]+y)
        delta_x = pts[index][1]+x - sym.mid[0]
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
    skelton_mask[sym.mid[1]][sym.mid[0]] = 251





#kernel = [[0,1,2],[0,2],[0,1,2]]
index_path = 0
count = 0
kernel = [[0,-1],[1,0],[0,1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]]
NB_check = False
deadRoad_check = False
check_pnt = False
list_path = []
order_syms_pnts = []
order_syms_pnts.append(last_pt.copy())
while(1):
    for i in range(len(kernel)):
        x,y = last_pt[0]+kernel[i][0],last_pt[1]+kernel[i][1]
        if skelton_mask[y][x] >= 250 : 
            skelton_mask[last_pt[1]][last_pt[0]] = 0
            last_pt[0],last_pt[1] = x,y
            if skelton_mask[y][x] != 255 and check_pnt == False :
                list_path.append([])
                if len(list_path) != 1 :
                    index_path += 1
                check_pnt = True
            if skelton_mask[y][x] == 255 :
                list_path[index_path].append(last_pt.copy())
                check_pnt = False
            if skelton_mask[y][x] == 251:
                order_syms_pnts.append(last_pt.copy())
            NB_check = True
            break
        elif i == 7  :
            deadRoad_check = True
        if NB_check :
            NB_check = False
            break
    if deadRoad_check:
        break


poly_lines =[]
for skl in list_path:
    pnts_skel = np.array(skl)
    poly_pnts = cv2.approxPolyDP(pnts_skel,0.02*skelton_mask.shape[1],False)
    poly_lines.append(poly_pnts)

#cv2.polylines(skelton_mask, poly_lines,False, 255, 1)    
for line in poly_lines :
    for pnts in line :
        cv2.circle(pre_map_img,(pnts[0][0],pnts[0][1]) , 2, (255,0,0), 2)
        #skelton_mask[pnts[0][1]][pnts[0][0]] = 255

new_poly_lines = []
list_color = []
last_pt = [0,0]
run_set= [0,0,1]
index_input = True
index_line = 0

for line in poly_lines :
    if len(line) > 1 :
        new_poly_lines.append([])
        for index in range(len(line)-1):
            delta_y = line[index][0][1] - line[index+1][0][1] 
            delta_x = line[index+1][0][0] - line[index][0][0]
            theta = abs(round(np.arctan2(delta_y,delta_x) * 180 / np.pi))
            delta_y *= -1
            if index == 0 :
                new_poly_lines[index_line].append([line[index][0][0],line[index][0][1],hvs_map[line[index][0][1]][line[index][0][0]][1],theta])
            all_deriva = 0
            count = 0
            clear_check = None
            abs_check = None
            if delta_x == 0 :
                m = 0
            else :
                m = delta_y/delta_x
            if  abs(delta_y) > abs(delta_x) :
                index_input = True
                run_set[0],run_set[1] = line[index][0][1],line[index+1][0][1]
                run_set[2] = 1
                if delta_y < 0 :
                    run_set[2] = -1
            else :
                index_input = False
                run_set[0],run_set[1] = line[index][0][0],line[index+1][0][0]
                run_set[2] = 1
                if delta_x < 0 :
                    run_set[2] = -1  
            
            for i in range(run_set[0]+run_set[2],run_set[1],run_set[2]):
                deriva = 0
                for j in [1,0]:
                    j *= run_set[2] 
                    if index_input :
                        x = round((i+j-line[index][0][1])/m + line[index][0][0])
                        y = i+j
                    else :
                        y = round((i+j-line[index][0][0])*m + line[index][0][1])
                        x = i+j
                    if j == 0:
                        deriva -= hvs_map[y][x][1]
                    else :
                        deriva += hvs_map[y][x][1]

                all_deriva += deriva
                if deriva > 0 :
                    abs_check = "+"
                elif deriva < 0:
                    abs_check = "-"
                elif deriva == 0:
                    abs_check = "0"
                
                if clear_check == None :
                    clear_check =  abs_check
                elif clear_check != abs_check  :
                    count += 1
                    if count == 1 :
                        last_pt[0],last_pt[1] = x,y 
                elif count > 0 :
                    count -= 1

                if abs(all_deriva) > 40 :
                    all_deriva = 0
                    new_poly_lines[index_line].append([x,y,hvs_map[y][x][1],theta])
                    pre_map_img = cv2.circle(pre_map_img,(x,y) , 2, (0,0,255), 2)
                if count == 10 :
                    count = 0
                    clear_check = abs_check
                    new_poly_lines[index_line].append([last_pt[0],last_pt[1],hvs_map[last_pt[1]][last_pt[0]][1],theta])
                    pre_map_img = cv2.circle(pre_map_img,(last_pt[0],last_pt[1]) , 2, (0,0,255), 2)
                #img = cv2.circle(pre_map_img.copy(),(x,y) , 2, (0,0,255), 2)
                #print("Pt : {} {} --- {} -- {}".format(x,y,hvs_map[y][x][1],all_deriva))
                #cv2.imshow("IMG",img)
                #cv2.imshow("Mask",skelton_mask)
                #key = cv2.waitKey()
                #if key == ord('q') or key == ord('Q') :
                    #break  
            new_poly_lines[index_line].append([line[index+1][0][0],line[index+1][0][1],hvs_map[line[index+1][0][1]][line[index+1][0][0]][1],theta])
        index_line +=1

pre_map_img = cv2.circle(pre_map_img,(192, 372) , 2, (0,255,255), 2)
print(new_poly_lines[0])
print(new_poly_lines[1])
print(new_poly_lines[2])
cv2.imshow("IMG",img)  
cv2.waitKey()       

num = len(new_poly_lines)
for i in range(num):
    inten = new_poly_lines[i][0][2]
    new_poly_lines[i] = [[order_syms_pnts[i][0],order_syms_pnts[i][1],inten,new_poly_lines[i][0][3]]] + new_poly_lines[i]
    if i+1 < num :
        new_poly_lines[i].append([order_syms_pnts[i+1][0],order_syms_pnts[i+1][1],inten,new_poly_lines[i][-1][3]])

print("A")
print(new_poly_lines[0])
print(new_poly_lines[1])
print(new_poly_lines[2])

"""
            for i in range(run_set[0]+run_set[2],run_set[1],run_set[2]):
                if index_input :
                    x = round((i-line[index][0][1])/m + line[index][0][0])
                    y = i
                else :
                    y = round((i-line[index][0][0])*m + line[index][0][1])
                    x = i
                if last_inten == None :
                    last_inten = int(hvs_map[y][x][1])
                    new_poly_lines.append([line[index][0][0],line[index][0][1],last_inten,theta])
                diff =  hvs_map[y][x][1] - last_inten
                if abs(diff) >= 45 :
                    count += 1
                    if count == 1 :
                        last_pt[0],last_pt[1] = x,y
                else :
                    count = 0
                if count == 3 :
                    count = 0
                    last_inten = int(hvs_map[y][x][1])
                    new_poly_lines.append([last_pt[0],last_pt[1],hvs_map[last_pt[1]][last_pt[0]][1],theta])
                    pre_map_img = cv2.circle(pre_map_img,(x,y) , 2, (0,255,0), 2)
                    mat_check = True
                print("Pt : {} {} --- {} --- {}".format(x,y,hvs_map[y][x][1],diff))
                img = cv2.circle(pre_map_img.copy(),(x,y) , 2, (0,0,255), 2)
                cv2.imshow("IMG",img)
                #cv2.imshow("Mask",skelton_mask)
                key = cv2.waitKey()
                if key == ord('q') or key == ord('Q') :
                    break  
"""                  