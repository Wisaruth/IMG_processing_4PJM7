import numpy as np
import cv2


class Symbol:
    def __init__(self, name,mid,corner,order):
        self.name = name
        self.mid = mid
        self.corner = corner
        self.order = order

class Target:
    def __init__(self, name,num_corner=0,img = None,area = 0, priori=0):
        self.name = name
        self.corners = num_corner
        self.area = area
        self.template = img
        self.priori = priori
    
    
        
    
class Detection:
    def __init__(self):
        self.symbols = []
        self.targets = []
        self.ratio = 0
        self.offset = [0,0]
        self.paths = []
        self.near_kernel = [[0,-1],[1,0],[0,1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]]
    
    # def set_worldCoordi (self,Onecm2pix,Onecm2count):
    #    self.count2pix_ratio = Onecm2count/Onecm2pix
    def update_target (self,name,img,corners,area,priori):
        new = Target(name,corners,img,area,priori)
        self.targets.append(new)

    def convert2World (self,hight,min_R=100,max_R=190):
        
        for path_index in range(len(self.paths)) :
            inten = []
            last_inten = None
            for pnt_index in range(len(self.paths[path_index])):
                self.paths[path_index][pnt_index][0] = self.paths[path_index][pnt_index][0]*self.ratio+self.offset[0]
                self.paths[path_index][pnt_index][1] = (hight-self.paths[path_index][pnt_index][1])*self.ratio+self.offset[1]
                if self.paths[path_index][pnt_index][2] == 0:
                    if last_inten is None :
                        for i in  range(pnt_index,len(self.paths[path_index])):
                            if self.paths[path_index][pnt_index+i][2] != 0:
                                self.paths[path_index][pnt_index][2] = self.paths[path_index][pnt_index+i][2]
                                break
                    else :
                        self.paths[path_index][pnt_index][2] = last_inten 
                else:
                    last_inten= self.paths[path_index][pnt_index][2]
                    inten.append(last_inten) 
            max_in,min_in=max(inten),min(inten)
            range_r=max_R-min_R
            ratio=(max_in-min_in)/range_r
            for pnt_index in range(len(self.paths[path_index])):
                self.paths[path_index][pnt_index][2] =round(((self.paths[path_index][pnt_index][2]-min_in)/ratio)+min_R)
     
    
    def rotation(self,img,angle):
        img_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)    
        return result
    
    # Canny with high boost filter
    def find_contours(self,edges,mode,area_thres):
        approx_thres = 0.08
        pass_sym ={"cnts":[],"boxcoords":[],"approxs":[],"area":[]}
        if mode :
            _, cnts, _= cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else :
            _, cnts, _= cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > area_thres :#and area < area_thres[1]  :
                pass_sym["cnts"].append(cnt)
                if  mode is True :
                    rect = cv2.boundingRect(cnt)
                    epsilon = approx_thres * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    pass_sym["approxs"].append(approx)
                    pass_sym["boxcoords"].append(rect)
                    pass_sym["area"].append(area)
        return pass_sym

    def check_matching(self,img,name_target,match_thres,boxcoord):
        template = None
        for target in self.targets:
            if target.name == name_target :
                template = target.template
                break
        if template is not None :
            x,y,w,h = boxcoord
            crop_img = img[y:y+h,x:x+w]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = cv2.resize(crop_img, (template.shape[1], template.shape[0]),
                            interpolation = cv2.INTER_LINEAR)
            res = cv2.matchTemplate(crop_img,template,cv2.TM_CCOEFF_NORMED)
            _, max_val,_,_ = cv2.minMaxLoc(res)
            if max_val >= match_thres:
                return True
            else :
                return False
        else :
            return False
    
    def find_symWithCorner(self,cntset,img = None):
        last_coord =[0,0,0]
        result = None
        for index in range(len(cntset["cnts"])) :
            for target in self.targets :
                if len(cntset["approxs"][index])== target.corners:
                    x,y,w,h = cntset["boxcoords"][index]
                    mid_ = [int(x+(w/2)),int(y+(h/2))]
                    dis_chess = max([abs(last_coord[0]-mid_[0]),abs(last_coord[1]-mid_[1])])
                    if dis_chess> last_coord[2] :
                        if img is not None:
                            result = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
                            result = cv2.putText(result,target.name,(x,y),
                                                 cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        last_coord = [mid_[0],mid_[1],max([w,h])]
                        if target.priori == 1 :
                            order = 0
                        else :
                            order = 1
                        new = Symbol(name= target.name, 
                                        mid = mid_, 
                                        corner = [x,y,w,h],
                                        order=order)
                        self.symbols.append(new)
        return result  


    def reconstruct_map (self,skelton_map,w,h):
        for index in range(len(self.symbols)) :
            sym = self.symbols[index]
            if sym.order == 0:
                self.symbols.pop(index)
                self.symbols.insert(0,sym)
            x,y = sym.mid[0]-round(w/2),sym.mid[1]-round(h/2)
            skelton_map[y:y+h,x:x+w] = 0
            pts = np.argwhere(skelton_map[y-1:y+h+1,x-1:x+w+1]==255)
            for index in range(len(pts[:2])) :
                delta_y = sym.mid[1]-(pts[index][0]+y)
                delta_x = pts[index][1]+x - sym.mid[0]
                theta = round(np.arctan2(delta_y,delta_x) * 180 / np.pi)
                if theta > 45 and theta <= 135 :
                    pt = [sym.mid[0],y]
                elif theta > -45 and theta <= 45:
                    pt = [x+w,sym.mid[1]]
                elif theta > -135 and theta <= -45 :
                    pt = [sym.mid[0],y+h]
                else:
                    pt = [x,sym.mid[1]]
                cv2.line(skelton_map, (x-1+pts[index][1],y-1+pts[index][0]), (pt[0],pt[1]), 255,1)
                cv2.line(skelton_map, (pt[0],pt[1]), (sym.mid[0],sym.mid[1]), 250,1)
        skelton_map[sym.mid[1]][sym.mid[0]] = 251
        return skelton_map
    
    def XZ_path_generator (self,skelton_mask,img,ep):
        index_path = 0
        found_check = False
        deadRoad_check = False
        check_pnt = False
        last_pt = self.symbols[0].mid.copy()
        poly_lines =[]
        lines = []
        order_syms_pnts = []
        order_syms_pnts.append(last_pt.copy())
        if img is not None:    
            cv2.circle(img,(last_pt[0],last_pt[1]) , 2, (0,255,0), 2)
        while(1):
            for i in range(len(self.near_kernel)):
                x,y = last_pt[0]+self.near_kernel[i][0],last_pt[1]+self.near_kernel[i][1]
                if skelton_mask[y][x] >= 250 : 
                    skelton_mask[last_pt[1]][last_pt[0]] = 0
                    last_pt[0],last_pt[1] = x,y
                    if skelton_mask[y][x] != 255 and check_pnt == False :
                        lines.append([])
                        check_pnt = True
                        if len(lines) != 1 :
                            index_path += 1
                    if skelton_mask[y][x] == 255 :
                        lines[index_path].append(last_pt.copy())
                        check_pnt = False
                    if skelton_mask[y][x] == 251:
                        order_syms_pnts.append(last_pt.copy())
                    found_check = True
                    break
                elif i == 7  :
                    deadRoad_check = True
                if found_check :
                    found_check = False
                    break
            if deadRoad_check:
                break
        for pnts in lines:
            pnts = np.array(pnts)
            pnts = cv2.approxPolyDP(pnts,ep*skelton_mask.shape[1],False)
            poly_lines.append(pnts)
            if img is not None:
                for pnt in pnts :
                    cv2.circle(img,(pnt[0][0],pnt[0][1]) , 2, (0,255,0), 2)
        return  poly_lines,order_syms_pnts,img

    
    def Y_path_generator (self,img,XZlines,hsv_map,max_deriva,max_count,order_syms_pnts):
        list_color = []
        last_pt = [0,0]
        run_set= [0,0,1]
        index_input = True
        index_line = 0
        for line in XZlines :
            if len(line) > 1 :
                self.paths.append([])
                for index in range(len(line)-1):
                    delta_y = line[index][0][1] - line[index+1][0][1] 
                    delta_x = line[index+1][0][0] - line[index][0][0]
                    theta = abs(round(np.arctan2(delta_y,delta_x) * 180 / np.pi))
                    delta_y *= -1
                    if index == 0 :
                        self.paths[index_line].append([line[index][0][0],line[index][0][1],hsv_map[line[index][0][1]][line[index][0][0]][2],theta])
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
                                deriva -= hsv_map[y][x][2]
                            else :
                                deriva += hsv_map[y][x][2]
                        if hsv_map[y][x][2] != 0  :
                            all_deriva += deriva
                            if deriva > 0 :
                                abs_check = "+"
                            elif deriva < 0:
                                abs_check = "-"
                            else:
                                abs_check = "0"
                
                            if clear_check == None :
                                clear_check =  abs_check
                            elif clear_check != abs_check  :
                                count += 1
                                if count == 1 :
                                    last_pt[0],last_pt[1] = x,y 
                            elif count > 0 :
                                count -= 1

                            if abs(all_deriva) > max_deriva :
                                all_deriva = 0
                                self.paths[index_line].append([x,y,hsv_map[y][x][2],theta])
                                cv2.circle(img,(x,y) , 2, (0,0,255), 2)
                            if count == max_count :
                                count = 0
                                clear_check = abs_check
                                self.paths[index_line].append([last_pt[0],last_pt[1],hsv_map[last_pt[1]][last_pt[0]][2],theta])
                                cv2.circle(img,(x,y) , 2, (0,0,255), 2)
                    self.paths[index_line].append([line[index+1][0][0],line[index+1][0][1],hsv_map[line[index+1][0][1]][line[index+1][0][0]][2],theta])
                index_line +=1
        num = len(self.paths)
        for i in range(num): 
            inten = self.paths[i][0][2]
            self.paths[i] = [[order_syms_pnts[i][0],order_syms_pnts[i][1],inten,self.paths[i][0][3]]] + self.paths[i]
            if i+1 < num :
                self.paths[i].append([order_syms_pnts[i+1][0],order_syms_pnts[i+1][1],inten,self.paths[i][-1][3]])


        return list_color


        
