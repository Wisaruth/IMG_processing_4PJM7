from Image import Image
from Preparation import BG_subtractor,Calibration
from Detection import Symbol,Target,Detection
from Control import Control,high_byte,low_byte
import cv2
import time
import serial
import numpy as np

def nothing(x):
    pass

path_folder = "C:/Users/wisar/OneDrive/My work/Project_module7/"
folder_var = "variable4IMG/"
folder_img = "IMG_test/BG_test/"

# Setup 
# X : 169 
#---------------------------------------------------------
state = 0       # mechanic state  
num_sample = 30 # Number of image for getting background
period = 0.5      # seconds 
camera_index = 2
mode_rotate = 0

main_window = "Image"
#---------------------------------------------------------
camera = Image(camera_index)           
cam_calib = Calibration()
field = BG_subtractor()
det = Detection()
com = Control(name= 'COM4')
com.uart1_connect()
index_saved_img = 0

det.offset =[-50,220]
det.ratio = 4
img = cv2.imread(path_folder+folder_img+"M1AK3.jpg")
camera.update_img(img)
det.update_target(name="tri",img=None,
                            corners=3,
                            area=0,
                            priori=1)
if camera.cap.isOpened() :
    camera.setting()
#---------------------------------------------------------
print("Main ---> Test")
print("Camera state : {}".format(camera.cap.isOpened()))
print ("State SYS : {}".format(state))

        
camera.hue = [90,120]
camera.sat = [40,100]
camera.val = [52,80]

#field.load_Aruco(path_folder+folder_var)
#------------------------------------
while(1):
    if   state == 0 : # Menu
        command = input("Enter :")
        #Find The Chessboard
        if command == "a1":     
            state = 1
            print ("----| State SYS : {} (Find The Chessboard)".format(state))
        #Show Calibration& ARUCO
        elif command == "target":
            state = 2
            print ("----| State SYS : {} (Set Targets)".format(state))
        elif command == "show":   
            state = 4
            print ("----| State SYS : {} (Show Calibration & ARUCO)".format(state))
        # Sampling image
        elif command == "sam":
            state = 5
            print ("----| State SYS : {} (Sampling For Getting The Field)".format(state))
        #Show the result of BG subtactor
        elif command == "median":   
            state = 7
            print ("----| State SYS : {} (Median The Field)".format(state))
        elif command == "reset":   
            state = 8
            print ("----| State SYS : {} (Reset)".format(state))
        elif command == "mid":   
            state = 9
            print ("----| State SYS : {} (Set mid)".format(state))
        elif command == "find":   
            state = 10
            print ("----| State SYS : {} (Find Sysmbols)".format(state))
        elif command == "hsv":   
            state = 11
            print ("----| State SYS : {} (Set HSV Threshold)".format(state))
        elif command == "path":   
            state = 12
            print ("----| State SYS : {} (Find Paths)".format(state))
        elif command == "go":   
            state = 13
            print ("----| State SYS : {} (Find Paths)".format(state))
        elif command == "set":
            print ("        Connect Camera/Robot    : 0")
            print ("        Set offset              : 1") 
            print ("        Set Counter of sampling : 2") 
            print ("        set period              : 3") 
            print ("        set rotation            : 4") 
            print ("--> Counter of sampling : {}".format(num_sample))
            print ("--> Deley of sampling : {}".format(period))
            print ("--> offset : {}".format(det.offset))
            command = input()
            if command == '0':
                camera.cap = cv2.VideoCapture(camera_index+cv2.CAP_DSHOW)
                if camera.cap.isOpened():
                    print("----> Connected")
                    camera.setting()
                else :
                    print("----> Disconnected")
                com.uart1_connect()
            elif command == '1':
                command = input("Z: ")
                det.offset[0] = int(command) 
                command = input("X: ")
                det.offset[1] = int(command)
                print("----> Setting Completed")   
            elif command == '2':
                command = input()
                num_sample = int(command)
                print("----> Setting Completed")
            # Set deley in sampling    
            elif command == '3':   
                command = input()
                period = float(command)
                print("----> Setting Completed")
            elif command == '4':   
                command = input()
                if command == 'true':
                    mode_rotate = True
                else :
                    mode_rotate = False
            else :
                print ("----> Unknow command")
        elif command == 'home':
            com.set_home_command()
        elif command == "save":         # save image in Image class' Object 
            print("Saved Image : {}".format(index_saved_img))
            command = input()
            ret = camera.saveImg(command,path_folder+folder_img)
            if ret :
                index_saved_img += 1
                print("----> Saving Completed")
            else :
                print("----> Saving Failed")  
        elif command == "exit":
            break
        else :
            print ("Not Find this command, try agin")

    elif state == 1 : # Find The Chessboard and Show where it is on an image
        if ret :
            img=cam_calib.show_chessboard(img)
            cv2.imshow("Find The Chessboard",img)
        key = cv2.waitKey(30)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Find The Chessboard")
            print ("        Camera : {}".format(ret))
            print ("----| State SYS : {}".format(state))
    
    elif state == 2:
        ret,img = camera.cap.read()
        #ret = True
        #img = camera.image
        if ret :
            cv2.imshow("Crop",img)
            aruco_ret,field_img,_ = field.cropWith_aruco(img.copy(),False)
            if aruco_ret :
                img = field_img
            x,y,w,h = cv2.selectROI("Crop", img, True)
            if w and h :
                #print(w,h)
                crop_img = img[y:y+h,x:x+w]
                target = camera.show_OTSU(crop_img.copy(),6)
                target = 255 - target
                set_sym = det.find_contours(target,mode=True,area_thres=500)
                cv2.destroyWindow("Crop")
                cv2.imshow("Target",target)
                print(crop_img.shape)
                print(set_sym["area"])
                for index in range(len(set_sym["area"])):
                    x,y,w,h = set_sym["boxcoords"][index]
                    crop_img = cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("Show",crop_img)
                    key = cv2.waitKey()
                    if key == ord('s'):
                        command = input("Enter name:")
                        priori = input("Enter Priority 0 or 1:")
                        target = target[y:y+h,x:x+w]
                        #cv2.imwrite(path_folder+folder_img +"Target01Test.jpg",target) 
                        det.update_target(name=command,img=target,
                                            corners=len(set_sym["approxs"][index]),
                                            area=set_sym["area"][index],
                                            priori=int(priori))
                        break
                print(det.targets[0].corners)
                cv2.waitKey()
            state = 0
            cv2.destroyWindow("Show")
            cv2.destroyWindow("Target")
            print ("----| State SYS : {}".format(state))
    
    elif state == 4: # Show the result from Calibration & Find ARUCO markers and then show them
        ret,img = camera.cap.read()    
        if ret:
            cv2.imshow(main_window,img)
            aruco_ret,field_img,aruco_img = field.cropWith_aruco(img.copy(),True)  # Find ARUCO
            if aruco_ret :
                if mode_rotate :
                    field_img = cv2.rotate(field_img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow("Crop Image",field_img)
            if aruco_img is not None :
                cv2.imshow("Image",aruco_img)
            cam_calib.show_chessboard(img)
        key = cv2.waitKey(10)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Crop Image")
            cv2.destroyWindow(main_window)
            print ("----| State SYS : {}".format(state))

    elif state == 5: # Sampling an image and then Median them all ---| num_sample and period are the input
        count = 0
        #cv2.destroyWindow("Image")
        command = input("Enter Name:")
        if com.port_connected:
            com.posx = 100
            com.posz = 1140
            data = serial.to_bytes(
                [0x46, 0x58, high_byte(com.posx), low_byte(com.posx), 0x5A, high_byte(com.posz),
                 low_byte(com.posz), 0x53])
            com.ser.write(data)
            print("X:", com.posx, "Z:", com.posz)
            print("Passcode: ", data)
            while(1):
                ret,img = camera.cap.read()
                if ret :       # Sampling
                    if field.Height is not None:
                        if mode_rotate :
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
                        img = field.add_imgset(img.copy())
                        if img is not False :
                            cv2.imwrite(path_folder+folder_img+command+str(count)+".jpg",img)
                if count== num_sample:
                    break
                time.sleep(period)
                count+= 1
            if len(field.imgset) != 0 :
                camera.update_img(field.median2getBG())
                state = 6 
            else :
                state = 0
            print(com.ser.readline().decode())
        else:
            print("Warning: Serial Port", com.COM_PORT, "is not opened.")
            state = 0
        print ("----| State SYS : {}".format(state))
    
    elif state == 6: # Show images from Sampling
        count = 0
        print ("---->  Image in set : {}".format(len(field.imgset)))
        for img in field.imgset :
            count+=1
            print ("    Image {}".format(count))
            cv2.imshow("Image in set",img)
            key = cv2.waitKey()
            if key == ord('q') :
                break
        state = 7
        cv2.destroyWindow("Image in set")

    elif state == 7: # Show images from Median
        cv2.imshow("Result1",camera.image)
        cv2.imshow("Result2",camera.bin_image)
        key = cv2.waitKey(30)
        if key == ord('q') or camera.image is None :       
            state = 0
            cv2.destroyWindow("Result1")
            cv2.destroyWindow("Result2")
            print ("----| State SYS : {}".format(state))
    
    elif state == 8:
        field.imgset = []
        state = 0
        print ("----| State SYS : {}".format(state))

    elif  state == 9:
        com.posx = 40
        com.posy = 0
        com.posz = 300
        data = serial.to_bytes([0x46, 0x58, high_byte(com.posx), low_byte(com.posx),
                                    0x5A, high_byte(com.posz), low_byte(com.posz),
                                    0x59, high_byte(com.posy), low_byte(com.posy), 0x53])
        print("X:", com.posx, "Z:", com.posz, "Y:", com.posy)
        #data = serial.to_bytes(
        #    [0x46, 0x58, high_byte(0), low_byte(0), 0x5A, high_byte(com.posmidz),
        #     low_byte(com.posmidz), 0x53])
        #print("X: 0", "Z:", com.posmidz)
        print("Passcode: ", data)
        if com.port_connected:
            com.ser.write(data)
            print(com.ser.readline().decode())
        else:
            print("Warning: Serial Port", com.COM_PORT, "is not opened.")
        state = 0
        print ("----| State SYS : {}".format(state))

    elif state == 10 :
        cntset = det.find_contours(camera.bin_image,mode=True,area_thres=1000)
        img = det.find_symWithCorner(cntset,camera.image.copy())
        if img is not None :
            cv2.imshow("Result",img)
            cv2.waitKey()
        state = 0
        print ("----| State SYS : {}".format(state))
        cv2.destroyWindow("Result")

    elif state == 11:
        cv2.namedWindow("Bar")
        cv2.createTrackbar("UpHue","Bar",0,255,nothing)
        cv2.createTrackbar("LowHue","Bar",0,255,nothing)
        cv2.createTrackbar("UpSat","Bar",0,100,nothing)
        cv2.createTrackbar("LowSat","Bar",0,100,nothing)
        cv2.createTrackbar("UpVal","Bar",0,100,nothing)
        cv2.createTrackbar("LowVal","Bar",0,100,nothing)
        while True:
            key = cv2.waitKey(30)
            up_hue = cv2.getTrackbarPos("UpHue","Bar")
            low_hue = cv2.getTrackbarPos("LowHue","Bar")
            up_sat = cv2.getTrackbarPos("UpSat","Bar")
            low_sat = cv2.getTrackbarPos("LowSat","Bar")
            up_val = cv2.getTrackbarPos("UpVal","Bar")
            low_val = cv2.getTrackbarPos("LowVal","Bar")
            camera.hue = [low_hue,up_hue]
            camera.sat = [low_sat,up_sat]
            camera.val = [low_val,up_val]
            clrs_img = camera.color_detection (500)
            cv2.imshow("Result", clrs_img)
            if key == ord('q') or key == ord('Q') :
                cv2.destroyWindow("Result")
                cv2.destroyWindow("Bar")
                break
        #hsv_blue = [90,120]
        #sat = [40,100]
        #val = [52,80]
        
        # hsv_or = [0,152]
        # sat_or = [0,34]
        # val_or = [0,83]
        state = 0
        print ("----| State SYS : {}".format(state))

    elif state == 12 :
        hsv_img,clr_mask = camera.color_detection (1000)
        binary_img = 255- camera.bin_image
        #cv2.imshow("CLR1",binary_img)
        kernel = np.ones((25,25),np.uint8) 
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((10,10),np.uint8)
        binary_img = cv2.dilate(binary_img,kernel,iterations = 1)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        binary_img=cv2.bitwise_not(binary_img)
        mask = cv2.bitwise_or(binary_img, clr_mask)
        
        #cv2.imshow("CLR2",clr_mask)
        #cv2.imshow("CLR3",mask)
        mask = camera.skelton_mask(mask,7)
        mask = det.reconstruct_map(mask,82,78)
        xz_paths,order_syms,res = det.XZ_path_generator(mask,camera.image.copy(),ep = 0.5)
        
        
        cv2.namedWindow("Path")
        cv2.createTrackbar("Deriva","Path",30,100,nothing)
        cv2.createTrackbar("Count","Path",30,100,nothing)
        cv2.imshow("Result", res)
        while(1): # Deriva : 19 , Count : 9 
            det.paths = []
            img = res.copy()
            key = cv2.waitKey(30)
            deri = cv2.getTrackbarPos("Deriva","Path")
            count_path = cv2.getTrackbarPos("Count","Path")
            _ = det.Y_path_generator(img=img,XZlines=xz_paths,hsv_map=hsv_img,max_deriva=deri,
                                    max_count=count_path,order_syms_pnts=order_syms)
            cv2.imshow("Path", img)
            if key == ord('q') :
                break
        #_ = det.Y_path_generator(img=res,XZlines=xz_paths,hsv_map=hsv_img,max_deriva=19,
        #                            max_count=9,order_syms_pnts=order_syms)
        cv2.imshow("Result", res)
        cv2.waitKey() 
        cv2.destroyWindow("Result")
        cv2.destroyWindow("CLR1")
        cv2.destroyWindow("CLR2")
        cv2.destroyWindow("CLR3")
        cv2.destroyWindow("Path")
        det.convert2World(hight =hsv_img.shape[0],min_R=16,max_R=28)
        print(det.paths[0])
        state = 0
        print ("----| State SYS : {}".format(state))

    elif state == 13:
        if com.port_connected:
            pnt = det.paths[0][0]
            print(pnt)
            com.posx,com.posz,com.posy = pnt[1],pnt[0],0
            com.send_Y()
            com.send_XZ()
            for path in det.paths:
                for pnt in path:
                    print(pnt)
                    com.posx,com.posz,com.posy = pnt[1],pnt[0],pnt[2]
                    com.send_Y()
                    com.send_XZ()
        #print("X:", com.posx, "Z:", com.posz)
        #print(det.paths[0])
        
        #cv2.destroyWindow("Path")
        state = 0
        print ("----| State SYS : {}".format(state))

 
   



        

        
    
        


        

        

        

