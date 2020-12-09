from Image import Image
from Preparation import BG_subtractor,Calibration
import cv2
import time
import serial
from Control import Control,high_byte,low_byte

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
#---------------------------------------------------------
camera = Image(camera_index)           
cam_calib = Calibration()
field = BG_subtractor()
com = Control(name= 'COM4')
com.uart1_connect()
index_saved_img = 0

if camera.cap.isOpened() :
    camera.setting()
#---------------------------------------------------------
print("Main ---> Test")
print("Camera state : {}".format(camera.cap.isOpened()))
print ("State SYS : {}".format(state))


#field.load_Aruco(path_folder+folder_var)
#------------------------------------
while(1):
    if   state == 0 : # Menu
        command = input()
        #Find The Chessboard
        if command == "a1":     
            state = 1
            print ("----| State SYS : {} (Find The Chessboard)".format(state))
        #Calibration
        #elif command == "A2":   
        #    state = 2
        #    print ("----| State SYS : {} (Calibration)".format(state))
        #Show Calibration& ARUCO
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
            print ("----| State SYS : {} (Median The Field)".format(state))
        elif command == "mid":   
            state = 9
            print ("----| State SYS : {} (Median The Field)".format(state))
        # Set Counter of sampling
        elif command == "set":
            print ("        Connect Camera          : 0")
            print ("        Show setting            : 1") 
            print ("        Set Counter of sampling : 2") 
            print ("        set period              : 3") 
            print ("        set rotation            : 4") 
            command = input()
            if command == '0':
                camera.cap = cv2.VideoCapture(camera_index+cv2.CAP_DSHOW)
                if camera.cap.isOpened():
                    print("----> Connected")
                    camera.setting()
                else :
                    print("----> Disconnected")
            elif command == '1':
                print ("--> Counter of sampling : {}".format(num_sample))
                print ("--> Deley of sampling : {}".format(period))
            elif command == '2':
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
        ret,img = camera.cap.read()
        if ret :
            img=cam_calib.show_chessboard(img)
            cv2.imshow("Find The Chessboard",img)
        key = cv2.waitKey(30)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Find The Chessboard")
            print ("        Camera : {}".format(ret))
            print ("----| State SYS : {}".format(state))
    
    elif state == 4: # Show the result from Calibration & Find ARUCO markers and then show them
        ret,img = camera.cap.read()
        if ret:
            aruco_ret,field_img,aruco_img = field.cropWith_aruco(img,True)  # Find ARUCO
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
            cv2.destroyWindow("Image")
            cv2.destroyWindow("Crop Image")
            print ("----| State SYS : {}".format(state))

    elif state == 5: # Sampling an image and then Median them all ---| num_sample and period are the input
        count = 0
        command = input()
        com.posx = 0
        com.posz = 1140
        data = serial.to_bytes(
                [0x46, 0x58, high_byte(com.posx), low_byte(com.posx), 0x5A, high_byte(com.posz),
                 low_byte(com.posz), 0x53])
        print("X:", com.posx, "Z:", com.posz)
        print("Passcode: ", data)
        if com.port_connected:
            com.ser.write(data)
            print(com.ser.readline().decode())
        else:
            print("Warning: Serial Port", com.COM_PORT, "is not opened.")
        
        while(1):
            ret,img = camera.cap.read()
            #print(count)
            if ret :       # Sampling
                #if cam_calib.roi is not None:
                #img = cam_calib.calib_img(img)
                if field.Height is not None:
                    if mode_rotate :
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
                    img = field.add_imgset(img.copy())
                    if img is not False :
                        cv2.imwrite(path_folder+folder_img+command+str(count)+".jpg",img)
                    #cv2.imshow("Image",field.imgset[len(field.imgset)])

            if count== num_sample:
                break
            time.sleep(period)
            count+= 1
        if len(field.imgset) != 0 :
            camera.update_img(field.median2getBG()) 
        state = 6
    
    elif state == 6: # Show images from Sampling
        count = 0
        print ("---->  Image in set : {}".format(len(field.imgset)))
        for img in field.imgset :
            count+=1
            print ("    Image {}".format(count))
            cv2.imshow("Image",img)
            key = cv2.waitKey()
            if key == ord('q') :
                break
        state = 7
        cv2.destroyWindow("Image")

    elif state == 7: # Show images from Median
        cv2.imshow("Result",camera.image)
        key = cv2.waitKey(30)
        if key == ord('q') or camera.image is None :       
            state = 0
            cv2.destroyWindow("Result")
            print ("----| State SYS : {}".format(state))
    
    elif state == 8:
        field.imgset = []
        state = 0
        print ("----| State SYS : {}".format(state))

    elif  state == 9:
        data = serial.to_bytes(
            [0x46, 0x58, high_byte(0), low_byte(0), 0x5A, high_byte(com.posmidz),
             low_byte(com.posmidz), 0x53])
        print("X: 0", "Z:", com.posmidz)
        print("Passcode: ", data)
        if com.port_connected:
            com.ser.write(data)
            print(com.ser.readline().decode())
        else:
            print("Warning: Serial Port", com.COM_PORT, "is not opened.")
        state = 0
        print ("----| State SYS : {}".format(state))
        

        
    
        


        

        

        

