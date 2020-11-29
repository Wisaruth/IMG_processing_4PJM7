from Image_processing import Image,BG_subtractor,Calibration
import cv2
import time

path_folder = "C:/Users/wisar/OneDrive/My work/Project_module7/"
folder_var = "variable4IMG/"
folder_img = "IMG_test/"

# Setup 
#------------------------------------
state = 0       # mechanic state  
num_sample = 50 # Number of image for getting background
period = 1      # seconds 

camera = Image(camera_index = 2)           
cam_calib = Calibration()
field = BG_subtractor()

print("Main ---> Test")
print ("State SYS : {}".format(state))
#------------------------------------
while(1):
    if state == 0 : # Menu
        command = input()
        if command == "A1":
            state = 1
            print ("----| State SYS : {} (Find The Chessboard)".format(state))
        elif command == "A2":
            state = 2
            print ("----| State SYS : {} (Calibration)".format(state))
        elif command == "A3":
            state = 4
            print ("----| State SYS : {} (Show Calibration & ARUCO)".format(state))
        elif command == "B1":
            state = 5
            print ("----| State SYS : {} (Sampling For Getting The Field)".format(state))
        elif command == "B2":
            state = 7
            print ("----| State SYS : {} (Sampling For Getting The Field)".format(state))
        elif command == "set sample BG":
            command = input()
            num_sample = int(command)
            print("Setting Completed")
        elif command == "set period BG":
            command = input()
            period = int(command)
            print("Setting Completed")
        elif command == "save img":
            ret = camera.saveImg("field",path_folder+folder_img)
            if ret :
                print("Saving Completed")
            else :
                print("Saving Failed")  
        elif command == "exit":
            break
        else :
            print ("Not Find this command, try agin")

    elif state == 1 : # Find The Chessboard and Show where it is on an image
        ret,img = camera.cap.read()
        if ret :
            img = cam_calib.show_chessboard(img)
            cv2.imshow("Find The Chessboard",img)
            key = cv2.waitKey(30)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Find The Chessboard")
            print ("        Camera : {}".format(ret))
            print ("----| State SYS : {}".format(state))

    elif state == 2 : # Sampling The Chessboard image for calibration
        key = cv2.waitKey()                 # press any key to continue Sampling
        print ("        Searching...")
        while(1):
            if key == ord('q') :
                state = 3
                break
            ret,img = camera.cap.read()
            if ret :
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret_chess, corners = cv2.findChessboardCorners(gray, (7,6),None)
                if ret_chess :
                    cam_calib.imgs.append(gray)
                    print("        Image : {}".format(len(cam_calib.imgs))) 
                    state = 3
                    break
            else :
                state = 0
                break

    elif state == 3:  # Find the matrix for Calibration
        if len(cam_calib.imgs) > 3 :
            ret = cam_calib.find_matrix4Calib(path_folder+folder_var)
            if ret :
                print ("        Calibrate Done")
            else :
                print ("        Calibrate Fail")
        else :
            print ("        Can't Calibrate")
        cam_calib.imgs = []
        state = 0
        print ("----| State SYS : {}".format(state))
        
    elif state == 4: # Show the result from Calibration & Find ARUCO markers and then show them
        ret,img = camera.cap.read()
        if ret:
            if cam_calib.roi is not None :
                calib_img = cam_calib.calib_img(img)                            # Calibration
                aruco_ret,field_img,aruco_img = field.cropWith_aruco(img,True)  # Find ARUCO
                cv2.imshow("Calibration",calib_img)
                if aruco_ret :
                    cv2.imshow("Crop Image",field_img)
                    cv2.imshow("ARUCO",aruco_img)
            cv2.imshow("Image",img)
            key = cv2.waitKey(30)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Image")
            cv2.destroyWindow("Calibration")
            cv2.destroyWindow("Crop Image")
            cv2.destroyWindow("ARUCO")
            print ("----| State SYS : {}".format(state))

    elif state == 5: # Sampling an image and then Median them all ---| num_sample and period are the input
        while(1):
            ret,img = camera.cap.read()
            if ret and cam_calib.roi is not None:       # Sampling
                img = cam_calib.calib_img(img)
                field.add_imgset(img)
            if len(field.imgset)== num_sample:
                break
            time.sleep(period)
        camera.update_img(field.median2getBG())         # Median
        state = 6
    
    elif state ==6: # Show images from Sampling
        count = 0
        print ("    Image in set : {}".format(len(field.imgset)))
        for img in field.imgset :
            print ("    Image {}".format(count+1))
            cv2.imshow("Image",img)
            count += 1
            key = cv2.waitKey()
            if key == ord('q') :
                break
        else :
            state = 7
            cv2.destroyWindow("Image")

    elif state ==7: # Show images from Median
        key = cv2.waitKey(30)
        if key == ord('q') or camera.image is None :
            state = 0
            cv2.destroyWindow("Image")
            print ("----| State SYS : {}".format(state))
        else :
            cv2.imshow("Result",camera.image)

        

        

        

