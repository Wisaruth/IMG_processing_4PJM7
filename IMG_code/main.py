from Image_processing import Image,Aruco,Calibration
import cv2

path_folder = "C:/Users/wisar/OneDrive/My work/Project_module7/"
folder_var = "variable4IMG/"
state = 0

camera = Image(1)
cam_calib = Calibration()
aruco_mark = Aruco()


key = 0
print("Main ---> Test")
print ("State SYS : {}".format(state))
while(1):
    if state == 0 :
        command = input()
        if command == "A1":
            state = 1
            print ("----| State SYS : {} (Find The Chessboard)".format(state))
        elif command == "A2":
            state = 2
            print ("----| State SYS : {} (Calibration)".format(state))
        elif command == "show":
            state = 4
            print ("----| State SYS : {} (Calibration)".format(state))
        elif command == "exit":
            break
        else :
            print ("Not Find this command, try agin")

    if state == 1 :
        ret,img = camera.cap.read()
        if ret :
            img = cam_calib.show_chessboard(img)
            cv2.imshow("Find The Chessboard",img)
            key = cv2.waitKey(1)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Find The Chessboard")
            print ("        Camera : {}".format(ret))
            print ("----| State SYS : {}".format(state))

    elif state == 2 :
        key = cv2.waitKey()
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
                    break
            else :
                break

    elif state == 3:
        if len(cam_calib.imgs) > 3 :
            ret = cam_calib.find_matrix4Calib(path_folder+folder_var)
            if ret :
                print ("        Calibrate Done")
            else :
                print ("        Calibrate Fail")
        else :
            print ("        Can't Calibrate")
        state = 0
        print ("----| State SYS : {}".format(state))
        
    elif state == 4:
        ret,img = camera.cap.read()
        if ret:
            if cam_calib.roi is not None :
                img = calib_img(img)
            cv2.imshow("Image",img)
            key = cv2.waitKey(1)    
        if key == ord('q') or ret is False:
            state = 0
            cv2.destroyWindow("Image")
            print ("----| State SYS : {}".format(state))

        
        

