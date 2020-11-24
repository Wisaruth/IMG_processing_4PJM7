import numpy as np
import cv2

imgs = []
path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/BG_test/"
backsub = cv2.createBackgroundSubtractorMOG2()
#for i in range(6):
#    frame = cv2.imread(path+"bgtest ("+str(i+1) +").jpg")
#    imgs.append(frame)
    
cap = cv2.VideoCapture(2) 
count = 0

while(1): 
    # read frames 
    ret, img = cap.read()
    key = cv2.waitKey(30)
    
    if key == ord('q') or ret is False :
        break 
    bg_mask = backsub.apply(img)
    bg_mask = cv2.medianBlur(bg_mask,5)
    if np.any(bg_mask):
        imgs.append(img)
        count = 1
    elif count == 1 :
        count = 0
        imgs = np.asarray(imgs)       
        median_img = np.median(imgs,axis=0).astype(np.uint8)
        #median_img = cv2.medianBlur(median_img,5)
        cv2.imshow("Remove Obj",median_img)
        imgs = []
    cv2.imshow("Mask",bg_mask)
    cv2.imshow("CAP",img)
