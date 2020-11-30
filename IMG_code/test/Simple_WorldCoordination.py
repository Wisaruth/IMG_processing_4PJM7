# Group No.13
# Milestones I
#
import cv2
from numpy import sqrt


# Set up
# Image's shape : 480x640
window_name = "Camera_No.1"
ratio = 30      # 30 pixels : 1 cm
x_mosue = 0
y_mosue = 0
# Origin point in world coordinate
ord_world = [0,640] 
position =[]    # valuable for saving position's 2 points to calculate distance 
last_key = 0    # Mode
count = 0       # count click
check = 0       # check click if there isn't any click : set it to 0

def mouse_click(event, x ,y ,flags, param):
    global x_mosue,y_mosue,check
    if event == cv2.EVENT_LBUTTONDOWN:
        x_mosue,y_mosue = x,y
        check = 1

# calculate distance between 2 points in Image cooordinate      
def distance_cal (p1,p2):       # 2 parameters are list 
    return sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    
cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name,mouse_click)
print ("Start :")
print ("Origin point in World Coordinate :",ord_world)
while True:
    ret,frame = cap.read()
    if  ret is False :
        break
    cv2.imshow(window_name,frame)
    key = cv2.waitKey(1) & 0xFF # delay & get input from keyboard
    if  last_key is 0:  
        posi_mosue = []
        if key == ord('d'):     # show distance between 2 points in Image cooordinate
            print ("Distance in Image :")
            last_key = 1
        if key == ord('w'):     # Change an origin point in World cooordinate
            ord_world =[]
            print ("Set an origin point in World:")
            last_key = 2 
        if key == ord('h'):     # Check image's shape
            print ("Shape img : ",frame.shape)    
        if key == ord('p'):     # save image
            print ("Save IMG : to C:/Users/ASUS/Downloads/img.jpg")
            cv2.imwrite("C:/Users/ASUS/Downloads/img.jpg", frame)
        if len(ord_world) != 0 : # tranfer the point from image cooordinate to world cooordinate
            if check == 1 :
               print ("\nPixel coordinate : ",x_mosue,y_mosue)
               print ("In World coordinate -----")
               print ("X : {:.2f} cm".format((x_mosue-ord_world[0])/ratio))
               print ("Y : {:.2f} cm".format((ord_world[1]-y_mosue)/ratio))
               print ("Distance from ordigin point : {:.2f} cm".format(distance_cal(ord_world,[x_mosue,y_mosue])/ratio))
               check =0


    if last_key == 1:
        if check == 1 :
            position.append([x_mosue,y_mosue])
            print ("Pixel ",count,": ",x_mosue,y_mosue)
            count += 1
            check =0
        if len(position) == 2:
            count = 0
            last_key = 0
            print ("dis_set :",distance_cal(position[0],position[1]))
            position = []
    if last_key ==2:
        if check == 1 :
            ord_world.append(x_mosue)
            ord_world.append(y_mosue)
            print ("\nNew Ordigin point : {} ------".format(ord_world))
            last_key = 0
            count = 0
            check =0   
    if key == ord('q') or key == ord('Q') :
        cv2.destroyWindow(window_name)
        break