import numpy as np
import cv2
from Image_processing import Image

path = "C:/Users/wisar/OneDrive/My work/Project_module7/IMG_test/"


image = Image(0,path)
image.frame = cv2.imread(path+"Map_2A.jpg")

cv2.imshow("Img",image.frame )
cv2.waitKey()
