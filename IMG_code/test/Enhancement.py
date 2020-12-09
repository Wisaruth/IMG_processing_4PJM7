import cv2
import numpy as np
from matplotlib import pyplot as plt

def log_trans (img_,c=None):
    if c is None :
        c = 255 / np.log(1 + np.max(img_))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_[i][j] = c * (np.log(img_[i][j]+1))
    return np.array(img_, dtype = np.uint8) 

def gamma_fc (img_,gamma):
    return np.array(255*(img_ / 255) ** gamma, dtype = 'uint8')

def hsv_equalized(hsvimg_):
    H, S, V = cv2.split(hsvimg_)
    eq_V = cv2.equalizeHist(V)
    return cv2.cvtColor(cv2.merge([H, S, eq_V]),cv2.COLOR_HSV2BGR)

def bgr_equalized(img_):
    channels = cv2.split(img_)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    """for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([eq_image], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])
    plt.show()"""
    return eq_image


path = "C:/Users/ASUS/OneDrive/My work/Project_module7/IMG_test/BG_test/"
img = cv2.imread(path+"result_rota1.jpg",0)
window_name = 'Color Detection'
cv2.imshow("Original",img)
#hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#res = gamma_fc (img,0.7)
  
# show the plotting graph of an image 

#cv2.imshow("Result",res)
#cv2.waitKey()

plt.hist(img.ravel(),256,[0,256])
plt.show()

img =cv2.equalizeHist(img)
cv2.imshow("A",img)
plt.hist(img.ravel(),256,[0,256])
plt.show()