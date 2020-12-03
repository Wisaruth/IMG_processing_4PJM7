import numpy as np
import cv2


class Symbol:
    def __init__(self, name,mid,corner):
        self.name = name
        self.mid = mid
        self.corner = corner

class Tample:
    def __init__ (self,name,img):
        self.name = name
        self.img = img
    
class Detection:
    def __init__(self):
        self.targets = []
        self.tamples = []
        
