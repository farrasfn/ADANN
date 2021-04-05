import numpy as np
import cv2
import pydicom as dicom
import glob
from sklearn.cluster import KMeans
from PIL import Image as im
from matplotlib import pyplot as plt
from skimage import exposure
import os
  
def readDCM(path):
    ds=dicom.dcmread(path)
    dcm_sample=ds.pixel_array
    return dcm_sample

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def pngize(arraydata, name):
    data = im.fromarray(arraydata)
    data.save(name + '.png')

def preprocess(imgpath):
    csimg = cv2.imread(imgpath,0) #read image as grayscale
    ret,newg = cv2.threshold(csimg,40,255,cv2.THRESH_TOZERO)
    return (newg)

for filename in glob.glob('AD\*'):
    #print (filename)
    ph = convert(readDCM(filename), 0, 255, np.uint8)
    pngize(ph, "PNG/" + filename[2:-4])
    prep = preprocess("PNG/" + filename[2:-4] + ".png")
    pngize(prep, ("PREP/" + filename[2:-4]))
 
#    new_name = '{} {}'.format(f_name, f_ext)
#    os.rename(f, new_name)