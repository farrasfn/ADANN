import numpy as np
import cv2
import pydicom as dicom
import glob
from sklearn.cluster import KMeans
from PIL import Image as im
from matplotlib import pyplot as plt
from skimage import exposure

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
    r = 1200.0/csimg.shape[1]
    dim=(1200,int(csimg.shape[0]*r))
    rz = cv2.resize(csimg,dim,interpolation=cv2.INTER_AREA)
    g = 0.2 * (np.log(1 + np.float32(rz)))
    ret,newg = cv2.threshold(g,0.2,255,cv2.THRESH_TOZERO)
    return (newg)

sample2 = convert(readDCM('sample2.dcm'), 0, 255, np.uint8)
pngize(sample2,'sample2')
prep = preprocess('sample2.png')
plt.imshow(prep)
plt.show()
#cv2.imshow('sample image', preprocessed)
#cv2.waitKey()


#path='sample2.png';
#processedimg=preprocess(path)
#cv2.imshow('threshold = 0.2',processedimg)
#cv2.waitKey(0)