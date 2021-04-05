# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:02:24 2021

@author: fnabi
"""

import numpy as np
import cv2
import pydicom as dicom

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

ds=dicom.dcmread('sample2.dcm')
dcm_sample=ds.pixel_array
imgu8 = convert(dcm_sample, 0, 255, np.uint8)
cv2.imshow('sample image dicom',imgu8)



cv2.waitKey()
cv2.destroyAllWindows()