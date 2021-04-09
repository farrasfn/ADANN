import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
import tensorflow as tf
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def feature_extractor(img):
    image_dataset = pd.DataFrame()
    
    #print(image)
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    #Reset dataframe to blank after each loop.
    
################################################################
#START ADDING DATA TO THE DATAFRAME             
        #Full image
    #GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    GLCM = greycomatrix(img, [1], [0])       
    GLCM_Energy = greycoprops(GLCM, 'energy')[0]
    df['Energy'] = GLCM_Energy
    GLCM_corr = greycoprops(GLCM, 'correlation')[0]
    df['Corr'] = GLCM_corr       
    GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
    df['Diss_sim'] = GLCM_diss       
    GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
    df['Homogen'] = GLCM_hom       
    GLCM_contr = greycoprops(GLCM, 'contrast')[0]
    df['Contrast'] = GLCM_contr


    GLCM2 = greycomatrix(img, [3], [0])       
    GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
    df['Energy2'] = GLCM_Energy2
    GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
    df['Corr2'] = GLCM_corr2       
    GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
    df['Diss_sim2'] = GLCM_diss2       
    GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
    df['Homogen2'] = GLCM_hom2       
    GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
    df['Contrast2'] = GLCM_contr2

    GLCM3 = greycomatrix(img, [5], [0])       
    GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
    df['Energy3'] = GLCM_Energy3
    GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
    df['Corr3'] = GLCM_corr3       
    GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
    df['Diss_sim3'] = GLCM_diss3       
    GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
    df['Homogen3'] = GLCM_hom3       
    GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
    df['Contrast3'] = GLCM_contr3

    GLCM4 = greycomatrix(img, [0], [np.pi/4])       
    GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
    df['Energy4'] = GLCM_Energy4
    GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
    df['Corr4'] = GLCM_corr4       
    GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
    df['Diss_sim4'] = GLCM_diss4       
    GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
    df['Homogen4'] = GLCM_hom4       
    GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
    df['Contrast4'] = GLCM_contr4
    
    GLCM5 = greycomatrix(img, [0], [np.pi/2])       
    GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
    df['Energy5'] = GLCM_Energy5
    GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
    df['Corr5'] = GLCM_corr5       
    GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
    df['Diss_sim5'] = GLCM_diss5       
    GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
    df['Homogen5'] = GLCM_hom5       
    GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
    df['Contrast5'] = GLCM_contr5
    
    #Add more filters as needed
    #entropy = shannon_entropy(img)
    #df['Entropy'] = entropy

    
    #Append features from current image to the dataset
    image_dataset = image_dataset.append(df)

    image_dataset.to_csv('Data/GLCMTest.csv', index=False)    


SIZE = 128
imeji_path = 'dementia/train/NonDemented/nonDem0.jpg'
imeji = cv2.imread(imeji_path, 0) #Reading color images
imeji = cv2.resize(imeji, (SIZE, SIZE)) #Resize images
feature_extractor(imeji)