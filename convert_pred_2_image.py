# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:53:21 2017

@author: Rafael Espericueta
"""
import keras
import numpy as np
import cv2

def convert_pred_2_image(x_softmax):
    # Convert softmax to image
    m, n = x_softmax.shape[1:3]
    xp = np.empty((m, n, 3))
    
    for i in range(m):
        for j in range(n):
            col = np.argmax(x_softmax[0, i, j])
            if col == 0:
                xp[i,j] = [0, 0, 0]     # black
            elif col ==  1:
                xp[i,j] = [0, 0, 255]   # red
            elif col ==  2:
                xp[i,j] = [0, 255, 0]   # green
            elif col ==  3:
                xp[i,j] = [255, 0, 0]   # blue
            else:
                xp[i,j] = [255, 255, 255]  # should never happen
    return  xp
     

# Read in trained segmentation model.
model = keras.models.load_model('/home/icg/FCC_VGG16s/keras_model.h5')
       
# Read in image to be segmented.           
x = cv2.imread('/home/icg/FCC_VGG16s/images/images_rect_r640x360/IMG_7540_15062016_frame00002.png')

# Preprocess image.
x = x.astype(np.int8)
x -=  np.array([111, 116, 121])  # subtract average color from each pixel
x = np.expand_dims(x, axis=0)  # to make a tensor of rank 4

# Make prediction.
x_softmax = model.predict(x)

# Convert softmax tensor to image.
xp = convert_pred_2_image(x_softmax)

# Save image
cv2.imwrite('/home/icg/FCC_VGG16s/pred_frame00002.png', xp)

