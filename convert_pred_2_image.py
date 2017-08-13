# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:53:21 2017

@author: Rafael Espericueta
"""
import keras
import numpy as np
import cv2
import os

model_file = '/home/icg/FCC_VGG16s/keras_model2.h5'
image_path = '/home/icg/FCC_VGG16s/images/images_rect_r640x360'
predicted_path = '/home/icg/FCC_VGG16s/pred2'

def convert_pred_2_image(x_softmax):
    ''' Input - a rank 4 tensor, which should be thought of as a 2-dim array
            of softmax values, one for each  of the 4 classes. These are the 
            tensors output by our semantic segmentation model.  
        Output - a numpy array of shape mxnx3, to be saved as an image.  '''
    m, n = x_softmax.shape[1:3]
    xp = np.empty((m, n, 3))  # to hold our segmented image
    
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
     

def save_predictions(model_file = model_file,
                     image_path = image_path,
                     predicted_path = predicted_path):
    """ Inputs:
           model_file - location of trained model
           image_path - path to a folder containing images to be segmented
           predicted_path  - folder in which to save segmented images. 
           
        NOTE: It's assumed that each image has been preprocessed to have a
              100 pixel border like that of the training images.  """
    # Read in trained segmentation model.
    model = keras.models.load_model(model_file)
    # Fetch the names of images to be segmented.
    imagenames = os.listdir(image_path)
    
    for imagename in imagenames:
       print imagename
       # Read in image to be segmented.
       x = cv2.imread(os.path.join(image_path, imagename))
       
       # Preprocess image.
       x = x.astype(np.int8)
       x -=  np.array([111, 116, 121])  # subtract average color from each pixel
       x = np.expand_dims(x, axis=0)  # to make a tensor of rank 4

       # Make prediction for each pixel.
       x_softmax = model.predict(x)

       # Convert softmax tensor to numpy array (an image).
       xp = convert_pred_2_image(x_softmax)

       # Save this image.
       cv2.imwrite(os.path.join(predicted_path, 'pred_' + imagename[-9:]), xp)


if __name__ == '__main__':
    save_predictions()
