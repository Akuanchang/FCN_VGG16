# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:56:53 2017

@author: Rafael Espericueta
"""
import os
import cv2

# Function that puts a border around an image.
def add_border(im, bordersize = 100, val = [0, 0, 0]):
    ''' 
        input:  cv2 rgb image
        output: same image with border of size borderesize with value val
    '''
    return cv2.copyMakeBorder(im,
                              top = bordersize,
                              bottom = bordersize,
                              left = bordersize,
                              right = bordersize,
                              borderType = cv2.BORDER_CONSTANT,
                              value = val)

def borderize(path, val = [111, 116, 121]):
    ''' Puts a border around every image in folder named "path".
        It's assumed the folder contains only images.  '''
    imagenames = os.listdir(path)
    num_images = len(imagenames)

    for i, imagename in enumerate(imagenames):
        if i % 400 == 0:
            print 'Progress: ', round(int((float(i)/num_images * 100.))), '%'
        fullname = os.path.join(path, imagename)
        im  = add_border( cv2.imread(fullname), val = val )
        cv2.imwrite(fullname, im)

path = '/home/icg/Martin/train_data_graz/images_rect_r640x360'

# The border is the average color over all the training images.
borderize(path, val = [111, 116, 121])

'''
im  = add_border( cv2.imread('rafa.jpg') )
cv2.imwrite('rafab.jpg', im)

im = cv2.imread('rafa.jpg')
cv2.imshow('image', im)
cv2.waitKey(0) & 0xFF   # still freezes console
'''