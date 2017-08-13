# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:53:21 2017

@author: Rafael Espericueta
"""
import numpy as np
import cv2
import os

label_path = '/home/icg/FCC_VGG16s/labels/labels_rect_r640x360'
 
def median_frequency_balancing(label_path = label_path):
    """ Inputs:
           label_path - path to a folder containing labels (masks) 
        Outputs:
           The median frequency balancing weights, as described in the
           paper, SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation 
                          by Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
                  https://arxiv.org/pdf/1511.00561v2.pdf
    """
    # Fetch the names of labels (masks) to be used for the computation.
    labelnames = os.listdir(label_path)
    num_labels = len(labelnames)

    class_sums = np.zeros((4, 1), dtype=np.uint64)
    pixels_per_image = 640 * 360
    
    for labelname in labelnames:
        # Read in label to be summed.
        x = cv2.imread(os.path.join(label_path, labelname), 0)
        for i in range(4):
            class_sums[i] += len(np.where(x==i)[0])

    print 'Class Sums:'
    for i in range(4):
        print '   Class ',i,' sum = ', class_sums[i]

    class_freqs = np.zeros((4, 1), dtype=np.float32)
    denom = np.uint64(pixels_per_image) * np.uint64(num_labels) 
    for i in range(4):
        class_freqs[i] = class_sums[i] / np.float32(denom)

    print 'Class Freqs:'
    for i in range(4):
        print '   Class ',i,' freq = ', class_freqs[i]

    median_freq = np.median(class_freqs)
    class_weights = np.zeros((4, 1), dtype=np.float32)
    for i in range(4):
        class_weights[i] = median_freq / class_freqs[i]
   
    print 'Class Weights:'
    for i in range(4):
        print '   Class ',i,' weight = ', class_weights[i]
       
    return class_weights
   
   
if __name__ == '__main__':
    median_frequency_balancing()

