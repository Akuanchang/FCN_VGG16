# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:53:21 2017

@author: Rafael Espericueta
"""

#
# Randomly selects 100 images (and their labels) from the set of 27,970 images
#

import os
import shutil
from random import randint

from_folder_images = '/home/icg/Martin/train_data_graz/images_rect_r640x360'
to_folder_images = '/home/icg/FCC_VGG16s/images/images_rect_r640x360'
from_folder_labels = '/home/icg/Martin/train_data_graz/labels_rect_r640x360'
to_folder_labels = '/home/icg/FCC_VGG16s/labels/labels_rect_r640x360'

imagenames = os.listdir(from_folder_images)
labelnames = os.listdir(from_folder_labels)
nb_imgs = len(imagenames)

indices_to_move = [randint(1, nb_imgs) for _ in range(100)]

for i in indices_to_move:
    shutil.copy2(os.path.join(from_folder_images, imagenames[i]), to_folder_images)
    shutil.copy2(os.path.join(from_folder_labels, labelnames[i]), to_folder_labels)