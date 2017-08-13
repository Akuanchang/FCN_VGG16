# Repository for semantic segmentation using VGG16s, using Keras 2 with a TensorFlow back-end, and implemented in Python 2.7.

Author - Rafael Espericueta (with help from Martin Hirzer of the Graz Institute for Computer Graphics and Vision)

The requirements are included in the files:  tf_requirements.txt  (CPU only), and TF_requirements.txt (using GPU)

The image folders included here only contain 10 image files and their associated masks. 

The full image files, containing over 29,000 images with their associated segmentationi masks (ground truth), can be downloaded from here:   https://drive.google.com/open?id=0ByrpNZ-YvZOyRlBURU9rNnh5Zlk

The images included here were preprocessed using add_border.py, which adds a 100 pixel border to each image (but not the masks). The color of the border is the average color (across all the 29K images). This average color is subtracted from each image during the preprocessing done in FCN_VGG16.py, at which point the border will be all 0's.

The file model_summary.txt is the output of the Keras command  model.summary().

Note that in FCN_VGG16.py, the VGG16 model is loaded with weights trained on ImageNet, and these values are frozen for the base encoder. Comment out the freeze layers code, and the weights will be trainable. Layer 0 to 18 are the VGG-16 base encoder layers.

The code in convert_pred_2_image.py shows how to read in the trained model (saved by FCN_VGG16.py), use it to make semantic segmentation predictions, and to convert the softmax output to a color image.

The saved model, keras_model.h5, is about one GB in size, and is thus too large to upload to Github. It's only (over) trained on 10 very similar images, so perhaps it's not much use. But you may download it from here:  https://drive.google.com/open?id=0ByrpNZ-YvZOyYzZ2UWYycGs3LVE

The code median_freq_balancing.py computes the weights as described in the file Keras_for_Segmentation.pdf, where you'll also find additional information about this site.

