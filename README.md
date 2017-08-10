#Repository for semantic segmentation using VGG16s, using Keras 2 with a TensorFlow back-end, and implemented in Python 2.7.

Author - Rafael Espericueta (with significant help from Martin Hirzer of the Graz Institute for Computer Graphics and Vision)

The requirements are included in the files:  tf_requirements.txt  (CPU only), and TF_requirements.txt (using GPU)

The image folders included here only contain 10 image files and their associated masks. 

The images were preprocessed using add_border.py, which adds a 100 pixel border to each image (but not the masks). The color 
of the border is the average color (across all the images). This average color is subtracted from each image during the preprocessing done in FCN_VGG16.py, at which point the border will be all 0's.

The file model_summary.txt is the output of the Keras command  model.summary().

Note that in FCN_VGG16.py, the VGG16 model is loaded with weights trained on ImageNet, and these values are frozen for the base encoder. Comment out the freeze layers code, and the weights will be trainable.

The code in convert_pred_2_image.py shows how to read in the trained model (saved by FCN_VGG16.py), use it to make semantic segmentation predictions, and to convert the softmax output to a color image.

The saved model, keras_model.h5, is about one GB in size, and is thus too large to upload to Github.  It can be created and saved by running FCN_VGG16.py, even with only a CPU in a couple of hours (run on the 10 image/mask pairs included in the repo).


