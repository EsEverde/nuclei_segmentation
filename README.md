# nuclei_segmentation
Work for kaggle competition https://www.kaggle.com/c/data-science-bowl-2018

NucleiSegmentation.py is a bare minimum TensorFlow implementation of a convolutional neural network for segmentation in the style U-net(https://arxiv.org/pdf/1505.04597.pdf) or Link-net (https://arxiv.org/pdf/1707.03718.pdf).

What this file includes:

-Input pipeline using Tensorflow Dataset API
-Convolutional neural network
-Basic loss function/gradient descent optimizer
-Basic execution loop

What this file does not include and has to be implemented:

-A proper reader for the mask files (current one assumes one mask per image, this is not the case)
-Image preprocessing in image_import and mask_import functions (add blur, rotate, etc...)
-Test datasets and iterators (for images and masks) to test accuracy.
-Tensorboard!!!








