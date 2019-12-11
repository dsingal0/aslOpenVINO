import tensorflow as tf
import numpy as np
import os
from skimage.transform import resize
import numpy as np
import cv2
from numpy import argmax
from tensorflow.keras.models import load_model

model = load_model('aslSlimCNN.h5')
print(model.summary())
print("break")
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29
test_dir = "./asl_alphabet_test/asl_alphabet_test/"
img_file = cv2.imread(test_dir + "Z_test.jpg")
img_arr=np.empty((imageSize,imageSize,3),dtype=np.float32)
if img_file is not None:
    img_file = resize(img_file, (1, imageSize, imageSize, 3))
    img_arr = np.asarray(img_file)
model.predict(img_arr)
