import tensorflow as tf
print(tf.__version__)
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import datetime
from math import ceil

#  Using the data Augmentation in traning data
ptrain = "./asl_alphabet_train"
pval = "./asl_alphabet_test"
batch = 32
size = 64
target_dims = (64, 64, 3)
num_classes = 29
datagen1 = ImageDataGenerator(rescale=1. / 255)
"""datagen1 = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)"""

datagen2 = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen1.flow_from_directory(ptrain,
                                               target_size=(size, size),
                                               batch_size=batch,
                                               class_mode='categorical')

validation_generator = datagen2.flow_from_directory(pval,
                                                    target_size=(size, size),
                                                    batch_size=batch,
                                                    class_mode='categorical')

count1 = 0
for root, dirs, files in os.walk(ptrain):
    for each in files:
        count1 += 1

count2 = 0
for root, dirs, files in os.walk(pval):
    for each in files:
        count2 += 1

train_steps_per_epoch = ceil(count1 / batch)
val_steps_per_epoch = ceil(count2 / batch)
model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=target_dims,
                                                    weights=None,
                                                    include_top=True,
                                                    classes=num_classes)

epochs = 30
opt = keras.optimizers.Adam()
earlystop = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                          patience=30,
                                          verbose=0,
                                          mode='auto')
cp_callback = keras.callbacks.ModelCheckpoint(filepath='asl_classifier.ckpt',
                                              save_weights_only=True,
                                              verbose=1)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[earlystop, tensorboard_callback])
model.save('asl_classifier_MNV2.h5')
