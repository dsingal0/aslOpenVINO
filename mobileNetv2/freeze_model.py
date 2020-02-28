import tensorflow as tf
print(tf.__version__)
import os
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.python.tools import freeze_graph
from keras.preprocessing.image import ImageDataGenerator
import datetime
from math import ceil

#Globals
target_dims, num_classes = (64,64,3), 29

def create_model():
  target_dims = (64, 64, 3)
  num_classes = 29
  model = keras.applications.mobilenet_v2.MobileNetV2(input_shape = target_dims, weights = None, include_top=True, classes= num_classes)
  print(type(model))
  print(model.inputs)
  print(model.outputs)
  model.trainable = False
  #model.summary()
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

K.clear_session()
K.set_learning_phase(0)

model = create_model()
model.load_weights("weights.h5")

save_dir = "./model"
tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join(save_dir, "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir=save_dir)
