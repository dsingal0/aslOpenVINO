#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
from tensorflow import keras
from itertools import zip_longest


# In[8]:


target_dims = (64,64,3)
virgin_mobilnetV2 = keras.applications.mobilenet_v2.MobileNetV2()
transfer_mobilenetV2_base = keras.applications.mobilenet_v2.MobileNetV2(input_shape = target_dims, weights='imagenet', include_top=False)


# In[11]:


virgin_layers = [layer.name for layer in virgin_mobilnetV2.layers]
transfer_layers = [layer.name for layer in transfer_mobilenetV2_base.layers]
zipped_layer = zip_longest(virgin_layers, transfer_layers, fillvalue='?')
for x in zipped_layer:
    print(x)


# In[ ]:




