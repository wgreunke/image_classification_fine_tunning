#This file does the same as cat_dog_tuning but with updated tensorflow functions
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 

image_dir="images"

base_dir = 'images\cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir,"validation")

#Load test images
train_ds=tf.keras.utils.image_dataset_from_directory("test",labels="inferred", image_size=(255,255),validation_split=.2) 
dir_class_names=train_ds.class_names
print("Class names from directorys")
print(dir_class_names)

#Show an image to be sure you have loaded it correctly
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#Normalize the images to 0,1 instead of 0,255
normalized_ds=train_ds.map(lambda x,y:(normalization_layer(x),y))
image_batch,labels_batch=next(iter(normalized_ds))
first_image=image_batch[0]
print(np.min(first_image),np.max(first_image))


#Now create the model
#First freeze layers that will stay the same


