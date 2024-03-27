#This file does the same as cat_dog_tuning but with updated tensorflow functions
#https://www.tensorflow.org/tutorials/images/classification

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

image_dim=180

#Load test images
train_ds=tf.keras.utils.image_dataset_from_directory(
  "test",labels="inferred", image_size=(255,255),validation_split=.2, subset="training" batch_size=30) 
dir_class_names=train_ds.class_names
print("Class names from directorys")
print(dir_class_names)


val_ds=tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=.2,
  subset="validation",
  seed=123,
  image_size=(image_dim,image_dim),
  batch_size=30
)

#Show an image to be sure you have loaded it correctly
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


#Normalize the images to 0,1 instead of 0,255
normalization_layer=layers.rescaling(1./255)
train_norm=train_ds.map(lambda x,y:(normalization_layer(x),y))
vald_norm=val_ds.map(lambda x,y:(normalization_layer(x),y))
#image_batch,labels_batch=next(iter(normalized_ds))
#first_image=image_batch[0]
#print(np.min(first_image),np.max(first_image))



#********************* Building the Model **********************
#Fetch existing model
from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(input_shape = (image_dim, image_dim, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')


#First freeze layers that will stay the same
for layer in base_model.layers:
    layer.trainable = False

#Flatten output layer
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

#New model consists of the base model and the additional layers you want to add.
model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

#Compile the model.
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])


print(model.summary)

#Fit the model
model.fit(train_ds,validation_data=val_ds, epochs=2)

model.save('models/updated.keras')