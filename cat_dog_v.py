import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model

print("")
print("")
print("*************")


#Get the model and freeze the layers
image_dim = 224  # Image size for VGG16

base_model = VGG16(input_shape=(image_dim, image_dim, 3), include_top=False, weights='imagenet')

for layer in base_model.layers[:-4]:  # Freeze all but the last 4 layers
    layer.trainable = False


#Add new layers
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # Output: 0 for cat, 1 for dog

model = Model(inputs=base_model.input, outputs=predictions)


#Compile and train the model

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'cats_and_dogs_filtered/train',
    target_size=(image_dim, image_dim),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    'cats_and_dogs_filtered/validation',
    target_size=(image_dim, image_dim),
    batch_size=32,
    class_mode='binary')

model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=validation_generator)

model.save('cat_dog_classifier.h5')