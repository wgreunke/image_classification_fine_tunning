#This file opens the model that was saved in cat_dog_fine_tuning.py and then validates the model using test data

print("")
print("")
print("***************************")

#Check to see if GPU is configured correctly
import tensorflow as tf
tf.config.list_physical_devices('GPU')



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

