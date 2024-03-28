
import tensorflow as tf
import os
import numpy as np

print("")
print("")
print("*************")


image_dim=244
#new_model = tf.keras.models.load_model('cat_dog_classifier.h5')
new_model=tf.keras.models.load_model('models/updated.keras')

#new_images_dir = 'new_images'
new_images_dir="final_test"

for filename in os.listdir(new_images_dir):
    img = tf.keras.preprocessing.image.load_img(
        os.path.join(new_images_dir, filename), target_size=(image_dim, image_dim))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = new_model.predict(img_batch)
    if prediction[0][0] < 0.5:
        print(f"{filename}: Cat")
    else:
        print(f"{filename}: Dog")
