import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling

# Load the pre-trained model
model = load_model('models/updated.keras')
print(model.summary)

# Load a single image using image_dataset_from_directory
# Assuming the image is in a directory named 'single_image_dir' and has a size of (224, 224)
dataset = image_dataset_from_directory(
    'images',
    image_size=(224, 224),
    batch_size=1,
    shuffle=False
)

# Normalize the image to [0, 1]
normalization_layer = Rescaling(1./255)
normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

# Predict the class of the image
for images, _ in normalized_dataset.take(1):  # Take 1 batch from the dataset
    predictions = model.predict(images)
    predicted_class = tf.argmax(predictions, axis=1).numpy()
    print(f'Predicted class: {predicted_class}')
