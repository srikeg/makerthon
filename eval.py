import tensorflow as tf
import numpy as np

# 1. Load your trained model
model = tf.keras.models.load_model('/home/StefanieGirod/makerthon/outputs/0527-134659/model.keras')
# model = tf.keras.models.load_model('/home/StefanieGirod/makerthon/outputs/0527-115113/model.keras')

IMAGE_DIR = "/home/StefanieGirod/makerthon/images/band_filter/acr"
# IMAGE_DIR = "/home/StefanieGirod/makerthon/images/band_filter/pc"

# 2. Create a dataset for inference
# Ensure shuffle=False so you can match predictions to filenames
ds = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    labels = None,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False 
)

# 3. Predict on the entire dataset
predictions = model.predict(ds)

# # 4. (Optional) Match predictions to filenames
file_paths = ds.file_paths
# for path, prediction in zip(file_paths[:5], predictions[:5]): # Print first 5
#     predicted_class = tf.math.argmax(prediction).numpy()
#     print(f"File: {path} | Predicted Class: {predicted_class}")

counter = 0
for path, prediction in zip(file_paths, predictions):
# for path, prediction in zip(file_paths[:10], predictions[:10]):
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Extract just the filename from the full path for cleaner printing
    filename = path.split('/')[-1] # Use '\\' instead of '/' if on Windows

    print(f"Image: {filename} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f}%")
    if predicted_class == 0:
        counter = counter + 1

print(f"correct classified: {counter}")