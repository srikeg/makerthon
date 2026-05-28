import tensorflow as tf
import numpy as np
import config

# model = tf.keras.models.load_model('/home/StefanieGirod/makerthon/outputs/model.keras')
model = tf.keras.models.load_model(config.MODEL_PATH)
# model = tf.keras.models.load_model('/home/StefanieGirod/makerthon/outputs/0527-115113/model.keras')


def check_predictions(IMAGE_DIR, expected_class):

    ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_DIR,
        labels = None,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False 
    )

    predictions = model.predict(ds)
    file_paths = ds.file_paths

    counter = 0
    counter_total = 0
    for path, prediction in zip(file_paths, predictions):
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        filename = path.split('/')[-1]

        # print(f"Image: {filename} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f}%")
        counter_total = counter_total + 1
        if predicted_class == expected_class:
            counter = counter + 1

    print(f"Predcitions for {IMAGE_DIR}")
    print(f"correct classified: {counter}")
    print(f"acc: {counter/counter_total}")
    print(f"\n")

def evaluate_image(IMAGE_DIR):

    ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_DIR,
        labels = None,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False 
    )

    predictions = model.predict(ds)
    file_paths = ds.file_paths

    for path, prediction in zip(file_paths, predictions):
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        filename = path.split('/')[-1]

        print(f"Image: {filename} | Predicted Class: {config.CLASS_LABELS[predicted_class]} | Confidence: {confidence:.2f}%")
        # print(f"Predictions: {predictions}")


# ACR = "/home/StefanieGirod/makerthon/images/band_filter/acr"
# PC = "/home/StefanieGirod/makerthon/images/band_filter/pc"
# PETG = "/home/StefanieGirod/makerthon/images/band_filter/petg"
# SBE = "/home/StefanieGirod/makerthon/images/band_filter/sbe"

# check_predictions(ACR, 0)
# check_predictions(PC, 1)
# check_predictions(PETG, 2)
# check_predictions(SBE, 3)

evaluate_image(config.EVAL_IMAGE_DIR)

