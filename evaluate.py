import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = tf.keras.models.load_model("models/mobilenet.h5")

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "data/test",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

print("\nClassification Report:")
print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices.keys()))