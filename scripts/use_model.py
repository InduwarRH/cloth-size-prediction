import tensorflow as tf

# Path where the model is saved
model_save_dir = 'D:/ICBT/cloth2/model/'

# Load the trained model
model = tf.keras.models.load_model(model_save_dir)

# Define the preprocess function (should be same as used during training)
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image

# Prediction function
def predict_landmarks(image_path):
    processed_image = preprocess_image(image_path)
    processed_image = tf.expand_dims(processed_image, 0)  # Add batch dimension
    landmarks = model.predict(processed_image)
    return landmarks.flatten()

# Example usage
image_path = 'path_to_the_image_to_predict.jpg'
predicted_landmarks = predict_landmarks(image_path)
print("Predicted landmarks:", predicted_landmarks)
