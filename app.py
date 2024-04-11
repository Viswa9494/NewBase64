from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='trained_vgg16_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define image size
image_size = (224, 224)

def preprocess_image(img):
    # Resize the image to match the input size expected by the model
    img = img.resize(image_size)

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess input (normalize pixel values and apply required transformations)
    img_array = preprocess_input(img_array)

    return img_array

def predict_image_class(base64_image_string):
    try:
        # Decode base64 image string into image
        img_data = base64.b64decode(base64_image_string)
        img = Image.open(io.BytesIO(img_data))

        # Preprocess the image
        img_array = preprocess_image(img)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Perform inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Decode predictions
        class_names = ['Gateway of India', 'India gate pics', 'Sun Temple Konark',
                       'charminar', 'lotus_temple', 'qutub_minar', 'tajmahal']  # Replace with your actual class names
        predicted_class_index = np.argmax(output_data)
        predicted_class = class_names[predicted_class_index]
        confidence = output_data[0][predicted_class_index]

        return predicted_class, confidence
    except Exception as e:
        print("Error:", e)
        return "Error", 0.0

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    base64_image = data.get('base64_image')

    if base64_image:
        # Predict class
        predicted_class, confidence = predict_image_class(base64_image)
        # Convert confidence to Python float
        confidence = float(confidence)
        # Return prediction
        return jsonify({"predicted_class": predicted_class, "confidence": confidence})
    else:
        return jsonify({"error": "Base64 encoded image not provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)
