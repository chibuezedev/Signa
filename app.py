from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="Model/TFLITE/model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_image(image):
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_image = np.expand_dims(image_resized, axis=0)
    input_image = input_image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label_index = np.argmax(output_data)
    confidence = output_data[0][predicted_label_index]

    return confidence, predicted_label_index


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_decoded = base64.b64decode(image_data)
    np_arr = np.fromstring(image_decoded, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    confidence, predicted_label_index = classify_image(image)
    labels = ["A", "B", "C"]  # Update with your labels
    result = {
        'prediction': labels[predicted_label_index],
        'confidence': str(confidence)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
