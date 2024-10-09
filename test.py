import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from pathlib import Path
import tensorflow as tf


class TFLiteClassifier:
    def __init__(self, model_path):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_prediction(self, image):
        # Preprocess image if necessary
        input_shape = self.input_details[0]['shape']
        input_image = cv2.resize(image, (input_shape[1], input_shape[2]))
        input_image = np.expand_dims(input_image, axis=0)  
        input_image = input_image.astype(np.float32)  

        # Set input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()

        # Get prediction results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_label_index = np.argmax(output_data)
        confidence = output_data[0][predicted_label_index]

        return confidence, predicted_label_index


class HandImageProcessor:
    def __init__(self, classifier, data_folder='Data/A', max_hands=1, offset=20, size=300):
        self.capture = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=max_hands)
        self.offset = offset
        self.size = size
        self.classifier = classifier
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def process_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None, None, None

        hands, frame = self.detector.findHands(frame)
        if not hands:
            return frame, None, None

        hand = hands[0]
        x, y, w, h = hand['bbox']
        image_crop = frame[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
        image_white = self.resize_and_pad(image_crop)

        return frame, image_crop, image_white

    def resize_and_pad(self, image):
        h, w = image.shape[:2]
        aspect_ratio = h / w
        image_white = np.ones((self.size, self.size, 3), np.uint8) * 255

        if aspect_ratio > 1:
            k = self.size / h
            width_cal = math.ceil(k * w)
            image_resize = cv2.resize(image, (width_cal, self.size))
            width_gap = math.ceil((self.size - width_cal) / 2)
            image_white[:, width_gap:width_cal + width_gap] = image_resize
        else:
            k = self.size / w
            height_cal = math.ceil(k * h)
            image_resize = cv2.resize(image, (self.size, height_cal))
            height_gap = math.ceil((self.size - height_cal) / 2)
            image_white[height_gap:height_cal + height_gap, :] = image_resize

        return image_white

    def save_image(self, image):
        self.counter += 1
        file_name = self.data_folder / f"{time.time()}.jpg"
        cv2.imwrite(str(file_name), image)
        print(f"Saved image {self.counter}")

    def run(self):
        while True:
            frame, image_crop, image_white = self.process_frame()
            if frame is None:
                break

            cv2.imshow('frame', frame)
            if image_crop is not None:
                cv2.imshow("Image Crop", image_crop)
            if image_white is not None:
                cv2.imshow("Image White", image_white)

                confidence, predicted_label_index = self.classifier.get_prediction(image_white)
                print(f"Prediction: {labels[predicted_label_index]}, Confidence: {confidence}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    labels = ["A", "B", "C"]  
    classifier = TFLiteClassifier('Model/TFLITE/model_unquant.tflite')
    processor = HandImageProcessor(classifier)
    processor.run()
