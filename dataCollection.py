import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from pathlib import Path

class HandImageProcessor:
    def __init__(self, data_folder='Data/A', max_hands=1, offset=20, size=300):
        self.capture = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=max_hands)
        self.offset = offset
        self.size = size
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

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s') and image_white is not None:
                self.save_image(image_white)

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = HandImageProcessor()
    processor.run()