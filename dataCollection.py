import cv2
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)
detector = HandDetector( maxHands=1)

while True:
    ret, frame = capture.read()
    hands = detector.findHands(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

