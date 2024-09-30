import cv2

capure = cv2.VideoCapture(0)

while True:
    ret, frame = capure.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

