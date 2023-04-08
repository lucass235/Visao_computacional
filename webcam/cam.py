import cv2

face_detector = cv2.CascadeClassifier(
    '../Cascades/haarcascade_frontalface_default.xml')
eyes_detector = cv2.CascadeClassifier('../Cascades/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections_face = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                     minNeighbors=5)

    detections_eyes = eyes_detector.detectMultiScale(image_gray, minSize=(20, 20),
                                                     minNeighbors=5, maxSize=(50, 50))

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections_face:
        print(f'face = {w}, {h}')
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in detections_eyes:
        print(f'eyes = {w}, {h}')
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
