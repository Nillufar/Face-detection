import cv2
import time
from datetime import datetime

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize timers
study_start_time = None
not_studying_start_time = None
absent_start_time = None

# Initialize status
current_status = "Initializing"

def get_time_string():
    return datetime.now().strftime('%H:%M:%S')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    time_now = time.time()

    if len(faces) == 0:
        if current_status != "Absent":
            absent_start_time = time_now
            print(f"Became Absent at {get_time_string()}")
        current_status = "Absent"
    else:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Placeholder logic: Assuming user is studying if face is detected
        if current_status != "Studying":
            study_start_time = time_now
            print(f"Became Studying at {get_time_string()}")
        current_status = "Studying"

    # Display status and time
    cv2.putText(frame, f"Status: {current_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {get_time_string()}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    cv2.imshow("Study Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
