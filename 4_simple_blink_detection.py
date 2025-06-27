import cv2
import time

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Blink detection variables
no_eyes_count = 0
blink_count = 0
last_blink_time = time.time()
drowsy_start_time = None

# Thresholds
BLINK_THRESHOLD = 1  # Frames without eyes to count as blink
DROWSY_THRESHOLD = 2.5  # Seconds of no eyes for drowsiness alert

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Simple blink detection started!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.25, minNeighbors=6, minSize=(30, 30)
    )
    
    total_eyes = 0
    current_time = time.time()
    
    # For each face, detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Look for eyes in face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Use more strict settings to only detect OPEN eyes
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.25,
            minNeighbors=6,
            minSize=(15, 15),
            maxSize=(40, 40)
        )
        
        total_eyes += len(eyes)
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Blink logic
    if len(faces) > 0:  # Only if face is detected
        if total_eyes == 0:  # No eyes detected (likely closed)
            no_eyes_count += 1
            if drowsy_start_time is None:
                drowsy_start_time = current_time
        else:  # Eyes detected (open)
            if no_eyes_count >= BLINK_THRESHOLD:
                blink_count += 1
                last_blink_time = current_time
            no_eyes_count = 0
            drowsy_start_time = None
    
    # Status and alerts
    status_color = (0, 255, 0)  # Green
    status_text = "AWAKE"
    
    if len(faces) == 0:
        status_text = "NO FACE DETECTED"
        status_color = (0, 0, 255)
    elif total_eyes == 0:
        if drowsy_start_time and (current_time - drowsy_start_time) > DROWSY_THRESHOLD:
            status_text = "DROWSY ALERT!"
            status_color = (0, 0, 255)
            # Draw alert rectangle
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
            cv2.putText(frame, "WAKE UP!", (frame.shape[1]//2 - 100, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            status_text = "EYES CLOSED"
            status_color = (0, 255, 255)
    
    # Display information
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Eyes detected: {total_eyes}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if drowsy_start_time:
        drowsy_duration = current_time - drowsy_start_time
        cv2.putText(frame, f"Eyes closed for: {drowsy_duration:.1f}s", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow('Driver Attention Monitor - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Session completed! Total blinks detected: {blink_count}")