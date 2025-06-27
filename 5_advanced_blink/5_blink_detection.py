import cv2
import dlib
import numpy as np
from imutils import face_utils
import time

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    # Calculate distances between eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance 1
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance 2
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Eye landmark indices (from the 68 facial landmarks)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold for closed eyes
EYE_AR_CONSEC_FRAMES = 2  # Number of consecutive frames for a blink
DROWSY_THRESH = 2.0  # Seconds of closed eyes for drowsy alert

# Counters
blink_counter = 0
total_blinks = 0
drowsy_start_time = None

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Advanced blink detection started!")
print("This uses precise eye measurements - much more accurate!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray, 0)
    
    current_time = time.time()
    
    # Process each face
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Extract eye coordinates
        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        
        # Calculate eye aspect ratios
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Average the eye aspect ratios
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw eye contours (much more precise!)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Draw all eye landmark points
        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        # Check if eyes are closed
        if ear < EYE_AR_THRESH:
            blink_counter += 1
            if drowsy_start_time is None:
                drowsy_start_time = current_time
            
            # Status display
            cv2.putText(frame, "EYES CLOSED", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Check for drowsiness
            if drowsy_start_time and (current_time - drowsy_start_time) > DROWSY_THRESH:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 255), -1)
                cv2.putText(frame, "DROWSY ALERT!", (frame.shape[1]//2 - 150, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                
        else:
            # If eyes were closed for enough consecutive frames, count it as a blink
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            
            blink_counter = 0
            drowsy_start_time = None
            cv2.putText(frame, "EYES OPEN", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face rectangle
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display information
    cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Threshold: {EYE_AR_THRESH}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if drowsy_start_time:
        drowsy_duration = current_time - drowsy_start_time
        cv2.putText(frame, f"Eyes closed: {drowsy_duration:.1f}s", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Show the frame
    cv2.imshow('Advanced Driver Attention Monitor - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Session completed! Total blinks detected: {total_blinks}")