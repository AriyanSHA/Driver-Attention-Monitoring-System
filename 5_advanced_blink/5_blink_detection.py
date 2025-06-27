"""
Advanced Blink Detection and Drowsiness Alert System
This script uses OpenCV, dlib, and imutils to perform real-time blink detection and drowsiness monitoring using a webcam.
It detects faces, extracts eye landmarks, calculates the Eye Aspect Ratio (EAR), and determines if the eyes are closed.
If the eyes remain closed for a specified duration, a drowsiness alert is triggered.
Main Features:
- Real-time face and eye detection using dlib's 68-point facial landmark predictor.
- Calculation of Eye Aspect Ratio (EAR) for precise blink detection.
- Counts total blinks and displays them on the video feed.
- Detects prolonged eye closure and displays a drowsiness alert.
- Visualizes eye contours, landmarks, and face bounding boxes on the video stream.
- Displays EAR, blink count, threshold, number of faces detected, and eye closure duration.
Constants:
- EYE_AR_THRESH: EAR threshold below which eyes are considered closed.
- EYE_AR_CONSEC_FRAMES: Number of consecutive frames with closed eyes to count as a blink.
- DROWSY_THRESH: Time in seconds for which eyes must remain closed to trigger a drowsiness alert.
Usage:
- Ensure 'shape_predictor_68_face_landmarks.dat' is present in the working directory, if not simply run 5_download_landmarks.py.
- Run the script. The webcam feed will display detection results.
- Press 'q' to quit the application.
Dependencies:
- OpenCV (cv2)
- dlib
- imutils
- numpy
Author: Aryan Shamsansari
"""
import cv2
import dlib
import numpy as np
from imutils import face_utils
import time

# Calculate Eye Aspect Ratio (EAR) to determine eye openness
def eyeAspectRatio(eye):
    # Calculate the vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Calculate the horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    
    # EAR formula: smaller value means eyes are likely closed
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Index ranges for the left and right eyes in the 68-point landmarks
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants (unchanged)
EYE_AR_THRESH = 0.25              # Threshold for detecting closed eyes
EYE_AR_CONSEC_FRAMES = 2          # Frames required to confirm a blink
DROWSY_THRESH = 2.0               # Seconds eyes must stay closed to trigger drowsy alert

# Runtime counters and flags
blinkCounter = 0                  # Counter for consecutive frames with closed eyes
totalBlinks = 0                   # Total blinks detected
drowsyStartTime = None            # Timestamp when eye closure starts

# Start capturing from the webcam
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
    
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray, 0)
    
    # Initialize timer
    currentTime = time.time()

    ear = None

    for face in faces:
        # Get 68 facial landmark coordinates
        shape = predictor(gray, face)
        landmarks = face_utils.shape_to_np(shape)
        
        # Extract left and right eye landmark points
        leftEye = landmarks[leftEyeStart:leftEyeEnd]
        rightEye = landmarks[rightEyeStart:rightEyeEnd]
        
        # Calculate EAR for both eyes
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)
        
        # Use average EAR of both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw eye contours for visual feedback
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Draw landmarks as small yellow dots
        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        # Check if eye aspect ratio indicates closed eyes
        if ear < EYE_AR_THRESH:
            blinkCounter += 1
            
            # Start drowsy timer if not already started
            if drowsyStartTime is None:
                drowsyStartTime = currentTime
            
            cv2.putText(frame, "EYES CLOSED", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Trigger drowsiness alert if eyes closed for too long
            if drowsyStartTime and (currentTime - drowsyStartTime) > DROWSY_THRESH:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 255), -1)
                cv2.putText(frame, "DROWSY ALERT!", (frame.shape[1] // 2 - 150, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            # If blink was valid (closed long enough), count it
            if blinkCounter >= EYE_AR_CONSEC_FRAMES:
                totalBlinks += 1
            
            # Reset counters and drowsiness tracking
            blinkCounter = 0
            drowsyStartTime = None
            cv2.putText(frame, "EYES OPEN", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw rectangle around detected face
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display real-time statistics
    cv2.putText(frame, f"Blinks: {totalBlinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "EAR: N/A", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Threshold: {EYE_AR_THRESH}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show duration of closed eyes if active
    if drowsyStartTime:
        drowsyDuration = currentTime - drowsyStartTime
        cv2.putText(frame, f"Eyes closed: {drowsyDuration:.1f}s", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display the result frame
    cv2.imshow('Advanced Driver Attention Monitor - Press Q to quit', frame)
    
    # Exit when user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print(f"Session completed! Total blinks detected: {totalBlinks}")
