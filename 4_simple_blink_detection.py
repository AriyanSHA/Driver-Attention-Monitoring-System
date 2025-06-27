"""
Simple Blink and Drowsiness Detection using OpenCV.

This script captures video from the webcam and uses Haar Cascade classifiers to detect faces and eyes in real-time.
It tracks eye closure to detect blinks and issues a drowsiness alert if eyes remain closed for a specified duration.

Features:
- Detects faces and eyes using Haar Cascades.
- Counts blinks based on consecutive frames without detected eyes.
- Triggers a drowsiness alert if eyes are closed for more than a threshold duration.
- Displays real-time status, blink count, and eye detection statistics on the video feed.
- Visual feedback includes colored rectangles for faces and eyes, and alert banners for drowsiness.

Usage:
- Run the script. The webcam feed will open with overlays.
- Press 'q' to quit the application.

Dependencies:
- OpenCV (cv2)
- Python standard libraries: time

Constants:
- BLINK_THRESHOLD: Number of consecutive frames without eyes to count as a blink.
- DROWSY_THRESHOLD: Seconds of eyes closed to trigger a drowsiness alert.

Note:
- Requires a working webcam.
- Haar Cascade XML files are loaded from OpenCV's data directory.

Author: Aryan Shamsansari
"""
import cv2
import time

# Load Haar Cascade classifiers for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Blink detection state variables
noEyesCount = 0          # Counts consecutive frames with no eyes detected
blinkCount = 0           # Total number of blinks detected
lastBlinkTime = time.time()  # Timestamp of the last blink detected
drowsyStartTime = None       # Timestamp when eyes first closed (for drowsiness alert)

# Thresholds (unchanged)
BLINK_THRESHOLD = 1     # Number of frames without eyes to consider a blink
DROWSY_THRESHOLD = 2.5  # Seconds of eyes closed to trigger drowsiness alert

# Initialize webcam capture
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

    # Convert frame to grayscale for detection algorithms
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.25,
        minNeighbors=6,
        minSize=(30, 30)
    )

    totalEyes = 0
    currentTime = time.time()

    # For each detected face, detect eyes inside the face region
    for (x, y, w, h) in faces:
        # Draw a blue rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest in grayscale and color images for eye detection
        roiGray = grayFrame[y:y + h, x:x + w]
        roiColor = frame[y:y + h, x:x + w]

        # Detect eyes within the face region, using stricter parameters to focus on open eyes
        eyes = eyeCascade.detectMultiScale(
            roiGray,
            scaleFactor=1.25,
            minNeighbors=5,
            minSize=(15, 15),
            maxSize=(40, 40)
        )

        # Update the count of total eyes detected in this frame
        totalEyes += len(eyes)

        # Draw green rectangles around each detected eye
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Blink detection logic
    if len(faces) > 0:  # Only process if a face is detected
        if totalEyes == 0:
            # No eyes detected: likely eyes are closed
            noEyesCount += 1
            # If this is the first frame without eyes, mark the start time for drowsiness detection
            if drowsyStartTime is None:
                drowsyStartTime = currentTime
        else:
            # Eyes detected: if we had consecutive no-eyes frames above the threshold, count as a blink
            if noEyesCount >= BLINK_THRESHOLD:
                blinkCount += 1
                lastBlinkTime = currentTime
            # Reset counters and timers when eyes are detected again
            noEyesCount = 0
            drowsyStartTime = None

    # Set default status: awake (green)
    statusColor = (0, 255, 0)
    statusText = "AWAKE"

    # Handle different status cases
    if len(faces) == 0:
        statusText = "NO FACE DETECTED"
        statusColor = (0, 0, 255)  # Red color for alert
    elif totalEyes == 0:
        if drowsyStartTime and (currentTime - drowsyStartTime) > DROWSY_THRESHOLD:
            # Eyes have been closed for too long — trigger drowsiness alert
            statusText = "DROWSY ALERT!"
            statusColor = (0, 0, 255)
            # Draw a red alert banner at the top
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
            cv2.putText(frame, "WAKE UP!", (frame.shape[1] // 2 - 100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            # Eyes closed but not long enough for alert
            statusText = "EYES CLOSED"
            statusColor = (0, 255, 255)  # Yellowish color

    # Display status and statistics on the frame
    cv2.putText(frame, f"Status: {statusText}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2)
    cv2.putText(frame, f"Blinks: {blinkCount}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Eyes detected: {totalEyes}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # If drowsiness timer is active, show how long eyes have been closed
    if drowsyStartTime:
        drowsyDuration = currentTime - drowsyStartTime
        cv2.putText(frame, f"Eyes closed for: {drowsyDuration:.1f}s", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the camera frame with all overlays
    cv2.imshow('Driver Attention Monitor - Press Q to quit', frame)

    # Exit loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Session completed! Total blinks detected: {blinkCount}")
