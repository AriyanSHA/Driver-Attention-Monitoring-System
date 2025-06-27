import cv2

# Load Haar cascade classifiers for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the camera (default device 0)
videoCapture = cv2.VideoCapture(0)

# Check if camera opened successfully
if not videoCapture.isOpened():
    print("Error: Could not open camera")
    exit()

print("Eye detection started!")
print("Press 'q' to quit")

# Main loop to capture frames and detect faces and eyes
while True:
    success, frame = videoCapture.read()
    
    if not success:
        print("Error: Can't receive frame")
        break
    
    # Convert the frame to grayscale for detection
    grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(
        grayscaleFrame,
        scaleFactor=1.25,
        minNeighbors=6,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    totalEyes = 0  # Initialize count of detected eyes
    
    # For each face detected
    for (x, y, w, h) in faces:
        # Draw blue rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Define regions of interest (ROI) for eyes detection within the face
        roiGray = grayscaleFrame[y:y + h, x:x + w]  # Grayscale face ROI
        roiColor = frame[y:y + h, x:x + w]          # Color face ROI
        
        # Detect eyes within the face ROI
        eyes = eyeCascade.detectMultiScale(
            roiGray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15),
            maxSize=(40, 40)
        )
        
        totalEyes += len(eyes)
        
        # Draw green rectangles around each detected eye
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roiColor, 'Eye', (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Display counts of faces and eyes detected on the frame
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Eyes: {totalEyes}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display status message based on eye detection count
    if len(faces) > 0 and totalEyes >= 2:
        cv2.putText(frame, 'Both eyes detected - Ready for drowsiness detection!', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif len(faces) > 0 and totalEyes == 1:
        cv2.putText(frame, 'Only one eye detected', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(faces) > 0:
        cv2.putText(frame, 'Face detected but no eyes found', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Show the frame with annotations
    cv2.imshow('Eye Detection - Press Q to quit', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
videoCapture.release()
cv2.destroyAllWindows()
print("Eye detection completed!")
