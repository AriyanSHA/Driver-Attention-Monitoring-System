import cv2

# Load the Haar cascade classifier for frontal face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera (0 is usually the default webcam)
videoCapture = cv2.VideoCapture(0)

# Check if the camera was successfully opened
if not videoCapture.isOpened():
    print("Error: Could not open camera")
    exit()

print("Face detection started!")
print("Press 'q' to quit")

# Main loop for continuous video capture and face detection
while True:
    # Capture a single frame
    success, frame = videoCapture.read()
    
    # If the frame couldn't be read, exit the loop
    if not success:
        print("Error: Can't receive frame")
        break
    
    # Convert the frame to grayscale for better face detection
    grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(
        grayscaleFrame,
        scaleFactor=1.25,
        minNeighbors=6,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw rectangles around each detected face and label them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Show total number of detected faces on the screen
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the annotated frame
    cv2.imshow('Face Detection - Press Q to quit', frame)
    
    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy all OpenCV windows
videoCapture.release()
cv2.destroyAllWindows()
print("Face detection completed!")
