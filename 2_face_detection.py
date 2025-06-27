import cv2

# Load the face detection classifier (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Face detection started!")
print("Press 'q' to quit")

while True:
    # Capture frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Convert to grayscale (face detection works better on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.25, 6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Show number of faces detected
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Detection - Press Q to quit', frame)
    
    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Face detection completed!")