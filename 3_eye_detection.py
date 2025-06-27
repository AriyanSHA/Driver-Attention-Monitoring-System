import cv2

# Load classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Eye detection started!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.25,
        minNeighbors=6,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    total_eyes = 0
    
    # For each face detected
    for (x, y, w, h) in faces:
        # Draw blue rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Get the face region to look for eyes (we only look for eyes inside the face)
        roi_gray = gray[y:y+h, x:x+w]  # Face region in grayscale
        roi_color = frame[y:y+h, x:x+w]  # Face region in color
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15),
            maxSize=(40, 40)
        )
        
        total_eyes += len(eyes)
        
        # Draw green rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Display information
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Eyes: {total_eyes}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Status message
    if len(faces) > 0 and total_eyes >= 2:
        cv2.putText(frame, 'Both eyes detected - Ready for drowsiness detection!', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif len(faces) > 0 and total_eyes == 1:
        cv2.putText(frame, 'Only one eye detected', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(faces) > 0:
        cv2.putText(frame, 'Face detected but no eyes found', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Eye Detection - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Eye detection completed!")