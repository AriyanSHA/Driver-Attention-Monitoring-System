import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit")

# Main loop to display video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    
    # Display the frame
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Camera test completed!")