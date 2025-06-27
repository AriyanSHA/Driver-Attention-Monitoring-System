import cv2

# Initialize the default camera (0 is typically the built-in webcam)
videoCapture = cv2.VideoCapture(0)

# Check if the camera was successfully opened
if not videoCapture.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit")

# Main loop to continuously capture and display video frames
while True:
    # Read a single frame from the camera
    success, frame = videoCapture.read()
    
    # If frame wasn't read successfully, exit the loop
    if not success:
        print("Error: Can't receive frame. Exiting...")
        break
    
    # Display the current frame in a window
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all OpenCV windows
videoCapture.release()
cv2.destroyAllWindows()
print("Camera test completed!")