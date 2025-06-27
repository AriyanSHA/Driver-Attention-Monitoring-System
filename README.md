# Driver-Drowsiness-and-Attention-Monitoring-System
Real-time computer vision project for detecting driver fatigue, blinks, and drowsiness using OpenCV.  This project aims to support driver safety by identifying closed eyes, blink frequency, and drowsiness patterns via a webcam or in-vehicle camera system, crucial for reducing the risk of road accidents.

## Versions
### Basic Version – Blink Detection via Haar Cascades

* Uses OpenCV Haar cascades for face and eye detection

* Counts blinks and raises an alert if eyes stay closed too long

### Advanced Version (Optional Extension)

* Uses dlib and 68-point face landmarks

* Computes Eye Aspect Ratio (EAR) for precise eye state detection

* Detects drowsiness based on sustained EAR drop

* Optional deployment on Jetson Nano or Raspberry Pi

## Features
* Real-time video capture from webcam

* Face and eye detection using Haar cascades

* Blink detection logic (based on eye visibility)

* Drowsiness alert after configurable eye-closed duration

* Visual feedback overlay (status, blinks, alerts)

* Portable Python script, no external training required

## Relevance to Automotive Industry
→ This project simulates a core feature in modern ADAS (Advanced Driver Assistance Systems) found in vehicles by BMW, Audi, and other OEMs:

* Drowsy Driver Detection for long drives

* Driver Monitoring System (DMS) integration

* Fatigue monitoring for autonomous driving

* Demonstrates my ability to apply computer vision to real-world automotive safety problems

## Dependencies / Requirements 
→ Install dependencies using pip: ``` pip install opencv-python dlib imutils numpy ```

* Python 3.x

* opencv-python

* dlib

* imutils

* numpy

* shape_predictor_68_face_landmarks.dat (download required separately from Dlib’s model zoo)

## How to run
1. Download the facial landmark model file shape_predictor_68_face_landmarks.dat by running: ``` 5_download_landsmarks.py ```
2. Run ``` 5_blink_detection.py ```, make sure you are in the same directory.


## Configurable Parameters (in the script):

* ``` EYE_AR_THRESH: EAR threshold to detect closed eyes (default: 0.25) ```

* ``` EYE_AR_CONSEC_FRAMES: Number of frames to confirm a blink (default: 2) ```

* ``` DROWSY_THRESH: Time in seconds to trigger drowsiness alert (default: 2.0) ```

## Notes

* This method is sensitive to lighting and camera quality.

* Works best when the face is clearly visible and frontal.

* Glasses or occlusion may reduce accuracy.

### Author: Aryan Shamsansari
### GitHub: https://github.com/AriyanSHA

## License
This project is licensed under the MIT License.
 
