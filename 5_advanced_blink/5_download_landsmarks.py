import urllib.request
import bz2
import os

print("Downloading facial landmarks file...")
print("This is a one-time download (about 60MB)")

# URL for the facial landmarks predictor
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
filename = "shape_predictor_68_face_landmarks.dat.bz2"
extracted_filename = "shape_predictor_68_face_landmarks.dat"

# Check if already exists
if os.path.exists(extracted_filename):
    print(f"✅ {extracted_filename} already exists!")
else:
    try:
        # Download the file
        print("Downloading... (this may take a few minutes)")
        urllib.request.urlretrieve(url, filename)
        print("✅ Download complete!")
        
        # Extract the file
        print("Extracting...")
        with bz2.BZ2File(filename, 'rb') as f_in:
            with open(extracted_filename, 'wb') as f_out:
                f_out.write(f_in.read())
        print("✅ Extraction complete!")
        
        # Clean up
        os.remove(filename)
        print(f"✅ {extracted_filename} is ready to use!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("If download fails, you can manually download from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Then extract it to your project folder")

print("Done! Now you can run the advanced blink detection.")