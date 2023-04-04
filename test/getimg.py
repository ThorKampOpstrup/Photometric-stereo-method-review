from pypylon import pylon   # for camera access
import cv2   # for image processing

# Connect to the first available camera
camera = pylon.TlFactory.GetInstance().CreateFirstDevice()

# Set camera properties (optional)
camera.Open()
camera.ExposureTime.SetValue(10000)  # set exposure time to 10ms
camera.Gain.SetValue(0)  # set gain to 0 dB

# Create a grab result holder
grabResult = pylon.CGrabResultPtr()

# Start the camera and grab a single frame
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

# Convert the image to a numpy array
image = grabResult.Array

# Process the image as needed (e.g. display it using OpenCV)
cv2.imshow('Image', image)
cv2.waitKey(0)

# Clean up
grabResult.Release()
camera.StopGrabbing()
camera.Close()
