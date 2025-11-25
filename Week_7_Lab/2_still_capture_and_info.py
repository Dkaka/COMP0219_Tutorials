# 2_still_capture_and_info.py
from picamera2 import Picamera2
import cv2, time

picam2 = Picamera2()
print("Camera properties:", picam2.camera_properties)   # model, bits, etc.
# e.g. choose still (full-res) or specify a size explicitly:
config = picam2.create_still_configuration(main={"format": "RGB888", "size": (4056, 3040)})
picam2.configure(config)
picam2.start(); time.sleep(0.5)

frame = picam2.capture_array()
cv2.imwrite("still_fullres.jpg", frame)
print("Saved still_fullres.jpg with shape", frame.shape)

picam2.stop()
