# 1_minimal_python_preview.py
from picamera2 import Picamera2
import cv2, time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
picam2.configure(config)
picam2.start(); time.sleep(0.3)   # short warm-up for AE/AWB

print("Sensor opened. Press 'q' to quit.")
while True:
    rgb = picam2.capture_array()               # HxWx3 RGB (uint8)
    cv2.imshow("Preview", rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
