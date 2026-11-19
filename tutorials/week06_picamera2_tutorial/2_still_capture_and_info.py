# 2_still_capture_and_info.py
from picamera2 import Picamera2
import cv2, time

picam2 = Picamera2()
print("Camera properties:", picam2.camera_properties)   # model, pixel array size, colour filter, etc. (no bit depth)

# Each sensor mode has a size, a bit depth and a maximum frame rate.
# Read sensor_modes before configure(). To read it, Picamera2 configures the camera
# once in every mode. This takes a few seconds. If you read sensor_modes after
# configure(), it would undo your own configuration.
print("Sensor modes:")
for mode in picam2.sensor_modes:
    # crop_limits = (x, y, width, height): the part of the sensor that this mode uses
    print(f"  {mode['size']}  {mode['bit_depth']} bits  up to {mode['fps']:.0f} fps  sensor area {mode['crop_limits']}")

# e.g. choose still (full-res) or specify a size explicitly:
config = picam2.create_still_configuration(main={"format": "RGB888", "size": (4056, 3040)})
picam2.configure(config)
picam2.start(); time.sleep(0.5)

frame = picam2.capture_array()
cv2.imwrite("still_fullres.jpg", frame)
print("Saved still_fullres.jpg with shape", frame.shape)

picam2.stop()
