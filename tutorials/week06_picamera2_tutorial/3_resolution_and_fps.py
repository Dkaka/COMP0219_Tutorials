# 3_resolution_and_fps.py
from picamera2 import Picamera2
import cv2, time

def fps_to_limits(fps):
    period_us = int(1_000_000 / fps)
    return (period_us, period_us)

picam2 = Picamera2()
target_size = (1280, 720)
target_fps  = 30

config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": target_size},
    controls={"FrameDurationLimits": fps_to_limits(target_fps)}
)
picam2.configure(config)
# We did not ask for a sensor mode, so Picamera2 picks the 12-bit mode whose
# size and shape are closest to the output size. It ignores the frame rate.
print("Sensor mode in use:", picam2.camera_configuration()["sensor"])
picam2.start(); time.sleep(0.3)

t0, n = time.time(), 0
while True:
    frame = picam2.capture_array()   # HxWx3 uint8 in B, G, R order (OpenCV order, despite the name RGB888)
    n += 1
    if time.time() - t0 >= 1:
        print(f"Approx FPS: {n} at {target_size}")
        n, t0 = 0, time.time()
    cv2.imshow("Video mode", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('1'):   # quick keys to explore
        # 40 fps is only an example. Picamera2 picks the sensor mode for this size
        # by itself, and the script prints it ("Sensor mode in use"). On current
        # Raspberry Pi OS it is the 12-bit 1332x990 mode, which uses only the centre
        # of the sensor, so the view is narrower. On older systems it is 2028x1520.
        # A small output size alone does not give 120 fps (see 3.1).
        target_size = (640, 480); target_fps = 40
    elif k == ord('2'):
        target_size = (1280, 720); target_fps = 30
    elif k == ord('3'):
        target_size = (1920, 1080); target_fps = 30
    elif k == ord('q'):
        break
    else:
        continue

    # Stop the camera, configure the new size and frame rate, and start it again
    picam2.stop()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": target_size},
        controls={"FrameDurationLimits": fps_to_limits(target_fps)}
    )
    picam2.configure(config)
    print("Sensor mode in use:", picam2.camera_configuration()["sensor"])
    picam2.start()
    t0, n = time.time(), 0   # start counting again, so the next FPS value is for the new mode only

cv2.destroyAllWindows()
picam2.stop()
