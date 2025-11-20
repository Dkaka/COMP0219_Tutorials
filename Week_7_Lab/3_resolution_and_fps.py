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
picam2.start(); time.sleep(0.3)

t0, n = time.time(), 0
while True:
    rgb = picam2.capture_array()
    n += 1
    if time.time() - t0 >= 1:
        print(f"Approx FPS: {n} at {target_size}")
        n, t0 = 0, time.time()
    cv2.imshow("Video mode", rgb)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('1'):   # quick keys to explore
        target_size = (640, 480); target_fps = 60
    elif k == ord('2'):
        target_size = (1280, 720); target_fps = 30
    elif k == ord('3'):
        target_size = (1920, 1080); target_fps = 30
    elif k == ord('q'):
        break
    else:
        continue

    # Reconfigure on the fly
    picam2.stop()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": target_size},
        controls={"FrameDurationLimits": fps_to_limits(target_fps)}
    )
    picam2.configure(config)
    picam2.start()

cv2.destroyAllWindows()
picam2.stop()
