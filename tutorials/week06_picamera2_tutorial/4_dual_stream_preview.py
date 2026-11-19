# 4_dual_stream_preview.py
from picamera2 import Picamera2
import cv2, time, numpy as np

picam2 = Picamera2()
# Main and lores have the same 16:9 shape, so they show the same view without distortion.
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    lores={"size": (320, 180), "format": "YUV420"}
)
picam2.configure(config)
# The camera may adjust the lores size a little, so read the size it really uses.
lw, lh = picam2.camera_configuration()["lores"]["size"]
print("Lores size:", (lw, lh), " Press 'q' to quit.")
picam2.start(); time.sleep(0.3)

while True:
    # Capture main and lores from the same frame
    (main, yuv), metadata = picam2.capture_arrays(["main", "lores"])
    y = yuv[:lh, :lw]                   # luminance (brightness) plane of the YUV420 image
    small = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

    # Show both images in one window: main on the left, lores at its real size on
    # the right. The window is 1280 + 320 = 1600 pixels wide, so it fits on a screen
    # that is 1920 pixels wide.
    h_main, w_main, _ = main.shape
    vis = np.zeros((h_main, w_main + lw, 3), dtype=np.uint8)   # black background
    vis[:, :w_main] = main
    vis[:lh, w_main:] = small
    cv2.imshow("Main (left) + Lores (right)", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
