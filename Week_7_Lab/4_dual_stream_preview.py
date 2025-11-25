# 4_dual_stream_preview.py
from picamera2 import Picamera2
import cv2, time, numpy as np

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    lores={"size": (320, 240), "format": "YUV420"}
)
picam2.configure(config)
picam2.start(); time.sleep(0.3)

while True:
    # Capture both; lores arrives as YUV420 (Planar)
    yuv = picam2.capture_array("lores")
    y = yuv[:240, :]                    # luminance plane
    small = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    main  = picam2.capture_array("main")
    h_main, w_main, _ = main.shape
    h_small, w_small, _ = small.shape
    scale = h_main / h_small
    w_new_small = int(w_small * scale)
    small_resized = cv2.resize(small, (w_new_small, h_main))

    vis   = np.hstack([main, small_resized])
    cv2.imshow("Main (left) + Lores (right)", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
