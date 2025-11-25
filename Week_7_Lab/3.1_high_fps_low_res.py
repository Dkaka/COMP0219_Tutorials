# 3.1_high_fps_low_res.py
from picamera2 import Picamera2
import cv2, time

def main():
    picam2 = Picamera2()
    
    # To achieve high frame rates, you often need to use a smaller resolution.
    # The Raspberry Pi camera has specific sensor modes that can achieve high FPS.
    # For example, the HQ camera can do 120fps at 1332x990.
    # We will request 120fps at 1280x720.
    target_size = (1280, 720)
    target_fps = 120
    
    # We use a video configuration and set the frame rate using FrameDurationLimits.
    frame_duration_us = int(1_000_000 / target_fps)
    config = picam2.create_video_configuration(
        main={"size": target_size, "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    
    print(f"Running at {target_fps} FPS with resolution {target_size}. Press 'q' to quit.")
    
    t0, n = time.time(), 0
    while True:
        rgb = picam2.capture_array()
        n += 1
        if time.time() - t0 >= 1:
            print(f"Approx FPS: {n}")
            n, t0 = 0, time.time()
        
        cv2.imshow("High FPS Preview", rgb)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
