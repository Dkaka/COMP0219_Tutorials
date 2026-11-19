# 3.1_high_fps_low_res.py
from picamera2 import Picamera2
import cv2, time

def main():
    picam2 = Picamera2()

    # The HQ camera (IMX477) can reach 120 fps only in a 1332x990 sensor mode.
    # The 1332x990 modes read only the central 2664x1980 pixels of the sensor, so the
    # field of view is narrower. Current Raspberry Pi OS has three 1332x990 modes:
    # 12 bits per pixel (about 102 fps), 10 bits (120 fps) and 8 bits (about 148 fps).
    # Older systems have only the 10-bit mode. If we give only the size, Picamera2
    # picks the 12-bit mode (about 102 fps), so we also ask for 10 bits.
    sensor_mode = {"output_size": (1332, 990), "bit_depth": 10}

    # The main stream has the same size and shape as the sensor mode, so the
    # whole 1332x990 sensor image is used (a different shape would be cropped further).
    target_size = (1332, 990)
    target_fps = 120

    # We set the frame rate with FrameDurationLimits (in microseconds).
    frame_duration_us = int(1_000_000 / target_fps)
    config = picam2.create_video_configuration(
        main={"size": target_size, "format": "RGB888"},
        sensor=sensor_mode,
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)}
    )
    picam2.configure(config)

    # Check which sensor mode Picamera2 really chose.
    # It should print {'bit_depth': 10, 'output_size': (1332, 990)}.
    print("Sensor mode in use:", picam2.camera_configuration()["sensor"])

    picam2.start()
    time.sleep(0.5)

    print(f"Target: {target_fps} fps at {target_size}. Press 'q' to quit.")

    # picam2.frames counts every frame that reaches Picamera2, including the
    # frames that this loop is too slow to show.
    t0, n, f0 = time.time(), 0, picam2.frames
    while True:
        # capture_request() gives us one frame together with its own metadata.
        request = picam2.capture_request()
        frame = request.make_array("main")   # HxWx3 uint8 in B, G, R order (OpenCV order, despite the name RGB888)
        metadata = request.get_metadata()
        request.release()                    # always give the buffer back to the camera
        n += 1
        dt = time.time() - t0
        if dt >= 1:
            # FrameDuration is the frame time that the camera was set to, in microseconds.
            set_fps = 1_000_000 / metadata["FrameDuration"]
            received_fps = (picam2.frames - f0) / dt
            print(f"Camera set to: {set_fps:.1f} fps   Frames received: {received_fps:.0f} fps   "
                  f"Display loop: {n / dt:.0f} fps")
            t0, n, f0 = time.time(), 0, picam2.frames

        cv2.imshow("High FPS Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
