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

    # Drawing a large window is a lot of work for the whole Pi (the desktop draws it
    # too). While it does this, the camera system cannot keep up and drops frames:
    # with a full-size window on every frame, only about 40 of the 120 fps arrive.
    # So we show only every 4th frame (about 30 fps, enough for the eye), at half
    # size. Then the loop can get all 120 frames per second, but only while each
    # pass of the loop, including your own processing (for example finding the
    # string), takes less than about 8 ms on average.
    show_every = 4
    display_size = (target_size[0] // 2, target_size[1] // 2)

    # picam2.frames counts every frame that reaches Picamera2.
    t0, n, shown, f0 = time.time(), 0, 0, picam2.frames
    while True:
        # capture_request() gives us one frame together with its own metadata.
        request = picam2.capture_request()
        frame = request.make_array("main")   # HxWx3 uint8 in B, G, R order (OpenCV order, despite the name RGB888)
        metadata = request.get_metadata()
        request.release()                    # always give the buffer back to the camera
        n += 1
        # Put your own processing of `frame` here. It runs on every frame only if the
        # loop stays fast (less than about 8 ms per pass on average). If "Processing
        # loop" is clearly below "Frames received", the loop is too slow and misses frames.

        dt = time.time() - t0
        if dt >= 1:
            # FrameDuration is the frame time that the camera was set to, in microseconds.
            set_fps = 1_000_000 / metadata["FrameDuration"]
            received_fps = (picam2.frames - f0) / dt
            print(f"Camera set to: {set_fps:.1f} fps   Frames received: {received_fps:.0f} fps   "
                  f"Processing loop: {n / dt:.0f} fps   Window: {shown / dt:.0f} fps")
            t0, n, shown, f0 = time.time(), 0, 0, picam2.frames

        if n % show_every == 0:
            cv2.imshow("High FPS Preview", cv2.resize(frame, display_size))
            shown += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
