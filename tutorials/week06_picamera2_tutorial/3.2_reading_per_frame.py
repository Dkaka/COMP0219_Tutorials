# 3.2_reading_per_frame.py
# Get one reading from every camera frame at 120 fps: here, the angle of a yellow
# string from vertical. The script also checks that no frame is skipped.
from picamera2 import Picamera2
import cv2, csv, math, signal, statistics, time
import numpy as np

# Region of interest (ROI): x, y, width, height in pixels of the 1332x990 image.
# The script processes only this part of each frame. A smaller ROI is faster.
# The full frame takes too long at 120 fps (see the README), so choose a strip
# that just contains the string. This one is the central third of the image.
roi = (444, 0, 444, 990)

# HSV colour range of the string (the Week 8 default for a yellow string).
# Week 8 shows how to choose these numbers for your own string and light.
lower = np.array([21, 78, 152], dtype=np.uint8)
upper = np.array([41, 255, 255], dtype=np.uint8)

kernel = np.ones((3, 3), np.uint8)

# The window. Updating it takes time. Picamera2 keeps one finished frame waiting for
# the loop, so one slow pass of the loop only delays the next frame. Frames are
# skipped in two cases. First, when the work on one frame (the processing and the
# window together) takes longer than about two frame times (16.7 ms). The time in
# which capture_request() waits for the next frame does not count. Second, when this
# work takes longer than one frame time (8.3 ms) on average. This is why the full
# frame is too slow (see roi above). Drawing the window also gives the whole Pi more
# work, and then the camera itself can drop frames (see script 3.1).
# So the window is smaller and slower than in script 3.1: one third of the size,
# every 6th frame (20 fps). For a measurement in which every frame counts, set
# show_window = False. Then the script opens no window (it also works in Remote
# shell); stop it with Ctrl+C.
show_window = True
show_every = 6
display_scale = 3


def string_angle(bgr):
    """Return the angle of the string from vertical in degrees, or None if not found.
    0 = hanging straight down; positive = the bottom end is to the right."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) <= 50:
        return None
    vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vy < 0:                       # make the direction point down the image
        vx, vy = -vx, -vy
    return math.degrees(math.atan2(vx, vy))


def main():
    # In Remote shell, closing the browser tab sends the script a stop signal
    # (SIGHUP), often twice. Without the lines below, the script would end at once
    # and readings.csv would not be saved. The function below makes the first
    # SIGHUP stop the script like Ctrl+C, so that it still saves the file. The same
    # happens for SIGTERM, which the kill command sends. After the first signal, the
    # script ignores any later SIGHUP or SIGTERM, so that a second signal cannot stop
    # it before the file is saved. Ctrl+C works as before.
    def stop_like_ctrl_c(signum, frame):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGHUP, stop_like_ctrl_c)
    signal.signal(signal.SIGTERM, stop_like_ctrl_c)

    picam2 = Picamera2()
    # The same 120 fps setting as script 3.1.
    config = picam2.create_video_configuration(
        main={"size": (1332, 990), "format": "RGB888"},
        sensor={"output_size": (1332, 990), "bit_depth": 10},
        controls={"FrameDurationLimits": (8333, 8333)}
    )
    picam2.configure(config)
    print("Sensor mode in use:", picam2.camera_configuration()["sensor"])
    picam2.start()
    time.sleep(0.5)

    x, y, w, h = roi
    if not (0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= 1332 and y + h <= 990):
        raise SystemExit(f"roi = {roi} does not fit inside the 1332x990 image. Change roi at the top of the script.")
    display_size = (1332 // display_scale, 990 // display_scale)
    readings = []                     # (time in seconds, angle in degrees or None) for every frame
    first_ts = last_ts = None
    skipped_total = 0
    if show_window:
        # Open the window before the measurement starts: opening a window is slow,
        # and the camera would skip frames while it opens.
        cv2.imshow("Reading per frame", np.zeros((display_size[1], display_size[0], 3), np.uint8))
        cv2.waitKey(500)
        print("Press 'q' in the window (or Ctrl+C in the terminal) to stop.")
    else:
        print("Press Ctrl+C to stop.")

    t0, n, skipped, proc_ms, f0 = time.time(), 0, 0, [], picam2.frames
    try:
        while True:
            request = picam2.capture_request()
            frame = request.make_array("main")          # B, G, R order, as OpenCV expects
            metadata = request.get_metadata()
            request.release()

            # SensorTimestamp is the time (in nanoseconds) at which the camera took this frame.
            # If the gap to the previous frame is clearly longer than one frame time, frames
            # were skipped. Either the loop was too slow to take them (then "Readings" is
            # lower than "Frames received"), or they never reached Picamera2 (then "Frames
            # received" is also lower than 120).
            ts = metadata["SensorTimestamp"]
            frame_time = metadata["FrameDuration"] * 1000            # microseconds -> nanoseconds
            if last_ts is None:
                first_ts = ts
            else:
                missed = round((ts - last_ts) / frame_time) - 1
                skipped += missed
                skipped_total += missed
            last_ts = ts

            # The reading: process only the ROI. frame[y:y+h, x:x+w] is a view, not a copy.
            start = time.perf_counter()
            angle = string_angle(frame[y:y+h, x:x+w])
            proc_ms.append((time.perf_counter() - start) * 1000)
            readings.append(((ts - first_ts) / 1e9, angle))
            n += 1

            dt = time.time() - t0
            if dt >= 1:
                received = (picam2.frames - f0) / dt
                text = "no string found" if angle is None else f"{angle:+.1f} deg"
                print(f"Frames received: {received:.0f} fps   Readings: {n / dt:.0f} per s   "
                      f"Skipped: {skipped}   Processing: {statistics.median(proc_ms):.1f} ms   Angle: {text}")
                t0, n, skipped, proc_ms, f0 = time.time(), 0, 0, [], picam2.frames

            if show_window and len(readings) % show_every == 0:
                k = display_scale
                small = cv2.resize(frame, display_size)
                cv2.rectangle(small, (x // k, y // k), ((x + w) // k, (y + h) // k), (255, 0, 0), 2)
                label = "no string" if angle is None else f"{angle:+.1f} deg"
                cv2.putText(small, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Reading per frame", small)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:      # Ctrl+C in the terminal also stops the script
        pass

    cv2.destroyAllWindows()
    picam2.stop()

    with open("readings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "angle_deg"])
        for t, a in readings:
            writer.writerow([f"{t:.6f}", "" if a is None else f"{a:.2f}"])
    duration = readings[-1][0] if readings else 0
    print(f"Saved {len(readings)} readings over {duration:.1f} s to readings.csv. "
          f"Skipped frames: {skipped_total}.")


if __name__ == "__main__":
    main()
