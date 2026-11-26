import argparse
import os
import queue
import sys
import threading
from pathlib import Path

import cv2

# Parameters
# A calibration is only valid for the same image size, sensor mode and lens focus as the videos that
# you measure later. So use the size and the sensor mode of your wind-tunnel recordings, and do not
# turn the focus ring after you calibrate.
# The default is the simplest setup that keeps the full field of view: photos of 2028x1520 pixels.
# For this size, Picamera2 chooses the 2028x1520 sensor mode, which reads the whole sensor.
# If your recordings use another mode, set CAPTURE_SIZE and SENSOR to the values of your recording
# script. For example, the 120 fps mode of script 3.1 of the week 6 Picamera2 tutorial uses only the
# centre of the sensor: CAPTURE_SIZE = (1332, 990), SENSOR = {"output_size": (1332, 990), "bit_depth": 10}.
# The frame rate does not matter here.
CAPTURE_SIZE = (2028, 1520)  # Size (width, height) of the saved photos
SENSOR = None  # Sensor mode of the Raspberry Pi camera; None lets Picamera2 choose it from CAPTURE_SIZE
chessboard_size = (4, 7)  # Inner corners in the chessboard; must be the same as in calibration.py
capture_key = 'c'  # Key to capture the image
exit_key = 'q'  # Key to exit the program
detect_width = 640  # The chessboard check runs on a smaller copy of this width, so it stays fast
preview_width = 1014  # Width of the preview window, so that it fits on the screen (photos keep CAPTURE_SIZE)

parser = argparse.ArgumentParser(description="Capture photos of a chessboard for calibration.py.")
parser.add_argument("--nogui", action="store_true",
                    help="terminal mode: no preview window; press Enter to save a photo, type q and Enter to quit")
args = parser.parse_args()

# The preview window needs a display. On Linux without one (for example in the Pi Connect remote shell
# or over SSH), OpenCV's window library (Qt) would stop the whole program with an unclear error. So
# without a display, the script uses terminal mode: it opens no window, it prints whether it can see
# the chessboard, and you type commands in the terminal instead of pressing keys in the window.
# With --nogui, the script uses terminal mode on any computer.
has_display = (not sys.platform.startswith("linux")
               or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
show = has_display and not args.nogui

# Save the photos next to this script, in the folder that
# "python calibration.py --images calibration_images" reads.
output_folder = Path(__file__).resolve().parent / "calibration_images"
output_folder.mkdir(exist_ok=True)

# Continue the numbering after the photos that are already in the folder, so that a new run
# never overwrites the photos of an earlier run.
numbers = [int(p.stem.split("_")[-1]) for p in output_folder.glob("calibration_image_*.jpg")
           if p.stem.split("_")[-1].isdigit()]
image_count = max(numbers, default=-1) + 1

# Open the camera.
# On the Raspberry Pi, the HQ camera works through libcamera, so we use Picamera2
# (cv2.VideoCapture cannot read this camera on a Pi 5).
# On a laptop, Picamera2 is not installed, so we use the webcam with cv2.VideoCapture instead.
# This is only for practice: it calibrates the webcam, not the Pi camera.
try:
    from picamera2 import Picamera2
except ImportError as error:
    Picamera2 = None
    # On a Pi this message usually means the environment cannot see the apt Picamera2
    # (see the rule in tutorials/README.md).
    print(f"Picamera2 is not available ({error}), so the laptop webcam is used.")

if Picamera2 is not None:
    picam2 = Picamera2()
    # "RGB888" gives arrays in B, G, R order, which is the order OpenCV expects
    if SENSOR is None:
        config = picam2.create_video_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"})
    else:
        config = picam2.create_video_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"},
                                                   sensor=SENSOR)
    picam2.configure(config)
    print("Camera: Raspberry Pi camera (Picamera2)")
    # The sensor mode that Picamera2 really chose. With the defaults above it should be
    # {'bit_depth': 12, 'output_size': (2028, 1520)}.
    print("Sensor mode in use:", picam2.camera_configuration()["sensor"])
    picam2.start()
else:
    cap = cv2.VideoCapture(0)  # Change the index to 1, 2, etc., if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
    if not cap.isOpened():
        sys.exit("Error: Unable to access the camera.")
    print("Camera: laptop webcam (cv2.VideoCapture), for practice only")


def read_frame():
    """Return the next camera frame as a BGR image, or None if there is no frame."""
    if Picamera2 is not None:
        return picam2.capture_array()
    ret, frame = cap.read()
    return frame if ret else None


def board_position(corners, image):
    """Say where the chessboard is in the image, so that you can follow the list of photos to take.
    The position names one of 3 x 3 equal parts of the image: the part with the middle of the inner corners."""
    # OpenCV 4 gives the corners as an array of shape (28, 1, 2) and OpenCV 5 as (28, 2), so the
    # array is reshaped to one row (x, y) per corner, as in calibration.py.
    xs, ys = corners.reshape(-1, 2).T
    height, width = image.shape[:2]
    row = ["top ", "", "bottom "][min(2, int(3 * ys.mean() / height))]
    column = ["left", "", "right"][min(2, int(3 * xs.mean() / width))]
    place = (row + column).strip() or "centre"
    return (f"Board position: {place}. The inner corners cover {100 * (xs.max() - xs.min()) / width:.0f}% "
            "of the image width.")


print(f"Photos are saved in: {output_folder}")
commands = queue.Queue()  # In terminal mode: the lines that you type in the terminal
if show:
    print(f"Press '{capture_key}' to capture an image and save it (only when the chessboard is found).")
    print(f"Press '{exit_key}' to quit.")
    print("If no window appears (for example over SSH), press Ctrl+C and run the script again with --nogui.")
else:
    # Terminal mode. A second thread waits for the lines that you type and puts them in a queue,
    # so that the main loop can keep reading the camera while it waits.
    def read_terminal():
        for line in sys.stdin:
            commands.put(line.strip().lower())
        commands.put(exit_key)  # The input has ended (for example with Ctrl+D), so quit

    threading.Thread(target=read_terminal, daemon=True).start()
    print("Terminal mode: " + ("no display found, so " if not has_display else "") + "there is no preview window.")
    print("The script prints 'Chessboard found' or 'No chessboard' each time this changes.")
    print(f"Press Enter to save a photo (only while the chessboard is found). Type {exit_key} and press Enter to quit.")

last_found = None  # In terminal mode: was the chessboard found in the previous frame?
try:
    while True:
        # Read a frame from the camera
        frame = read_frame()
        if frame is None:
            print("Error: Unable to read from the camera.")
            break

        # Look for the chessboard in a small grey copy of the frame. The first two flags are
        # OpenCV's defaults (calibration.py uses them too); CALIB_CB_FAST_CHECK gives up quickly
        # when there is no chessboard, so the preview stays responsive.
        scale = min(1.0, detect_width / frame.shape[1])
        small_gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            small_gray, chessboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

        if show:
            # Show a smaller copy of the frame, so that the window fits on the screen. The corners are
            # drawn on this copy, so the saved photo stays clean.
            preview_scale = min(1.0, preview_width / frame.shape[1])
            preview = cv2.resize(frame, None, fx=preview_scale, fy=preview_scale)
            if found:
                cv2.drawChessboardCorners(preview, chessboard_size, corners * (preview_scale / scale), found)
                cv2.putText(preview, "Chessboard found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(preview, "No chessboard", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Calibration Capture", preview)

            # Wait for a key press
            key = cv2.waitKey(1) & 0xFF
            capture = key == ord(capture_key)
            stop = key == ord(exit_key)
        else:
            # Terminal mode: say when the chessboard appears or disappears, and read the typed lines.
            if found != last_found:
                print("Chessboard found: press Enter to save a photo" if found else "No chessboard")
                last_found = found
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = None
            capture = command in ("", capture_key)  # Enter alone, or c and Enter
            stop = command == exit_key
            if command is not None and not capture and not stop:
                print(f"Unknown command '{command}'. Press Enter to save a photo, or type {exit_key} and press Enter to quit.")

        if capture:  # Capture and save the image
            if found:
                image_path = output_folder / f"calibration_image_{image_count:02d}.jpg"
                if cv2.imwrite(str(image_path), frame):
                    print(f"Image saved: {image_path} ({frame.shape[1]}x{frame.shape[0]})")
                    print(board_position(corners, small_gray))
                    image_count += 1
                else:
                    print(f"Error: the photo could not be saved in {output_folder}. Check that you can write "
                          "to this folder and that the disk (the SD card on the Pi) is not full.")
            else:
                print("No chessboard detected, so nothing was saved. Please adjust the position and try again.")

        elif stop:  # Exit the program
            print("Exiting...")
            break

except KeyboardInterrupt:  # Ctrl+C in the terminal also stops the script. The saved photos stay saved.
    print("Exiting...")

# Release the camera and close all windows
if Picamera2 is not None:
    picam2.stop()
    picam2.close()
else:
    cap.release()
if show:
    cv2.destroyAllWindows()
