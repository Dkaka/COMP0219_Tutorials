import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Usage (you can run it from any folder):
#   python calibration.py                              # uses the sample photos
#   python calibration.py --images calibration_images  # uses the photos from calibration_cap.py
#   python calibration.py --video my_video.mp4         # uses frames from a video of the chessboard
# Add --nogui to save the pictures in the output folder instead of showing them in windows.

HERE = Path(__file__).resolve().parent  # the folder that contains this script

# Step 1: Prepare object points
# Define the chessboard size (number of internal corners)
chessboard_size = (4, 7)  # Must be the same as in calibration_cap.py
square_size = 34.0  # Size of a square in your defined unit (e.g., 1.0 for meters or mm)

# With few photos, the third radial distortion coefficient k3 cannot be measured reliably, so we keep it
# at 0. A synthetic test with the HQ camera, the 6 mm lens and the 17-photo shot list showed that fixing
# k3 did not change fx and slightly reduced the error near the image corners.
fix_k3 = True

# Prepare the object points for a standard chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# cv2.cornerSubPix moves each corner that findChessboardCorners found to its exact position, with
# sub-pixel accuracy. It searches a window of 11 x 11 pixels around each corner: (5, 5) means 5 pixels
# on each side. The window must stay smaller than about half a square in the image.
# It stops after 30 steps, or earlier when a step moves the corner by less than 0.001 pixels.
subpix_window = (5, 5)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(gray):
    """Return the inner corners of the chessboard in a grey image, or None if the board is not found."""
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, subpix_window, (-1, -1), subpix_criteria)


# Step 2: Choose the photos: a folder of photos (--images), or frames from a video (--video)
parser = argparse.ArgumentParser(description="Calibrate a camera from photos or a video of a chessboard.")
source = parser.add_mutually_exclusive_group()
source.add_argument("--images", default=str(HERE / "sample_calibration_images"),
                    help="folder with the chessboard photos (.jpg, .jpeg or .png) "
                         "(default: sample_calibration_images next to this script)")
source.add_argument("--video", help="a video of the chessboard; the script takes the photos from its frames")
parser.add_argument("--step", type=int, default=10,
                    help="with --video: look for the chessboard in every Nth frame (default: 10)")
parser.add_argument("--max-frames", type=int, default=30,
                    help="with --video: use at most this many frames, spread over the video (default: 30)")
parser.add_argument("--nogui", action="store_true",
                    help="do not open windows; only save the pictures in the output folder")
args = parser.parse_args()
if args.step < 1 or args.max_frames < 1:
    parser.error("--step and --max-frames must be 1 or more")

# Windows need a display. On Linux without one (for example in the Pi Connect remote shell or over SSH),
# OpenCV's window library (Qt) would stop the whole program. So we decide here, before the first window,
# and without a display we only save the pictures in the output folder and print where they are.
has_display = (not sys.platform.startswith("linux")
               or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
show = has_display and not args.nogui
if not show:
    print("No windows: the pictures are saved in the output folder." if has_display
          else "No display found, so no windows: the pictures are saved in the output folder.")


def save_video_frames(video_path, frames_folder):
    """Save, as JPG files, at most --max-frames frames of the video in which the chessboard is found."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit("Error: OpenCV cannot open this video. Check the --video option.")

    # Pass 1: look for the chessboard in every Nth frame, and note the numbers of the frames where it is found
    found = []
    tested = 0
    number = 0
    while cap.grab():  # grab() reads the next frame; retrieve() below converts it only when we need it
        if number % args.step == 0:
            ok, frame = cap.retrieve()
            tested += 1
            if ok and find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) is not None:
                found.append(number)
            if tested % 50 == 0:
                print(f"  frame {number}: chessboard found in {len(found)} of {tested} tested frames")
        number += 1
    cap.release()
    print(f"The video has {number} frames. The chessboard was found in {len(found)} of the {tested} "
          f"frames tested (one frame in every {args.step}).")
    if not found:
        sys.exit(f"Error: the chessboard was not found in the video. Check that chessboard_size "
                 f"{chessboard_size} is the number of inner corners of your board.")

    # Keep at most max_frames of these frames, spread evenly over the video
    keep = found[::math.ceil(len(found) / args.max_frames)]

    # Pass 2: read the video again and save the kept frames. The frames of an earlier run are deleted first.
    frames_folder.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_folder.glob("frame_*.jpg"):
        old_frame.unlink()
    cap = cv2.VideoCapture(str(video_path))
    saved = 0
    number = 0
    while number <= keep[-1] and cap.grab():
        if number in keep:
            ok, frame = cap.retrieve()
            if ok and cv2.imwrite(str(frames_folder / f"frame_{number:06d}.jpg"), frame):
                saved += 1
        number += 1
    cap.release()
    print(f"Saved {saved} frames in: {frames_folder}")
    print("The calibration below uses all of them. Look at them, delete the blurred ones, and then "
          "calibrate again from the remaining frames with:")
    print(f"  python calibration.py --images \"{frames_folder}\"\n")


if args.video:
    video_path = Path(args.video).resolve()
    print(f"Video: {video_path}")
    if not video_path.is_file():
        sys.exit("Error: this video file does not exist. Check the --video option.")
    # The results go to calibration_output/<video name>/, and the frames to its subfolder frames/
    output_folder = HERE / "calibration_output" / video_path.stem
    image_folder = output_folder / "frames"
    save_video_frames(video_path, image_folder)
else:
    image_folder = Path(args.images).resolve()
    # The results go to calibration_output/<name of the image folder>/, never into the image folder, so that
    # a later run cannot read them back as calibration photos. For the frames that --video saved, this is
    # the folder of the video, calibration_output/<video name>/.
    output_name = image_folder.parent.name if image_folder.name == "frames" else image_folder.name
    output_folder = HERE / "calibration_output" / output_name

print(f"Image folder: {image_folder}")
if not image_folder.is_dir():
    sys.exit("Error: this folder does not exist. Check the --images option.")

# Read only the original photos. Skip any picture that this script (or an older version of it) saved.
images = sorted(p for p in image_folder.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                and "_corners_detected" not in p.name and "_undistorted" not in p.name)
print(f"Found {len(images)} photos (.jpg, .jpeg or .png).")
if not images:
    sys.exit("Error: there are no .jpg, .jpeg or .png photos in this folder. Check the --images option.")

output_folder.mkdir(parents=True, exist_ok=True)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in the real world
imgpoints = []  # 2D points in the image plane
used_images = []  # the photos in which the chessboard was found
image_shape = None  # Store image shape for calibration

for fname in images:
    img = cv2.imread(str(fname))
    if img is None:
        print(f"  {fname.name}: cannot be read, skipped")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A calibration is only valid for one image size, so all photos must have the same size.
    if image_shape is None:
        image_shape = gray.shape[::-1]  # (width, height)
    elif gray.shape[::-1] != image_shape:
        sys.exit(f"Error: {fname.name} is {gray.shape[1]}x{gray.shape[0]} pixels, but the first "
                 f"photo is {image_shape[0]}x{image_shape[1]}. All photos must have the same size.")

    # Step 3: Find the chessboard corners, and refine them to sub-pixel accuracy
    corners = find_corners(gray)
    print(f"  {fname.name}: chessboard {'found' if corners is not None else 'NOT found'}")

    if corners is not None:
        objpoints.append(objp)  # Add object points
        imgpoints.append(corners)  # Add image points
        used_images.append(fname)

        # Draw the corners, save a copy in the output folder, and show it
        cv2.drawChessboardCorners(img, chessboard_size, corners, True)
        cv2.imwrite(str(output_folder / f"{fname.stem}_corners_detected.jpg"), img)
        if show:
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

if show:
    cv2.destroyAllWindows()
elif used_images:
    print(f"The photos with the detected corners are saved as: {output_folder / '*_corners_detected.jpg'}")

print(f"The chessboard was found in {len(used_images)} of {len(images)} photos.")
if not used_images:
    sys.exit(f"Error: the chessboard was not found in any photo. Check that chessboard_size "
             f"{chessboard_size} is the number of inner corners of your board.")
if len(used_images) < 3:
    sys.exit("Error: the calibration needs the chessboard in at least 3 photos (15 to 20 are better).")
if len(used_images) < 12:
    print("Note: this is enough to test the script, but a real calibration needs the chessboard "
          "in at least 12 photos (see the shot list in the README).")

# Step 4: Perform camera calibration
# rms is the RMS reprojection error in pixels (see Step 7)
flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None,
                                                                    flags=flags)

# Step 5: Display results
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Step 6: Save the calibration results
np.savez(output_folder / 'calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
         rvecs=rvecs, tvecs=tvecs, image_size=image_shape)
print(f"\nResults saved in: {output_folder}")

# Step 7: Check the calibration. The reprojection error is the distance, in pixels, between
# the corners found in a photo and the corners that the calibrated camera model predicts.
print(f"\nRMS reprojection error over all photos: {rms:.3f} px")
for i, fname in enumerate(used_images):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    # OpenCV 4 gives the corners as an array of shape (28, 1, 2) and OpenCV 5 as (28, 2), so both
    # arrays are reshaped to one row (x, y) per corner before they are compared.
    error = cv2.norm(imgpoints[i].reshape(-1, 2), imgpoints2.reshape(-1, 2), cv2.NORM_L2) / np.sqrt(len(imgpoints2))
    print(f"  {fname.name}: {error:.3f} px")  # RMS of this photo

# Step 8: Use the calibration results to undistort a photo from the same folder
img = cv2.imread(str(used_images[0]))
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Crop the image to the valid region (optional, based on ROI). If this region is tiny, the distortion
# coefficients are wrong, so we keep the whole image.
x, y, rw, rh = roi
if rw * rh >= 0.3 * w * h:
    undistorted_img = undistorted_img[y:y+rh, x:x+rw]
else:
    print("\nWarning: after undistortion only a small part of the image is valid. Take more photos with "
          "the chessboard near the corners of the image, and calibrate again.")

# Save the undistorted image, and display it next to the original
undistorted_path = output_folder / f"{used_images[0].stem}_undistorted.jpg"
cv2.imwrite(str(undistorted_path), undistorted_img)
print(f"\nUndistorted photo saved as: {undistorted_path}")
if show:
    print("Showing the photo before and after undistortion. Press any key in one of the windows to finish.")
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
