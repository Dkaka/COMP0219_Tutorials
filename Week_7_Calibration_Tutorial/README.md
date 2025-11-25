# Camera Calibration Tutorial

This tutorial provides scripts for capturing calibration images and performing camera calibration using OpenCV.

## Prerequisites

Before starting, ensure you have created and activated your Python virtual environment:

1. **Create a Python Virtual Environment:**

   It is recommended to use a virtual environment to manage the project's dependencies. The following command will create a virtual environment named comp0219 at your machine (Pi or laptop):

   ```bash
   python3 -m venv comp0219 --system-site-packages
   ```

2. **Activate the Virtual Environment:**

   Before installing packages and running the scripts, you need to activate the virtual environment:

   For Raspberry Pi or Linux systems,
   ```bash
   source comp0219/bin/activate
   ```

   For Windows,
   ```bash
   comp0219\Scripts\activate.bat
   ``` 

## Installation

### System-Level Dependencies (apt-get)

For Raspberry Pi or Linux systems, install the required system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv libopencv-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
```

**Note:** If you're on a laptop running Linux, the same packages apply. For macOS or Windows, you can skip the apt-get commands and rely on pip installation alone.

### Python Dependencies (pip)

With your virtual environment activated, install the required Python packages:

```bash
pip install --upgrade pip
pip install numpy opencv-python opencv-contrib-python
```

Alternatively, you can install from a requirements file if provided:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Week_7_Calibration_Tutorial/
├── calibration_cap.py              # Script for capturing calibration images
├── calibration.py                  # Script for camera calibration
├── sample_calibration_images/      # Sample images for testing
└── README.md                       # This file
```

## Usage

### Step 1: Capture Calibration Images

Use `calibration_cap.py` to capture images of a chessboard pattern from different angles and distances.

```bash
python calibration_cap.py
```

**Controls:**
- Press `c` to capture and save an image
- Press `q` to quit the program

**Important Parameters:**
- `chessboard_size = (4, 7)`: Inner corners in the chessboard (4 columns × 7 rows)
- Images will be saved in `./calibration_images/` folder
- Recommended: Capture at least 10-15 images from different angles and positions

**Tips for capturing:**
- Ensure the entire chessboard is visible in the frame
- Capture images from different angles (tilted, rotated)
- Vary the distance from the camera
- Ensure good lighting with minimal shadows

### Step 2: Perform Camera Calibration

Once you have captured sufficient images, run `calibration.py` to calculate the camera's intrinsic parameters and distortion coefficients.

```bash
python calibration.py
```

**What it does:**
1. Detects chessboard corners in all captured images
2. Calculates camera matrix and distortion coefficients
3. Saves calibration data to `calibration_data.npz`
4. Displays original and undistorted images for verification
5. Outputs the total calibration error

**Important Parameters:**
- `chessboard_size = (4, 7)`: Must match the size used in `calibration_cap.py`
- `square_size = 34.0`: Size of one square in mm (adjust based on your actual chessboard)

**Output:**
- `calibration_data.npz`: Contains camera_matrix, dist_coeffs, rvecs, and tvecs
- Images with detected corners saved as `*_corners_detected.jpg`
- Camera calibration results printed to console

## Calibration Parameters

The calibration process determines:

1. **Camera Matrix (K)**: Contains focal lengths (fx, fy) and optical centers (cx, cy)
2. **Distortion Coefficients**: Corrects for lens distortion (radial and tangential)
3. **Rotation Vectors (rvecs)**: Rotation of the chessboard relative to camera
4. **Translation Vectors (tvecs)**: Position of the chessboard relative to camera

## Troubleshooting

### Camera Not Opening
- Ensure no other application is using the camera
- Try changing the camera index in `calibration_cap.py`: `cap = cv2.VideoCapture(1)` or `2`

### Chessboard Not Detected
- Ensure you're using the correct `chessboard_size` parameter
- Check that the chessboard is clearly visible and well-lit
- Make sure the entire chessboard is within the frame

### OpenCV Display Issues
- If `cv2.imshow()` doesn't work, ensure you have the required GUI libraries installed
- On headless systems, you may need to use X11 forwarding or skip visualization

### High Calibration Error
- Capture more images (20-30 recommended)
- Ensure images are sharp and not blurry
- Verify the `square_size` parameter matches your actual chessboard
- Check that all images show the chessboard clearly

## Sample Data

Sample calibration images are provided in `sample_calibration_images/` for testing the calibration process without capturing new images.

## References

- [OpenCV Camera Calibration Documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Camera Calibration Theory](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## Notes

- The calibration quality depends on the number and variety of images captured
- Use a high-quality printed chessboard pattern (avoid warped or curved surfaces)
- Ensure the chessboard pattern is flat during image capture
- The square size can be in any unit (mm, cm, inches) as long as it's consistent

