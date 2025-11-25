import cv2
import numpy as np
import glob

# Step 1: Prepare object points
# Define the chessboard size (number of internal corners)
chessboard_size = (4, 7)  # Adjust based on your calibration board
square_size = 34.0  # Size of a square in your defined unit (e.g., 1.0 for meters or mm)

# Prepare the object points for a standard chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in the real world
imgpoints = []  # 2D points in the image plane
image_shape = None  # Store image shape for calibration

# Step 2: Load calibration images
images = glob.glob('./sample_calibration_images/*.jpg')  # Adjust the path as needed

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store the image shape from the first image
    if image_shape is None:
        image_shape = gray.shape[::-1]

    # Step 3: Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)  # Add object points
        imgpoints.append(corners)  # Add image points

        # Optional: Draw and display the corners for visualization
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
        output_fname = fname.replace('.jpg', '_corners_detected.jpg')  # Adjust extension if necessary
        cv2.imwrite(output_fname, img)

cv2.destroyAllWindows()

# Step 4: Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

# Step 5: Display results
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Step 6: Save the calibration results
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

# Step 7: Use the calibration results to undistort an image
img = cv2.imread('./calibration_images/calibration_image_02.jpg')  # Replace with an image to undistort
# img = cv2.cvtColor()
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)


# Crop the image (optional, based on ROI)
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))

# Display the original and undistorted images
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
