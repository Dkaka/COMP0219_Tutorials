import cv2
import os

# Parameters
output_folder = "./calibration_images"  # Folder to save the images
chessboard_size = (4, 7)  # Inner corners in the chessboard
capture_key = 'c'  # Key to capture the image
exit_key = 'q'  # Key to exit the program


# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the webcam (use the appropriate index if multiple cameras are connected)
cap = cv2.VideoCapture(0)  # Change the index to 1, 2, etc., if needed

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print(f"Press '{capture_key}' to capture an image and save it.")
print(f"Press '{exit_key}' to quit.")

image_count = 0  # Counter for saved images

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Convert the frame to grayscale for chessboard detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show the frame
    cv2.imshow("Calibration Capture", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord(capture_key):  # Capture and save the image
        if ret:
            image_path = os.path.join(output_folder, f"calibration_image_{image_count:02d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            image_count += 1
        else:
            print("No chessboard detected. Please adjust the position and try again.")

    elif key == ord(exit_key):  # Exit the program
        print("Exiting...")
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
