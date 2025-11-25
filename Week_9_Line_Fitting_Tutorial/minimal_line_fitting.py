"""
Minimal Line Fitting Example
Demonstrates how to fit a line to a segmented object using cv2.fitLine.
"""

import cv2
import numpy as np

import cv2
import numpy as np
import json
import os

# Open video
cap = cv2.VideoCapture("recovered_video.mp4")

# Load config
CONFIG_FILE = 'config.json'
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    roi = config['roi']
    lower = np.array(config['lower_bound'], dtype=np.uint8)
    upper = np.array(config['upper_bound'], dtype=np.uint8)
    print(f"Loaded config: ROI={roi}, HSV={lower} to {upper}")
    
    x, y, w, h = roi
else:
    print("Config file not found! Using defaults.")
    # Fallback defaults
    lower = np.array([21, 78, 152], dtype=np.uint8)
    upper = np.array([41, 255, 255], dtype=np.uint8)
    x, y, w, h = 0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop to ROI
    roi_frame = frame[y:y+h, x:x+w]
    
    # Convert to HSV and create mask
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for drawing
    result = frame.copy()
    
    # Draw ROI box
    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    if contours:
        # Find largest contour
        c = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c) > 50:
            # Fit line
            [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Extract scalar values
            vx_val = vx[0]
            vy_val = vy[0]
            x0_val = x0[0]
            y0_val = y0[0]
            
            # Calculate points to draw the line (relative to ROI)
            cols = roi_frame.shape[1]
            lefty = int((-x0_val * vy_val / vx_val) + y0_val)
            righty = int(((cols - x0_val) * vy_val / vx_val) + y0_val)
            
            # Adjust to full frame coordinates
            pt1 = (x + 0, y + lefty)
            pt2 = (x + cols - 1, y + righty)
            
            # Draw line
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)
            
            # Draw contour for reference (adjust offset)
            c_shifted = c + np.array([x, y])
            cv2.drawContours(result, [c_shifted], -1, (0, 0, 255), 1)

    # Display side-by-side
    cv2.imshow('Original | Line Fitted', np.hstack((frame, result)))
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
