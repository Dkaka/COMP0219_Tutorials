"""
Minimal Color Segmentation Example
The simplest possible implementation - just the core concept!
"""

import cv2
import numpy as np

# Open video
cap = cv2.VideoCapture("Sample_Color_Segmentation.mp4")
ret, frame = cap.read()

# Convert to HSV and pick a color at position (300, 300)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, s, v = hsv[875, 1452]  # Pick color at this position

# Create color range (±10 hue, ±60 saturation/value)
# Use max/min to cap values within valid HSV range
lower = np.array([max(0, int(h)-10), max(0, int(s)-60), max(0, int(v)-60)], dtype=np.uint8)
upper = np.array([min(179, int(h)+10), min(255, int(s)+60), min(255, int(v)+60)], dtype=np.uint8)

print(f"Selected HSV: H={h}, S={s}, V={v}")
print(f"Range: {lower} to {upper}")

# Process video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart from beginning

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to HSV and create mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Apply mask (keep color, black out rest)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Display side-by-side
    cv2.imshow('Original | Segmented', np.hstack((frame, result)))
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

