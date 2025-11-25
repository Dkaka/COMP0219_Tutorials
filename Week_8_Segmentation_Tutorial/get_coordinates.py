"""
Coordinate Picker Tool

Click on the video to see pixel coordinates and HSV/RGB values.
Use these coordinates in minimal_example.py

Usage:
    python get_coordinates.py
"""

import cv2
import numpy as np

# Global to store the frame
current_frame = None
current_hsv = None

def mouse_callback(event, x, y, flags, param):
    """Display coordinates and color values when clicking"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get BGR and HSV values
        bgr = current_frame[y, x]
        hsv = current_hsv[y, x]
        
        print("\n" + "="*60)
        print(f" CLICKED POSITION: ({x}, {y})")
        print("="*60)
        print(f"BGR values: B={bgr[0]}, G={bgr[1]}, R={bgr[2]}")
        print(f"HSV values: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
        print("="*60)
        print(f"\n To use in minimal_example.py, change line 14 to:")
        print(f"   h, s, v = hsv[{y}, {x}]  # {y} is y-coordinate, {x} is x-coordinate")
        print()

def main():
    global current_frame, current_hsv
    
    # Open video
    video_path = "Sample_Color_Segmentation.mp4"
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video '{video_path}'")
        return
    
    current_frame = frame.copy()
    current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create window
    window_name = 'Click to Get Coordinates (Press any key to exit)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "🎯 COORDINATE PICKER TOOL")
    print("="*60)
    print("Instructions:")
    print("  1. Click anywhere on the frame")
    print("  2. See the coordinates and color values in the terminal")
    print("  3. Copy the coordinates to use in minimal_example.py")
    print("  4. Press any key to exit")
    print("="*60 + "\n")
    
    # Add crosshair at center for reference
    height, width = frame.shape[:2]
    cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 1)
    cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 1)
    
    # Add text
    cv2.putText(frame, "Click to get coordinates", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Center: ({width//2}, {height//2})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

