"""
HSV vs RGB Comparison

This demonstrates why HSV is better than RGB for color segmentation.
It shows the same color detection using both methods side-by-side.

Usage:
    python hsv_vs_rgb_comparison.py
    
Try clicking on different colored objects to see the difference!
"""

import cv2
import numpy as np

# Global variables
selected_bgr = None
selected_hsv = None

def mouse_callback(event, x, y, flags, param):
    """Pick color on click"""
    global selected_bgr, selected_hsv
    
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, hsv_frame = param
        
        # Get BGR and HSV values
        selected_bgr = frame[y, x].copy()
        selected_hsv = hsv_frame[y, x].copy()
        
        print(f"\n🎨 Selected color at ({x}, {y})")
        print(f"   RGB: R={selected_bgr[2]}, G={selected_bgr[1]}, B={selected_bgr[0]}")
        print(f"   HSV: H={selected_hsv[0]}, S={selected_hsv[1]}, V={selected_hsv[2]}")

def create_rgb_mask(frame, target_bgr, tolerance=60):
    """Create mask using RGB color space"""
    if target_bgr is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Create range for each channel
    lower = np.array([
        max(0, int(target_bgr[0]) - tolerance),
        max(0, int(target_bgr[1]) - tolerance),
        max(0, int(target_bgr[2]) - tolerance)
    ], dtype=np.uint8)
    
    upper = np.array([
        min(255, int(target_bgr[0]) + tolerance),
        min(255, int(target_bgr[1]) + tolerance),
        min(255, int(target_bgr[2]) + tolerance)
    ], dtype=np.uint8)
    
    # Create mask in BGR space
    mask = cv2.inRange(frame, lower, upper)
    return mask

def create_hsv_mask(hsv_frame, target_hsv):
    """Create mask using HSV color space"""
    if target_hsv is None:
        return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    
    # HSV tolerances (can be different for each channel)
    h_tol = 10   # Small for hue (color type)
    s_tol = 60   # Larger for saturation
    v_tol = 60   # Larger for value/brightness
    
    lower = np.array([
        max(0, int(target_hsv[0]) - h_tol),
        max(0, int(target_hsv[1]) - s_tol),
        max(0, int(target_hsv[2]) - v_tol)
    ], dtype=np.uint8)
    
    upper = np.array([
        min(179, int(target_hsv[0]) + h_tol),
        min(255, int(target_hsv[1]) + s_tol),
        min(255, int(target_hsv[2]) + v_tol)
    ], dtype=np.uint8)
    
    mask = cv2.inRange(hsv_frame, lower, upper)
    return mask

def main():
    global selected_bgr, selected_hsv
    
    # Open video
    video_path = "Sample_Color_Segmentation.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Get first frame for color selection
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video '{video_path}'")
        return
    
    hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    
    # Setup for color selection
    window_name = 'Click to Select Color'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, (first_frame, hsv_first))
    
    print("\n" + "="*70)
    print("🎨 HSV vs RGB COLOR SEGMENTATION COMPARISON")
    print("="*70)
    print("Instructions:")
    print("  1. Click on a colored object in the frame")
    print("  2. Press ENTER to start the comparison")
    print("  3. Watch how HSV and RGB perform differently")
    print("  4. Press 'q' to quit")
    print("\nWhy is this comparison important?")
    print("  - RGB treats all channels equally (Red, Green, Blue)")
    print("  - HSV separates color (H) from brightness (V)")
    print("  - HSV is more robust to lighting changes!")
    print("="*70 + "\n")
    
    # Wait for color selection
    while True:
        display = first_frame.copy()
        cv2.putText(display, "Click on a color, then press ENTER", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if selected_bgr is not None:
            cv2.putText(display, "Color selected! Press ENTER", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and selected_bgr is not None:  # ENTER
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cv2.destroyWindow(window_name)
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create comparison windows
    print("\n📊 Processing video - compare the results!")
    print("   Layout: Original | RGB Detection | HSV Detection")
    print("\nControls:")
    print("   ENTER - Pause/Resume")
    print("   'q'   - Quit")
    print("\nNotice which method is more stable!\n")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
                continue
            
            frame_count += 1
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create masks using both methods
            rgb_mask = create_rgb_mask(frame, selected_bgr, tolerance=60)
            hsv_mask = create_hsv_mask(hsv, selected_hsv)
            
            # Apply masks
            rgb_result = cv2.bitwise_and(frame, frame, mask=rgb_mask)
            hsv_result = cv2.bitwise_and(frame, frame, mask=hsv_mask)
            
            # Create display frame with original on left
            display_frame = frame.copy()
            cv2.putText(display_frame, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add labels to results
            cv2.putText(rgb_result, "RGB Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(hsv_result, "HSV Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Calculate detection percentages
            rgb_percent = (np.count_nonzero(rgb_mask) / rgb_mask.size) * 100
            hsv_percent = (np.count_nonzero(hsv_mask) / hsv_mask.size) * 100
            
            cv2.putText(rgb_result, f"{rgb_percent:.1f}% detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(hsv_result, f"{hsv_percent:.1f}% detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show 3-panel comparison: Original | RGB | HSV
            comparison = np.hstack((display_frame, rgb_result, hsv_result))
        
        # Add pause indicator
        if paused:
            cv2.putText(comparison, "PAUSED (Press ENTER to resume)", (10, comparison.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        cv2.imshow('Original | RGB Detection | HSV Detection (ENTER=pause, q=quit)', comparison)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:  # ENTER key
            paused = not paused
            status = "PAUSED" if paused else "PLAYING"
            print(f"Video {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Comparison complete!")
    print("\n💡 Key Takeaways:")
    print("   1. HSV is usually more consistent across lighting changes")
    print("   2. RGB can miss colors when brightness varies")
    print("   3. HSV allows independent control of hue, saturation, and value")
    print("   4. For robust color tracking, HSV is the better choice!")

if __name__ == "__main__":
    main()

