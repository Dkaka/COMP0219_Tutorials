# Week 8: Color Segmentation Tutorial

This tutorial introduces color segmentation using OpenCV, demonstrating how to detect and track colored objects in video. The scripts are designed to be used in sequence to build understanding from basic concepts to practical comparisons.

## Files Overview

- `Sample_Color_Segmentation.mp4` - Sample video for testing color segmentation (you can capture you own with your phone or webcam)
- `get_coordinates.py` - Interactive tool to pick pixel coordinates and colors
- `minimal_example.py` - Basic color segmentation implementation
- `hsv_vs_rgb_comparison.py` - Compare RGB vs HSV segmentation methods

## Recommended Workflow

### Step 1: Get Coordinates (`get_coordinates.py`)

**Purpose:** Find the pixel coordinates and color values of objects you want to track.

**What it does:**
- Displays the first frame of the video
- Shows a crosshair at the center for reference
- Allows you to click anywhere to see:
  - Pixel coordinates (x, y)
  - BGR color values
  - HSV color values
- Provides ready-to-use code snippets for `minimal_example.py`

**Usage:**
```bash
python get_coordinates.py
```

**Instructions:**
1. Click on any colored object in the frame
2. Note the coordinates and HSV values printed in the terminal
3. Copy the suggested code snippet to use in Step 2
4. Press any key to exit

**Example output:**
```
============================================================
 CLICKED POSITION: (1452, 875)
============================================================
BGR values: B=45, G=180, R=230
HSV values: H=8, S=180, V=230
============================================================

 To use in minimal_example.py, change line 14 to:
   h, s, v = hsv[875, 1452]  # 875 is y-coordinate, 1452 is x-coordinate
```

---

### Step 2: Minimal Example (`minimal_example.py`)

**Purpose:** Demonstrate the simplest possible color segmentation implementation.

**What it does:**
- Reads the sample video
- Picks a color at a specific pixel coordinate (from Step 1)
- Creates a color range around that color in HSV space
- Applies a mask to segment/isolate that color
- Displays original and segmented video side-by-side

**Key concepts demonstrated:**
- Converting BGR to HSV color space
- Creating color ranges with tolerance
- Using `cv2.inRange()` to create binary masks
- Applying masks with `cv2.bitwise_and()`

**Usage:**
```bash
python minimal_example.py
```

**How to customize:**
1. Run `get_coordinates.py` to find a color you want to track
2. Update line 15 with the coordinates from Step 1
3. Run the script to see the segmentation in action
4. Press 'q' to quit

**Code snippet (line 14-20):**
```python
# Pick color at position from get_coordinates.py
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, s, v = hsv[875, 1452]  # Use coordinates from Step 1

# Create color range with tolerance
lower = np.array([max(0, int(h)-10), max(0, int(s)-60), max(0, int(v)-60)])
upper = np.array([min(179, int(h)+10), min(255, int(s)+60), min(255, int(v)+60)])
```

---

### Step 3: HSV vs RGB Comparison (`hsv_vs_rgb_comparison.py`)

**Purpose:** Understand WHY HSV is better than RGB for color segmentation.

**What it does:**
- Allows interactive color selection by clicking
- Processes the video using BOTH RGB and HSV methods
- Displays a 3-panel comparison:
  - Original frame
  - RGB-based detection
  - HSV-based detection
- Shows detection percentages for each method
- Supports pause/resume for detailed analysis

**Key insights:**
- **RGB method:** Treats R, G, B channels equally, sensitive to lighting changes
- **HSV method:** Separates color (Hue) from brightness (Value), more robust
- **Real-world benefit:** HSV maintains better tracking under varying lighting conditions

**Usage:**
```bash
python hsv_vs_rgb_comparison.py
```

**Instructions:**
1. Click on a colored object in the frame
2. Press ENTER to start the comparison
3. Watch how RGB and HSV perform differently
4. Press ENTER to pause/resume
5. Press 'q' to quit

**What to observe:**
- HSV typically maintains more consistent detection
- RGB may miss colors when brightness/shadows change
- HSV detection percentage is usually more stable

---

## Color Segmentation Concepts

### HSV Color Space
- **H (Hue):** The color type (0-179 in OpenCV)
- **S (Saturation):** Color intensity/purity (0-255)
- **V (Value):** Brightness (0-255)

### Why HSV for Color Segmentation?
1. **Separates color from brightness** - Hue represents the actual color
2. **More intuitive** - Easier to define color ranges
3. **Lighting invariant** - More robust to shadows and lighting changes
4. **Better for tracking** - Maintains consistent detection

### Typical HSV Tolerances
```python
h_tolerance = 10   # Small (hue is circular, represents color type)
s_tolerance = 60   # Larger (saturation can vary more)
v_tolerance = 60   # Larger (brightness varies with lighting)
```

## Common Use Cases

- **Object tracking** - Track colored markers or objects
- **Robot vision** - Detect colored targets or boundaries
- **Sports analysis** - Track colored jerseys or balls
- **Industrial inspection** - Detect colored defects or components
- **Augmented reality** - Track colored markers for AR overlays

## Requirements

use the same requirement as before with venv

## Tips

1. **Choose distinctive colors** - High saturation colors work best (e.g., bright red, blue, yellow, like our pingpong ball)
2. **Test under different lighting** - Real-world conditions vary
3. **Adjust tolerances** - Increase if detection is too selective, decrease if detecting too much
4. **Use Step 3 to validate** - Always compare methods to understand trade-offs
5. **Consider morphological operations** - Add erosion/dilation to clean up noise (advanced topic)

## Next Steps

After completing this tutorial, you can:
- Apply color segmentation to live camera feeds
- Implement object tracking with contours
- Add morphological operations for noise reduction
- Combine with other CV techniques (e.g., edge detection, blob detection)
- Build real-time tracking applications

---


