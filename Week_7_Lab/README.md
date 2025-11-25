# Week 7: Raspberry Pi Camera Lab

This lab provides a series of incremental scripts to get you started with the Raspberry Pi camera using Picamera2. Each step has a specific learning goal and a corresponding script.

---

## A. Raspberry Pi Setup

This session covers the setup process for the Raspberry Pi 5 lab sessions. Since we only have monitor access and tunnel during lab sessions, the tutorials focus on learning to code individual sessions without requiring immediate Pi access.

### Important Considerations

#### Workflow
1. Obtain video footage from a convenient camera or dataset at the lab sessions
2. Apply computer vision techniques to achieve the intended outcome at tutorial sessions (This can be done on your laptop or the Pi)
3. Try your code from 2 at the lab session on the Pi with the given camera.
#### Key Differences to Keep in Mind

**Video Format Differences:**
- There will likely be format/resolution/frame rate differences between your source video and the video captured by the Pi
- Plan accordingly when testing your code

**Performance Differences:**
- There may be differences in performance or availability of hardware (e.g., hardware encoder) between your development environment and the Pi
- You might see performance differences between tutorial results and actual results on the Pi, especially when running in real-time

**Optimization Strategies:**
Consider these approaches to ensure your project runs in real-time:
- Use lower resolution
- Reduce frame rate
- Process only one color channel
- Modify hardware design

### Setup Instructions

#### 1. Initial Raspberry Pi Setup

Follow the official Raspberry Pi documentation:
- Visit: https://www.raspberrypi.com/documentation/computers/getting-started.html
- **Device:** Raspberry Pi 5
- **OS:** Raspberry Pi OS 64-bit with Bookworm


We also supply sd card that are pre-flashed of user name and password both comp0219.

#### 2. Physical Setup

Use the lab monitor and mouse/keyborad to continue the setup process.

#### 3. Network Connection

Connect your Pi to eduroam Wi-Fi:
- Follow the UCL ISD guide: https://www.ucl.ac.uk/isd/how-to/connecting-to-eduroam-wi-fi-linux

#### 4. Remote Access Setup

##### Option A: Pi Connect (Recommended)

Use Raspberry Pi Connect for remote access:
- Documentation: https://www.raspberrypi.com/documentation/services/connect.html
- Pi Connect is already installed on your OS (no need to install Connect Lite)
- **Advantage:** Screen sharing without a physical monitor - the Pi spawns a virtual monitor and streams from there

##### Option B: Custom SSH/VNC Setup

You're welcome to use your own SSH or VNC setup, but note:
- Limited troubleshooting support will be available for custom configurations

## B. Environment Setup
To replicate the environment for this project, follow these steps:

#### 1.  Install the system-level dependencies:

```bash
sudo apt-get install -y libcap-dev python3-libcamera python3-opencv
```

#### 2.  Create and activate a new virtual environment:

It is recommended to use a virtual environment to manage the project's dependencies. The following command will create a virtual environment named `comp0219`and activate the virtual environment before installing packages and running the scripts.

```bash
python3 -m venv new_environment --system-site-packages
source new_environment/bin/activate
```

#### 3.  Install the Python packages:

Once the virtual environment is activated, install the necessary Python packages.
    
```bash
pip install -r requirements.txt
```

## C. Lab

### 0. Sanity Checks (Command Line)

**Goal:** Verify that the camera is detected and working correctly before writing any Python code.

**Commands:**

These commands use the `rpicam-apps` suite of tools, which are the official command-line utilities for interacting with the camera on Raspberry Pi OS.

```bash
# Update and install the official camera apps (rpicam-*), if needed
sudo apt update
sudo apt install -y rpicam-apps

# 1) “Hello” preview for 5 s (opens a live window)
# The -t 5000 argument sets the duration of the preview in milliseconds.
rpicam-hello -t 5000

# 2) Capture a high-resolution still
# The -o argument specifies the output file name.
rpicam-still -o test.jpg

# 3) Quick video test: 10 s at 1080p30 (H.264)
# -t: duration in ms
# -o: output file name
# --width, --height: resolution
# --framerate: frames per second
rpicam-vid -t 10000 -o test.h264 --width 1920 --height 1080 --framerate 30
```

**Script:** [`0_sanity_checks.sh`](0_sanity_checks.sh)

**Why CLI first?** If these commands fail, your Python scripts will also fail. These `rpicam-*` tools are the standard way to interact with the camera on Raspberry Pi OS.

---

### 1. Minimal Python Preview

**Goal:** Open the camera, acquire frames as NumPy arrays, and display them in a window.

**Script:** [`1_minimal_python_preview.py`](1_minimal_python_preview.py)

**Explanation:** This script is the “Hello, World!” of Picamera2. It shows the fundamental steps of creating a `Picamera2` object, configuring it for preview, starting the camera, and capturing frames in a loop. The captured frames are NumPy arrays, which can be easily manipulated and displayed with OpenCV.

---

### 2. Query Basics & Capture a Still

**Goal:** Read sensor metadata and capture a still image at a chosen resolution.

**Script:** [`2_still_capture_and_info.py`](2_still_capture_and_info.py)

**Explanation:** This script demonstrates how to get information about the camera and how to capture a high-resolution still image. The `camera_properties` attribute provides a dictionary of information about the camera sensor. The `create_still_configuration` method is used to configure the camera for high-resolution captures.

---

### 3. Toggle Resolution and Frame-rate

**Goal:** Change the output size and request a target frame rate using `FrameDurationLimits`.

**Script:** [`3_resolution_and_fps.py`](3_resolution_and_fps.py)

**Explanation:** This script shows how to control the video resolution and frame rate. The `FrameDurationLimits` control is the recommended way to set a specific frame rate. The script also demonstrates how to reconfigure the camera on the fly to change the resolution and frame rate based on user input.

---

### 3.1. High Framerate, Lower Resolution

**Goal:** Achieve a high frame rate (120fps) by using a lower resolution.

**Script:** [`3.1_high_fps_low_res.py`](3.1_high_fps_low_res.py)

**Explanation:** To achieve very high frame rates, you often need to reduce the resolution. This is because the camera sensor has a limited data readout speed. This script shows how to configure the camera for 120fps at a resolution of 1280x720. Note that not all resolutions are capable of high frame rates; you need to use a resolution that is supported by one of the camera's high-speed sensor modes.

**Can you still capture at 1080p?** Capturing at 1080p at 120fps is generally not possible with the Raspberry Pi HQ camera, as it exceeds the sensor's capabilities. You would need a specialized camera to achieve that.

---

### 4. Dual-stream (lores + main)

**Goal:** Produce a main high-quality stream and a low-resolution stream simultaneously. The low-res stream is ideal for real-time processing with OpenCV.

**Script:** [`4_dual_stream_preview.py`](4_dual_stream_preview.py)

**Explanation:** This script demonstrates a powerful feature of Picamera2: the ability to create multiple streams with different properties. This is very useful for applications where you need to perform real-time analysis on a low-resolution stream while simultaneously recording or displaying a high-resolution stream.

---

### 5. Record H.264 Video to MP4

**Goal:** Save an encoded video to a file and lock the auto-exposure and auto-white balance for consistent recordings.

**Script:** [`5_record_video_h264.py`](5_record_video_h264.py)

**Explanation:** This script shows how to record a video to a file using an H.264 encoder. It also demonstrates how to lock the auto-exposure and auto-white balance settings to ensure that the video has consistent brightness and color. This is crucial for any scientific or analytical application where reproducibility is important.
