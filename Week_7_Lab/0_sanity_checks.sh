#!/bin/bash

# This script runs some basic checks to verify that the camera is working correctly.

# Update and install the official camera apps (rpicam-*), if needed
sudo apt update
sudo apt install -y rpicam-apps

# 1) “Hello” preview for 5 s (opens a live window)
rpicam-hello -t 5000

# 2) Capture a high-resolution still
rpicam-still -o test.jpg

# 3) Quick video test: 10 s at 1080p30 (H.264)
rpicam-vid -t 10000 -o test.h264 --width 1920 --height 1080 --framerate 30

echo "Sanity checks complete. If you saw a preview window, a test.jpg file was created, and a test.h264 file was created, then your camera is working correctly."
