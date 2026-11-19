#!/bin/bash

# This script runs some basic checks to verify that the camera is working correctly.
# Run it in this folder with:   bash 0_sanity_checks.sh
# The camera apps (rpicam-apps) are already installed, so the script installs nothing.
# It saves test.jpg and test.mp4 in this folder.

failed=""

# Run one command. If the command fails, remember it.
run() {
    echo
    echo ">>> $*"
    if ! "$@"; then
        echo "FAILED: $*"
        failed="$failed
    $*"
    fi
}

# 0) Is a camera detected? The HQ camera is listed as imx477.
#    When there is no camera, this command prints "No cameras available!"
#    but still ends without an error, so we check what it prints.
echo ">>> rpicam-hello --list-cameras"
cameras=$(rpicam-hello --list-cameras 2>&1)
status=$?
echo "$cameras"
if [ $status -ne 0 ] || ! echo "$cameras" | grep -q "Available cameras"; then
    echo
    echo "No camera was found, or the command failed (see the message above),"
    echo "so the other checks were not run."
    echo "Shut down the Pi with 'sudo poweroff', check that the camera cable is pushed in"
    echo "fully at both ends, then start the Pi and run this script again."
    echo "If it still fails, ask a demonstrator."
    exit 1
fi

# 1) Live preview for 3 seconds. A window appears only on the Pi desktop (lab monitor or
#    Pi Connect Screen sharing). In Remote shell no window appears, but the camera is still tested.
echo
echo "The next check shows a preview for 3 seconds. The window appears only on the Pi desktop"
echo "(lab monitor or Pi Connect Screen sharing). In Remote shell no window appears. This is normal."
run rpicam-hello -t 3000

# 2) Capture a full-resolution still image
run rpicam-still -o test.jpg

# 3) Record a 10-second video at 1920 x 1080 pixels and 30 frames per second, as an MP4 file
run rpicam-vid -t 10000 -o test.mp4 --width 1920 --height 1080 --framerate 30

echo
if [ -z "$failed" ]; then
    echo "All checks passed. The camera works, and test.jpg and test.mp4 are in this folder."
    echo "Double-click them in the file manager to check them."
else
    echo "These commands failed:$failed"
    echo "Read the error messages above. If you cannot fix the problem, ask a demonstrator."
    exit 1
fi
