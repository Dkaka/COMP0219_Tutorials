# Week 6: Raspberry Pi Camera Tutorial

This tutorial gives you a series of short scripts that show how to use the Raspberry Pi camera with Picamera2. Each script builds on the one before it. Each step has a specific learning goal and a corresponding script.

---

## A. Before you start

Finish the Pi setup first: [`../week06_pi_setup/README.md`](../week06_pi_setup/README.md) for a Raspberry Pi 5, or [`../week06_pi_setup/CM5_Dev_Kit_Guide.md`](../week06_pi_setup/CM5_Dev_Kit_Guide.md) for the CM5 Development Kit. Do not write a new operating system to the SD card or the eMMC, because staff have already installed it.

When you start this tutorial, the camera must work with `rpicam-hello`, and you must be able to open a terminal on the Pi desktop.

## B. Set up the Python environment (on the Raspberry Pi)

Do these steps on the Raspberry Pi. The Picamera2 scripts in this folder need the Raspberry Pi camera and its libcamera software, so they cannot run on a laptop. The libcamera software cannot be installed with pip.

On the Pi, Picamera2, OpenCV and NumPy must all come from apt, which is the package manager of Raspberry Pi OS. Do not install anything with pip for this tutorial.

#### 1. Install the system packages with apt

The Raspberry Pi OS desktop image already includes `python3-picamera2`. The command below adds the packages that are still missing, for example OpenCV, so the Pi needs a working internet connection.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-matplotlib
```

#### 2. Create and activate the virtual environment

We use one virtual environment, named `comp0219`, for all the Part 2 tutorials. Create it in your home folder (`~`). The option `--system-site-packages` lets the environment use the apt packages from step 1. Without this option, the environment cannot see Picamera2.

```bash
python3 -m venv ~/comp0219 --system-site-packages
source ~/comp0219/bin/activate
```

If you already created `comp0219` earlier (for example in section F of the Pi 5 setup guide, or in section H of the CM5 guide), activate it with `source ~/comp0219/bin/activate`. Do not create it again.

Every time you open a new terminal, activate the environment again with `source ~/comp0219/bin/activate`.

#### 3. Check that Picamera2, OpenCV and NumPy load

Run this command inside the activated environment. It should print two version numbers, first the OpenCV version and then the NumPy version, and no error.

```bash
python -c "from picamera2 import Picamera2; import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

On Bookworm, the NumPy version starts with 1. On Trixie, the newer version of Raspberry Pi OS, it starts with 2. Both are correct when they come from apt. To see which version your Pi runs, type `grep VERSION_CODENAME /etc/os-release`. It prints a line such as `VERSION_CODENAME=trixie`.

#### Do not install camera packages with pip on the Pi

Do not run `pip install picamera2`, `pip install opencv-python` or `pip install numpy` on the Pi, and do not pip install other packages that install their own copy of NumPy. Inside the virtual environment, the pip copies hide the apt copies. The compiled packages that Picamera2 uses (such as simplejpeg and PyAV) were built for the NumPy version that comes from apt. So Picamera2 then stops working, often with the error `numpy.dtype size changed, may indicate binary incompatibility`. If this happens, remove the pip copies, for example with `pip uninstall numpy opencv-python`. If that is not enough, delete the environment folder `~/comp0219` and create the environment again with step 2. Then run the check in step 3 again.

The file `requirements.txt` in this folder lists no packages on purpose. It only explains this rule, so `pip install -r requirements.txt` installs nothing.

## C. Tutorial

**Get the files.** The files of this tutorial are in the public course repository on GitHub. On the Raspberry Pi, open a terminal and run these commands. They download the repository into your home folder and go to the folder of this tutorial:

```bash
cd ~
git clone --depth 1 https://github.com/Dkaka/COMP0219_Tutorials.git
cd ~/COMP0219_Tutorials/tutorials/week06_picamera2_tutorial
```

If `git clone` says that the folder `COMP0219_Tutorials` already exists, the repository is already on this kit. Then run `cd ~/COMP0219_Tutorials` and `git pull` to get the newest version, and go to the tutorial folder with the last command above.

In this folder, run each script with `python <name of the script>`, for example `python 1_minimal_python_preview.py`. Before you run a script, activate the environment in this terminal with `source ~/comp0219/bin/activate` (section B). Run the shell script of section 0 with `bash 0_sanity_checks.sh`.

Work through the sections in order. In a list of files, `3.1_high_fps_low_res.py` and `3.2_reading_per_frame.py` appear before `3_resolution_and_fps.py`, and `5.1_record_120fps.py` appears before `5_record_video_h264.py`. Do script 3 before scripts 3.1 and 3.2, and script 5 before script 5.1.

The scripts save their images, videos and data files in the folder in which you run them. If you follow the steps above, this is `~/COMP0219_Tutorials/tutorials/week06_picamera2_tutorial`. If you run a script again, it replaces the file from the earlier run, because it uses the same file name. Section D shows how to copy a file to your laptop.

**Where to run the scripts.** Programs that open a window must run in a terminal on the Pi desktop: either on the lab monitor, or inside Raspberry Pi Connect **Screen sharing**. This applies to `rpicam-hello` and to scripts 1, 3, 3.1, 3.2 and 4 of this tutorial. Raspberry Pi Connect **Remote shell** has no desktop, so these programs cannot open a window there. `rpicam-hello` then shows no picture, and the Python scripts stop with an error. `rpicam-still`, `rpicam-vid` and scripts 2, 5 and 5.1 do not need a window, so they also work in Remote shell. Script 3.2 also works there if you turn its window off (section 3.2). A program that you start in Remote shell stops when you close the browser tab.

**Only one program can use the camera at a time.** If Python stops with `Camera __init__ sequence did not complete` or `Device or resource busy`, another program is still using the camera. Look also for the line `Pipeline handler in use by another process` above the error. It means the same thing. Find the terminal that runs `rpicam-hello` or another camera script and press Ctrl+C there, or close its window. If several team members use the Pi through Pi Connect, agree who runs the camera. If you cannot find the program, restart the Pi.

**If the camera is not found.** If `rpicam-hello` says that no cameras are available, or Python stops with `No camera number 0 found`, the Pi does not detect the camera. Follow the Troubleshooting section of your setup guide. For example, shut down the Pi with `sudo poweroff`, unplug the power, and check both ends of the camera cable.

---

### 0. Check the Camera from the Command Line

**Goal:** Check that the camera is detected and works before you write any Python code.

If you already did the camera test in the Pi setup guide today and it worked, skip this section.

**Commands:**

The camera apps (`rpicam-apps`) are already installed. They are the official command-line programs for the camera on Raspberry Pi OS.

Run `rpicam-hello` in a terminal on the Pi desktop (lab monitor or Pi Connect Screen sharing), not in Remote shell, because it opens a preview window and Remote shell has no desktop. `rpicam-still` and `rpicam-vid` open no window, so they also work in Remote shell.

```bash
# 1) Live preview until you press Ctrl+C. Use it to focus the lens.
rpicam-hello -t 0

# 2) Capture a still image
rpicam-still -o week6_test.jpg

# 3) Record a 10-second video at 1920 x 1080 pixels and 30 frames per second, as an MP4 file
rpicam-vid -t 10000 -o week6_test.mp4 --width 1920 --height 1080 --framerate 30
```

Open `week6_test.jpg` to check that it is sharp. To check the video, double-click `week6_test.mp4` in the file manager.

**Script:** [`0_sanity_checks.sh`](0_sanity_checks.sh)

Run it with `bash 0_sanity_checks.sh`. It runs the same checks one after another. First it runs `rpicam-hello --list-cameras` and stops with a clear message if no camera is found. Then it shows a preview for 3 seconds with `rpicam-hello -t 3000` (in Remote shell no window appears), saves `test.jpg` and records `test.mp4`. At the end it prints "All checks passed" only if every command succeeded. Otherwise it names the command that failed.

**Why test the camera from the command line first?** If these commands fail, your Python scripts will also fail. The `rpicam-*` programs are the standard camera tools of Raspberry Pi OS, so they show whether the camera itself works.

---

### 1. Minimal Python Preview

**Goal:** Open the camera, acquire frames as NumPy arrays, and display them in a window.

**Script:** [`1_minimal_python_preview.py`](1_minimal_python_preview.py)

Run it with `python 1_minimal_python_preview.py`. It opens a window, so run it on the Pi desktop. To stop it, click the window and press `q`.

**Explanation:** This is the shortest useful Picamera2 program. It shows the basic steps: create a `Picamera2` object, configure it for preview, start the camera, and capture frames in a loop. The captured frames are NumPy arrays, which you can process and display with OpenCV. Although the format is called `"RGB888"`, Picamera2 stores each pixel in the order B, G, R, which is the order that OpenCV expects. So you can pass the frames straight to OpenCV, and in later tutorials you convert a Pi frame to HSV with `cv2.COLOR_BGR2HSV`, with no extra conversion before it.

---

### 2. Read the Camera Information and Capture a Still

**Goal:** Read information about the camera sensor and capture a still image at a chosen resolution.

**Script:** [`2_still_capture_and_info.py`](2_still_capture_and_info.py)

Run it with `python 2_still_capture_and_info.py`. It opens no window, so it also works in Remote shell. It saves the image as `still_fullres.jpg`.

**Explanation:** This script shows how to get information about the camera and how to capture a high-resolution still image. The `camera_properties` attribute is a dictionary with information about the camera sensor, for example its model and its pixel array size. The bit depth is not in this dictionary. It is in `picam2.sensor_modes`, the list of sensor modes (see section 3.1). The script prints this list before `picam2.configure()`. To read the list, Picamera2 configures the camera once in every mode. This takes a few seconds, and it would undo your own configuration if you read the list after `configure()`. The exact names in the `camera_properties` dictionary depend on the libcamera version. For example, libcamera 0.8 shows `PixelArrayArea` instead of `PixelArraySize`. The `create_still_configuration` method configures the camera for high-resolution images.

---

### 3. Change the Resolution and the Frame Rate

**Goal:** Change the output size and request a target frame rate using `FrameDurationLimits`.

**Script:** [`3_resolution_and_fps.py`](3_resolution_and_fps.py)

Run it with `python 3_resolution_and_fps.py`. It opens a window, so run it on the Pi desktop.

**Explanation:** This script shows how to control the video resolution and frame rate. The `FrameDurationLimits` control is the recommended way to set a specific frame rate. The script also shows how to configure the camera again while the script runs, so that you can change the resolution and frame rate with a key. The script starts at 1280x720 and 30 fps. Click the window, then press `1` for 640x480 at 40 fps, `2` for 1280x720 at 30 fps, `3` for 1920x1080 at 30 fps, or `q` to quit. Every second, the script prints the number of frames that the loop received and showed in the last second. This count starts again after every change.

The output size does not choose a fast sensor mode for you. This script does not ask for a sensor mode. So Picamera2 picks the 12-bit sensor mode whose size and shape are closest to the output size, and it does not look at the frame rate that you ask for. After every change, the script prints the sensor mode that Picamera2 chose, so read this line. For example, on current Raspberry Pi OS, 640x480 uses the 12-bit 1332x990 mode. This mode uses only the centre of the sensor, so the view is narrower. (On older systems, 640x480 uses the 2028x1520 mode.) 1280x720 uses the 2028x1080 mode, which can reach about 63 fps with 12 bits (about 50 fps on older systems). The 40 fps of key `1` is only an example. If you ask for a frame rate that the sensor mode cannot reach, the camera runs at the fastest rate of that mode instead. Section 3.1 shows how to choose the sensor mode yourself.

---

### 3.1. High Frame Rate, Lower Resolution

**Goal:** Run the camera at 120 fps by asking Picamera2 for the camera's fast sensor mode, and check the frame rate that you really get.

**Script:** [`3.1_high_fps_low_res.py`](3.1_high_fps_low_res.py)

Run it with `python 3.1_high_fps_low_res.py`. It opens a window, so run it on the Pi desktop. To stop it, click the window and press `q`.

**Explanation:** The image sensor in the HQ Camera (Sony IMX477) can work in several sensor modes. Each mode reads the sensor in a different way and has its own maximum frame rate. Since a change to the camera driver in October 2025 (in both 2026 Raspberry Pi OS images), the sensor has 5 mode sizes, and it can read each of them with 12, 10 or 8 bits per pixel:

| Sensor mode | Part of the sensor that is used | Maximum frame rate at 12 / 10 / 8 bits per pixel |
|---|---|---|
| 4056x3040 | the whole sensor | about 12 / 14 / 17 fps |
| 4056x2160 | the full width, only the central 2160 rows | about 16 / 20 / 24 fps |
| 2028x1520 | the whole sensor | about 45 / 54 / 66 fps |
| 2028x1080 | the full width, only the central 2160 rows | about 63 / 75 / 92 fps |
| 1332x990 | only the central 2664x1980 pixels | about 102 / 120 / 148 fps |

These values were calculated from the source code of the driver and of libcamera. They were not measured. Older Raspberry Pi OS images (kernel 6.12.47 or earlier) have only four modes: 4056x3040 at about 10 fps, 2028x1520 at about 40 fps, 2028x1080 at about 50 fps and 1332x990 with 10 bits at about 120 fps. To see the kernel version of your Pi, type `uname -r`. The list that `print(picam2.sensor_modes)` prints on your Pi is always correct. Script 2 prints this list. In your own script, put this line straight after `picam2 = Picamera2()`, before `configure()`.

Picamera2 uses 12 bits per pixel unless you ask for another bit depth. Only the 1332x990 modes can reach 120 fps, and only with 10 or 8 bits per pixel. If you give only the 1332x990 size, Picamera2 picks the 12-bit 1332x990 mode, and the camera runs at about 102 fps. A small output size, such as 1280x720, does not select a 1332x990 mode at all. As section 3 explains, Picamera2 then picks the 12-bit mode with the closest size and shape. For 1280x720 this is the 2028x1080 mode, so the camera cannot go above about 63 fps. So you must give both the size and the bit depth.

The correct way is to ask for the sensor mode with the `sensor` parameter, as the script does:

```python
config = picam2.create_video_configuration(
    main={"size": target_size, "format": "RGB888"},
    sensor=sensor_mode,
    controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)}
)
```

In the script, `sensor_mode` is `{"output_size": (1332, 990), "bit_depth": 10}`, `target_size` is `(1332, 990)`, and `frame_duration_us` is 8333 microseconds, which is about 1/120 of a second. The main stream has the same size and shape as the sensor mode, so the whole 1332x990 sensor image is used. A main stream with a different shape would be cropped further to fit that shape.

After `configure()`, the script prints `picam2.camera_configuration()["sensor"]`, which is the sensor mode that Picamera2 really chose. It should be `{'bit_depth': 10, 'output_size': (1332, 990)}`. While the script runs, it prints four numbers every second:

- The frame rate that the camera is set to. It comes from the `FrameDuration` value in the metadata of the latest frame. FrameDuration is the frame time that the camera was set to, in microseconds. The script reads the frame and its metadata together with `picam2.capture_request()`.
- The rate of frames received. The number of frames received shows how many frames really reached Picamera2 in the last second. The script reads this number from `picam2.frames`. It should be about 120.
- The processing loop rate. This is the number of frames that the loop took and could process in the last second. It should also be about 120.
- The window rate. This is the number of frames that the loop showed on the screen in the last second, about 30.

**Why the window shows only every 4th frame, at half size.** Drawing a large image in a window on the Pi desktop is a lot of work for the whole Pi, not only for your script: the desktop must draw the window too. While it does this, the camera system cannot keep up and the camera drops frames. If the script showed every frame at full size, only about 40 frames per second would arrive, even though the camera is set to 120 fps. This happens even when the window is drawn by a different program, so it is not caused by Python. So the script shows a half-size copy of every 4th frame. That is about 30 fps, which is smooth enough for your eyes, and then all 120 frames per second arrive. The loop can still get every frame, so you can put your own processing (for example finding the string) where the comment in the loop says. Then compare the processing loop rate with "Frames received". "Frames received" counts every frame that arrives from the camera, also the frames that the loop is too slow to take. If the processing loop rate is clearly below "Frames received", the loop misses frames, because your processing and the window together take too long. Then make your processing faster, for example by processing only a region of interest, or show the window less often or at a smaller size. Section 3.2 shows how to get a reading from every frame in this way.

**What changes in the 120 fps mode:**
- The field of view is narrower. The mode uses only the central 2664x1980 pixels of the sensor, which is about two-thirds of its width and two-thirds of its height. With the 6 mm lens, the view is about 38° wide instead of about 55°. The camera sees a smaller part of the scene, so check that the whole string stays inside the image, or move the camera further away.
- Each frame lasts only 8.3 ms, so the exposure time cannot be longer than 8.3 ms. The camera then raises the gain to keep the image bright, so the image can be darker or noisier than at 30 fps, and you may need more light.

**Can you still capture at 1080p?** Not at 120 fps. The HQ Camera cannot give 1920x1080 images at 120 fps. The fastest mode with 1080 rows, 2028x1080, stops at about 63 fps with 12 bits (about 75 fps with 10 bits and about 92 fps with 8 bits). If you need 120 fps, you must use a 1332x990 mode and accept its narrower field of view.

---

### 3.2. One Reading from Every Frame at 120 fps

**Goal:** Get one measurement from every camera frame at 120 fps, here the angle of a yellow string, and check that no frame is skipped.

**Script:** [`3.2_reading_per_frame.py`](3.2_reading_per_frame.py)

Run it with `python 3.2_reading_per_frame.py`. It opens a window, so run it on the Pi desktop. To stop it, click the window and press `q`, or press Ctrl+C in the terminal. It saves every reading in `readings.csv`: one row per frame, with the time since the first frame in seconds (`time_s`, measured by the camera) and the angle in degrees (`angle_deg`, empty when no string was found). Like the images and videos of the other scripts, the file is replaced every time you run the script, so copy or rename it if you want to keep it.

**Explanation:** The script uses the 120 fps setting of script 3.1. For every frame it finds the string with the colour method of Week 8: it converts the image to HSV, keeps the pixels in a yellow colour range, cleans the result, and fits a straight line to the largest yellow shape. The angle of this line from vertical is the reading. 0° means that the string hangs straight down, and a positive angle means that its bottom end is to the right. Week 8 explains each step and shows how to choose the colour range for your own string and light. Here, look at how the script keeps up with 120 fps.

**Each frame has 8.3 ms.** At 120 fps a new frame arrives every 8.3 ms. If the processing of every frame takes longer, the loop is not ready for the next frame, and Picamera2 keeps only the newest frame for it, so the frames in between are lost. Picamera2 does not warn you. On a Raspberry Pi 5, the colour method takes about 8 to 9 ms for the whole 1332x990 frame. Together with the time to get each frame from the camera, this is too slow: the loop then gets only about 80 of the 120 frames per second. So the script processes only a region of interest (ROI), a strip that contains the string. By default this is the central third of the image (`roi` near the top of the script). It takes about 3 ms. Change `roi` so that it just contains your string, with some space for its movement. Making the image smaller with `cv2.resize()` first also works.

**Check for skipped frames.** Every frame has a `SensorTimestamp`, the time at which the camera took it. If the time from one frame to the next is clearly longer than one frame time, frames were skipped. Every second the script prints the frames received from the camera ("Frames received", which stays near 120 even when your loop is too slow), the readings that your loop made ("Readings"), the frames skipped in that second ("Skipped"), the median time to process one frame ("Processing"), and the latest angle. At the end it prints the total number of skipped frames. If "Skipped" is large, look at "Frames received". If it stays near 120 but "Readings" is clearly lower, your loop is too slow. Then make the ROI smaller, or do less work per frame. If "Frames received" is also clearly below 120, some frames never reached Picamera2. Your loop could not take these frames, so a smaller ROI usually does not help with them. This can happen when the Pi draws windows on the desktop (see section 3.1). Then turn off the window (see below) and close other programs. Both can happen at the same time: if "Readings" is also clearly lower than "Frames received", your loop is too slow as well.

**The window costs frames too.** A frame that also updates the window takes longer than the others. So the window here is smaller and slower than in script 3.1: one third of the size, every 6th frame (20 fps). With the window, a few frames per second can still be skipped (in our tests usually up to about 6, sometimes more). When every frame counts, for example when you record a measurement, set `show_window = False` near the top of the script. Then the script opens no window, so it also works in Remote shell, and almost no frames are skipped (3 in 8 seconds in our test). Stop it with Ctrl+C. The `time_s` column comes from the camera, so it stays correct even if a frame is skipped.

**Does it help to use C++ instead of Python, or more threads?** Not for this kind of processing. The `cv2` functions run the same compiled C++ code of OpenCV that a C++ program would use. Python only adds about 2 microseconds to each `cv2` call. The script makes about 8 `cv2` calls per frame, so Python adds much less than 0.1 ms to the 3 ms that the processing of the ROI takes. Running the processing in several Python threads did not help either: on the Pi 5 it gave at most about 105 readings per second, and the camera itself dropped frames. Python is slow only when you go through the pixels in a Python loop, for example `for x in range(width):`. Use OpenCV or NumPy functions on the whole image or ROI instead.

**Heat.** At 120 fps the Pi works hard. A Pi without a fan can become hot (above 80 °C) after a few minutes and then slows itself down, so that more frames are skipped. If "Skipped" grows after some minutes, check the temperature with `vcgencmd measure_temp`.

---

### 4. Two Streams at Once (Main and Lores)

**Goal:** Get two images of different sizes from the same camera frame: a main image for display, and a small image for fast processing with OpenCV.

**Script:** [`4_dual_stream_preview.py`](4_dual_stream_preview.py)

Run it with `python 4_dual_stream_preview.py`. It opens a window, so run it on the Pi desktop. To stop it, click the window and press `q`.

**Explanation:** Picamera2 can give more than one image from each camera frame. Each image is called a stream. The main stream is the normal image. The lores stream is a second, low-resolution copy of the same frame ("lores" is short for "low resolution"). In this script the main stream is 1280x720 pixels and the lores stream is 320x180 pixels. Both have the same 16:9 shape, so they show the same view without distortion. The script reads the real lores size from `picam2.camera_configuration()["lores"]["size"]`, and it takes both images from the same frame with `picam2.capture_arrays(["main", "lores"])`. The lores stream is in YUV420 format, and the script uses only its Y plane, which is the brightness (grey-scale) image. So the script shows the main image in colour and the lores image in grey. Both images are in one window: the main image on the left, and the lores image at its real size on the right. A small image is quick to process, so you can analyse the lores stream in real time while you show or record the main stream.

---

### 5. Record H.264 Video to MP4

**Goal:** Save an encoded video to a file, and turn off automatic exposure and automatic white balance so that the brightness and colour stay the same during the recording.

**Script:** [`5_record_video_h264.py`](5_record_video_h264.py)

Run it with `python 5_record_video_h264.py`. It opens no window, so it also works in Remote shell. It saves the video as `clip.mp4`.

**Explanation:** This script shows how to record a video to a file using an H.264 encoder. It also shows how to lock the automatic exposure and automatic white balance, so that the video has the same brightness and colour from start to end. This is important when you want to measure something in the video and repeat the measurement later.

---

### 5.1. Record a 120 fps clip

**Goal:** Record a short video at 120 fps, with the fast sensor mode from section 3.1.

**Script:** [`5.1_record_120fps.py`](5.1_record_120fps.py)

Run it with `python 5.1_record_120fps.py`. It opens no window, so it also works in Remote shell. It records 5 seconds and saves the video as `clip_120fps.mp4`.

**Explanation:** The script uses the same sensor setting as script 3.1: the 1332x990 sensor mode with 10 bits per pixel, and a frame time of 8333 microseconds (about 1/120 of a second). It prints the sensor mode in use, so you can check it. Like script 5, it turns off the automatic exposure and the automatic white balance before it starts to record.

- The Raspberry Pi 5 has no hardware video encoder (the CM5 has none either), so the processor encodes the video. This is a lot of work at 120 fps. At the full 1332x990 size, the processor cannot keep up, and a 5-second clip has only about 520 frames instead of 600. So the script records at 888x660 (`video_size` near the top of the script). This is 2/3 of the sensor image in each direction, with the same shape and the same field of view, and the clip keeps all its frames.
- The script writes the file with `PyavOutput`, so that every frame keeps the camera's own time (its timestamp). Script 5 uses `FfmpegOutput` instead, which gives each frame the time at which it reached FFmpeg.

**Check the clip.** A 5-second clip at 120 fps should hold about 600 frames. At the end, the script prints this command, which counts the frames in the file:

```bash
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 clip_120fps.mp4
```

If the number is clearly smaller than 600, the processor did not encode every frame in time. This can happen if other programs are busy at the same time, or if the Pi is hot. Then make the video smaller: near the top of the script, change `video_size` to a smaller size, for example `(592, 440)`, and record again. Keep the height a multiple of 4. With other heights, the colours in the video are shifted. The sensor still runs at 120 fps, and the processor has less work to do.

---

## D. Copy a file to your laptop

Raspberry Pi Connect cannot copy files between the Pi and your laptop. Use one of these ways:

1. **USB stick.** Plug the USB stick into the Pi. Copy the file to the stick with the File Manager. Click the eject icon before you remove the stick.
2. **UCL OneDrive.** Open the web browser on the Pi, sign in to your UCL OneDrive, and upload the file. Then download it on your laptop. When you have finished, sign out of OneDrive in the browser on the Pi.
3. **Over the network, only if your laptop can reach the Pi.** Ask a demonstrator before you turn on SSH on a lab kit. Every kit has the same login. The Raspberry Pi 5 setup guide gives the user name and the password in section A. If you use a CM5 kit, ask a demonstrator for them. Before you turn on SSH or VNC, change the password with `passwd` and tell your team and a demonstrator the new one. Then:
   - Turn on SSH: run `sudo raspi-config`, choose **Interface Options**, then **SSH**, and answer **Yes**.
   - Find the address of the Pi: run `hostname -I` on the Pi. Use the first address that it prints, for example `10.1.2.3`.
   - On your laptop, in a terminal, run this command. Replace `<address>` with the address of the Pi, and `clip.mp4` with the name of your file. The `.` at the end means "copy it into the current folder of the laptop".

     ```bash
     scp comp0219@<address>:~/COMP0219_Tutorials/tutorials/week06_picamera2_tutorial/clip.mp4 .
     ```

     The first time, `scp` asks whether you want to continue connecting. Type `yes`. Then type the password of the Pi.
