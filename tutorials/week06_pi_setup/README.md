# Week 6: Raspberry Pi and Camera Setup

This session starts **Part 2** of COMP0219 — the *contact-less* (vision-based) airspeed sensor.
You will collect your kit, assemble the Raspberry Pi 5 with the High Quality (HQ) camera and
6 mm wide-angle lens, get the Pi on the network, set up **Raspberry Pi Connect** for remote
access, and capture a first image to confirm everything works.

> **Goal of this guide:** a working Pi and camera that you can reach from your laptop. After this
> guide, continue in the same session with the Picamera2 tutorial, where the coding starts:
> [`../week06_picamera2_tutorial/README.md`](../week06_picamera2_tutorial/README.md).

> **Does your team have the Raspberry Pi Development Kit for Compute Module 5 instead of a
> Raspberry Pi 5?** Some teams use this kit. Several steps are different: the operating system
> is stored on the module's built-in eMMC instead of a microSD card, the camera may need two lines
> in `config.txt`, and a mounting plate, if you have one, holds only the camera. Follow
> [`CM5_Dev_Kit_Guide.md`](CM5_Dev_Kit_Guide.md) and the checklist
> [`CM5_Lab_Handout.md`](CM5_Lab_Handout.md) instead of this page.

---

## A. What you receive (per team of 3)

- **Raspberry Pi 5 (8 GB)** — with active cooler fitted
- **Pre-flashed microSD card** — Raspberry Pi OS (64-bit); username **and** password are both `comp0219`
- **Official 27 W USB-C power supply** — use this one, because a weaker power supply can cause crashes
- **Raspberry Pi HQ Camera** — Sony IMX477, 12.3 MP, 4056 × 3040, CS-mount
- **6 mm wide-angle CS-mount lens** — about 55° horizontal × 45° vertical field of view on the HQ
  camera, with manual focus and aperture. The 120 fps mode uses only the centre of the sensor, so
  its view is narrower (about 38° wide).
- **Pi 5 camera FPC cable** — 22-pin (0.5 mm) ↔ 15-pin (1 mm), printed **CAMERA**
- **Micro-HDMI → HDMI cable, USB keyboard + mouse** — at the lab monitor station

Other lenses and cameras (for example the 16 mm telephoto lens, or the global-shutter camera) are
available later. Ask a demonstrator. For now everyone starts with the **HQ camera + 6 mm lens**.

> Handle the boards by their edges and work on the anti-static mat. The Pi must be **powered off**
> whenever you connect or disconnect the camera cable.

---

## B. Assemble the camera

1. **Power off** the Pi (it should be unpowered while you do this).
2. **Camera cable → HQ camera.** The camera may still have its original cable, which has the wide
   15-pin plug at both ends. This cable does not fit the Pi 5. Remove it and give it to a
   demonstrator: pull the dark flap of the camera's connector out until it stops, and slide the
   old cable out. Then fit the kit cable at the camera. Pull the dark flap out until it stops,
   insert the wide **15-pin** end of the cable straight with the metal contacts facing away from
   the flap, and push the flap back in. The cable should be straight and fully inserted, not at
   an angle.
3. **Camera cable → Pi 5.** The Pi 5 has two connectors labelled **CAM/DISP 0** and
   **CAM/DISP 1**. Both work, but in this course we always use **CAM/DISP 0**. Pull the dark flap
   out until it stops, insert the **22-pin** end of the cable straight with the metal contacts
   facing away from the flap, and push the flap back in. The Active Cooler is close to the
   connector. Work slowly and do not bend the cable sharply.
   *The Pi 5 uses the smaller 22-pin connector. That is why the cable has a different plug at
   each end (22-pin at the Pi, 15-pin at the camera).*
4. **Fit the lens.** The HQ camera is **CS-mount**. The 6 mm lens is also CS-mount, so it screws
   **directly** onto the camera. If a thin threaded ring (the C-CS adapter) is fitted, unscrew it
   and put it in the kit bag. You need it later only for the 16 mm C-mount lens.
5. **Screw the lens clockwise all the way in.** Do not force the threads. You will focus it in
   section E. Do not touch the **back focus adjustment ring** on the camera body (see section E).
6. **Insert the microSD card** (pre-flashed) into the slot on the underside.
7. Connect **keyboard, mouse, micro-HDMI → monitor**.
8. **Power last.** Plug in the 27 W USB-C supply. The Pi boots.

---

## C. First boot & network

The Pi logs in to the desktop automatically. If prompted, the login is `comp0219` / `comp0219`.

Connect the Pi to **eduroam** with your own UCL account.

1. Click the network icon at the top right of the screen, then click **eduroam**. A window called
   "Wi-Fi Network Authentication Required" opens.
2. Fill it in like this:
   - Wi-Fi security: WPA & WPA2 Enterprise (already selected)
   - Authentication: change "Tunneled TLS" to **Protected EAP (PEAP)**
   - Anonymous identity: leave it empty
   - Domain: `ucl.ac.uk`
   - CA certificate: leave "(None)", and tick **No CA certificate is required**
   - PEAP version: Automatic
   - Inner authentication: **MSCHAPv2**
   - Username: your UCL user ID followed by `@ucl.ac.uk`, for example `abcd123@ucl.ac.uk`
   - Password: your UCL password
3. Click **Connect**.

These values come from UCL's guide for Linux:
<https://www.ucl.ac.uk/isd/how-to/connecting-to-eduroam-wi-fi-linux>. Open that page on your
laptop or phone if you need it, because the Pi has no internet yet. Its menu steps are for a
different Linux desktop, but the values are the same.

When the Pi is connected, open a terminal and update the package lists:

```bash
sudo apt update
```

Some later steps depend on the version of Raspberry Pi OS. To see which version your Pi runs, type
`grep VERSION_CODENAME /etc/os-release`. It prints a line such as `VERSION_CODENAME=trixie`.

---

## D. Remote access with Raspberry Pi Connect

Raspberry Pi Connect lets you see the Pi's **screen** and use a **remote shell** (a terminal)
from a web browser on your laptop. Once it is set up, you do not need a monitor. You will use it
in the project to start and stop your sensing software remotely. The full `rpi-connect` package
(screen sharing and remote shell) is already installed on the lab image.

1. **Make one Raspberry Pi ID for your team.** A Raspberry Pi ID is an account on Raspberry Pi's
   websites. Make one new Raspberry Pi ID for your team, and use it only for this course. Give it
   a new password that you do not use anywhere else. Do not use a team member's personal account.
   Only the Raspberry Pi ID that owns the device can open Screen sharing and Remote shell, so the
   whole team uses this one ID. Create it on your laptop at <https://connect.raspberrypi.com>.
2. **Turn Connect on and sign in, on the Pi.** Run `rpi-connect on` and then `rpi-connect signin`
   in a terminal on the Pi:

   ```bash
   rpi-connect on
   rpi-connect signin
   ```

   The second command prints a link, for example
   `https://connect.raspberrypi.com/verify/ABCD-1234`. Open this link in the browser on your
   laptop and sign in with the team's Raspberry Pi ID. In this way the password is never typed on
   the Pi. If the page asks for a device name, use your team name, for example `team-07`.

   You can also use the **Connect icon** in the taskbar: *Turn on Raspberry Pi Connect*, then
   *Sign in*. This way opens the browser on the Pi, so you type the password on the Pi. If the
   Pi's browser offers to save the password, choose **Never**.
3. **Check the device name.** At <https://connect.raspberrypi.com>, the device should have your
   team name. If it does not, change the name there: Devices, then your device, then Settings.
4. **Test from your laptop.** At <https://connect.raspberrypi.com>, select your device and open
   both **Screen sharing** and **Remote shell**. You should see the Pi desktop and get a working
   terminal.

> **Where to run the scripts.** Programs that open a window must run in a terminal on the Pi
> desktop: either on the lab monitor, or inside Raspberry Pi Connect **Screen sharing**. This
> applies to `rpicam-hello` and to scripts 1, 3, 3.1, 3.2 and 4 of the Picamera2 tutorial. Raspberry Pi
> Connect **Remote shell** has no desktop, so these programs cannot open a window there.
> `rpicam-hello` then shows no picture, and the Python scripts stop with an error. `rpicam-still`,
> `rpicam-vid` and scripts 2, 5 and 5.1 do not need a window, so they also work in Remote shell. Script 3.2 also works there if you turn its window off (section 3.2 of the Picamera2 tutorial). A program that you start in Remote shell stops when you close the browser tab.

> **Screen sharing needs** a Wayland desktop and a desktop session that logs in automatically.
> Both are the default on Raspberry Pi OS with desktop. A grey **Screen sharing** badge under the
> device name is normal: it means that screen sharing is available. If Screen sharing is not
> offered, see Troubleshooting.
>
> Docs: <https://www.raspberrypi.com/documentation/services/connect.html>

At the end of the course, run `rpi-connect signout` on the Pi, and delete the device at
connect.raspberrypi.com (Devices, then your device, then Settings, then Delete device).

---

## E. First camera capture (sanity check)

Do this **before** you write any Python. If these command-line tools fail, the Picamera2 scripts
will also fail. The camera apps (`rpicam-apps`) are already installed.

Run `rpicam-hello` and the Python scripts in a terminal on the Pi desktop (lab monitor or Pi
Connect Screen sharing), not in Remote shell, because Remote shell has no desktop.

```bash
# 1) Live preview until you press Ctrl+C. Use it to focus the lens.
rpicam-hello -t 0

# 2) Capture a still image
rpicam-still -o week6_test.jpg

# 3) Record a 10-second video at 1920 x 1080 pixels and 30 frames per second, as an MP4 file
rpicam-vid -t 10000 -o week6_test.mp4 --width 1920 --height 1080 --framerate 30
```

Command 1 keeps the preview open until you press `Ctrl+C` in its terminal. Focus the lens while
the preview is open (see below). Then press `Ctrl+C` before you run command 2, because only one
program can use the camera at a time.

**Focusing the 6 mm lens.** Raspberry Pi's guide for this lens describes the following method:

1. **Aperture.** Hold the camera with the lens pointing away from you. Hold the front ring of the
   lens still and turn the middle ring. Clockwise closes the aperture and makes the image darker;
   anticlockwise opens it and makes the image brighter. When the brightness is good, tighten the
   small screw on the side of the lens to lock the aperture.
2. **Focus.** Lock the inner focus ring (marked NEAR and FAR) by tightening its screw. Then hold
   the two outer rings of the lens together and turn them. Clockwise focuses on nearer objects;
   anticlockwise focuses on more distant objects. From the fully screwed-in position, it can take
   four or five whole turns before the image becomes sharp.
3. You may need to adjust the aperture again after focusing.

Do not touch the back focus adjustment ring. For the 6 mm lens it stays screwed in fully. If it
is loose, ask a demonstrator.

To check at any time that the camera is detected, run:

```bash
rpicam-hello --list-cameras
```

You should now have `week6_test.jpg` and `week6_test.mp4` in the folder where you ran the
commands (your home folder, if you opened a new terminal). Open the still image and check that it
is sharp. Double-click the video in the file manager to check it.

---

## F. Set up the coding environment

You need Python later in this session for the Picamera2 tutorial. Set up the environment now. It is the
same `comp0219` environment for all Part 2 tutorials, so you do this only once. Follow section B
of the Picamera2 tutorial:
[Set up the Python environment (on the Raspberry Pi)](../week06_picamera2_tutorial/README.md#b-set-up-the-python-environment-on-the-raspberry-pi).
Then come back here for section G.

---

## G. Check before you continue

- [ ] Camera detected (`rpicam-hello --list-cameras` lists `imx477`).
- [ ] A sharp `week6_test.jpg` captured.
- [ ] Raspberry Pi Connect **Screen sharing** *and* **Remote shell** both work from your laptop.

**Next:** the Picamera2 tutorial, [`../week06_picamera2_tutorial/README.md`](../week06_picamera2_tutorial/README.md).
If the session ends before you start it, shut down with `sudo poweroff`, and return your kit as
instructed.

The next tutorials are calibration (Week 7) and segmentation and line fitting (Week 8). They lead to the contact-less airspeed project.

---

## Troubleshooting

### `No cameras available`
- Power off, then reseat **both** ends of the FPC cable. At both ends, the metal contacts must face away from the flap. Check that you used **CAM/DISP 0**.
- Raspberry Pi OS detects the camera automatically. There is no "enable camera" step.

### Preview is black
- Remove the lens cap.
- Open the aperture: hold the front ring of the lens and turn the middle ring anticlockwise. If the aperture is locked, loosen the small screw on the side of the lens first.
- A lens that is out of focus gives a blurred picture, not a black one, so do not change the focus to fix a black preview. Do not adjust the back focus adjustment ring. If it is loose, ask a demonstrator.

### Pi Connect does not offer screen sharing
- Make sure the full `rpi-connect` package is installed (not `rpi-connect-lite`).
- Run `rpi-connect doctor` and read which check fails. If auto login is off, run `sudo raspi-config`, choose **System Options**, then **Auto Login**, and answer **Yes** to the question about logging in to the desktop automatically. If the desktop does not use Wayland, run `sudo raspi-config`, choose **Advanced Options**, then **Wayland**, and choose **labwc**. Restart the Pi after each change. If screen sharing is still not offered, show the output of `rpi-connect doctor` to a demonstrator.

### You cannot sign in to Connect
- Run `rpi-connect signin` again.
- Run `rpi-connect doctor`. Every line should show a tick. If a network line fails on eduroam, tell a demonstrator.

### eduroam does not connect
- There is no "accept the certificate" button on the Raspberry Pi. Check the values in the eduroam list in section C. If you saved wrong values, clicking eduroam again only asks for the password. Delete the saved connection with `sudo nmcli connection delete eduroam` and connect again from the network icon. If this command says that there is no such connection, run `nmcli connection show` to see the name of the connection. If it still does not connect, ask a demonstrator.

### No display on the monitor
- Use the micro-HDMI port **nearest the USB-C** and reseat the cable.

### The Pi restarts by itself or shows an under-voltage warning
- Use the supplied **27 W** USB-C power supply.
