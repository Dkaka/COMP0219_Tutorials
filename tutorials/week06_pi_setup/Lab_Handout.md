# Week 6: Raspberry Pi & Camera Setup

COMP0219 · Start of Part 2 (contact-less, vision-based airspeed sensor)

**Work in your team of 3.** Bring a laptop and your UCL login. Today you first set up the Pi with
this checklist, and then you continue with the Picamera2 tutorial in the same session.

> **Using the Raspberry Pi Development Kit for Compute Module 5 instead of a Raspberry Pi 5?**
> Use [`CM5_Lab_Handout.md`](CM5_Lab_Handout.md) and the full guide
> [`CM5_Dev_Kit_Guide.md`](CM5_Dev_Kit_Guide.md) instead of this checklist.

> **Today's goal:** assemble your Raspberry Pi 5 with the HQ camera and 6 mm lens, get it on the
> network, set up **Raspberry Pi Connect** so you can reach it remotely from your laptop, and
> capture a first sharp image. Then the coding starts in the Picamera2 tutorial.

> **Before you touch anything:** handle boards by the edges on the anti-static mat, and **always
> power the Pi off before connecting or disconnecting the camera cable**. Connecting the camera while the
> power is on is the most common way to damage it.

---

## A. Check your kit

- **Raspberry Pi 5 (8 GB)** — active cooler fitted
- **Pre-flashed microSD card** — Raspberry Pi OS (64-bit); login `comp0219` / `comp0219`
- **27 W USB-C power supply** — use this one; weak supplies cause crashes
- **HQ Camera** — Sony IMX477, 12.3 MP (4056 × 3040), CS-mount
- **6 mm wide-angle lens** — CS-mount, about 55° horizontal view, manual focus + aperture
- **Camera FPC cable** — 22-pin ↔ 15-pin, printed **CAMERA** (the Pi 5 adapter cable)
- **Micro-HDMI → HDMI, keyboard, mouse** — at the monitor station

Other lenses and cameras (for example the 16 mm telephoto lens, or the global-shutter camera) are
available later. Ask a demonstrator.

---

## B. Assemble the camera

1. **Power off** the Pi. Keep it unplugged until step 7.
2. **Camera cable to the HQ camera.** The camera may still have its original cable, which has the
   wide 15-pin plug at both ends. This cable does not fit the Pi 5. Remove it (pull the dark flap
   out first) and give it to a demonstrator. Then connect the wide **15-pin** end of the kit
   cable: pull the dark flap out until it stops, insert the cable straight with the metal contacts
   facing away from the flap, and push the flap back in.
3. **Camera cable to the Pi 5.** The Pi 5 has two camera connectors (CAM/DISP 0 and CAM/DISP 1).
   Use **CAM/DISP 0**. Connect the narrow **22-pin** end in the same way: pull the dark flap out
   until it stops, insert the cable straight with the metal contacts facing away from the flap,
   and push the flap back in. The Active Cooler is close to the connector. Work slowly and do not
   bend the cable sharply.
4. **Fit the lens.** The HQ camera and the 6 mm lens are both CS-mount. If a thin threaded ring
   (the C-CS adapter) is fitted, unscrew it and put it in the kit bag. You need it later only for
   the 16 mm C-mount lens. Screw the lens clockwise all the way in. Do not force the thread. You
   will focus it in section E.
5. **Insert the microSD card** (underside slot).
6. Connect **keyboard, mouse, micro-HDMI → monitor**.
7. **Power last:** plug in the 27 W USB-C supply. The Pi boots.

---

## C. First boot & network

1. The Pi logs in automatically. If asked, use `comp0219` / `comp0219`.
2. Connect the Pi to **eduroam** with your own UCL account. Click the network icon at the top
   right of the screen, then click **eduroam**. In the window "Wi-Fi Network Authentication
   Required", set these values and then click **Connect**:
   - Wi-Fi security: WPA & WPA2 Enterprise (already selected).
   - Authentication: change "Tunneled TLS" to **Protected EAP (PEAP)**.
   - Anonymous identity: leave it empty. Domain: `ucl.ac.uk`.
   - CA certificate: leave "(None)", and tick **No CA certificate is required**.
   - PEAP version: Automatic. Inner authentication: **MSCHAPv2**.
   - Username: your UCL user ID followed by `@ucl.ac.uk`, for example `abcd123@ucl.ac.uk`.
     Password: your UCL password.

   These values come from UCL's guide for Linux:
   <https://www.ucl.ac.uk/isd/how-to/connecting-to-eduroam-wi-fi-linux>. Open it on your laptop or
   phone if you need it, because the Pi has no internet yet.
3. Open a terminal and update:

   ```bash
   sudo apt update
   ```

---

## D. Remote access — Raspberry Pi Connect

Raspberry Pi Connect gives you the Pi's **screen** and a **remote shell** in any browser, with no
monitor attached. In the project, you use it to start and stop your sensing software. The full
package is already installed on the lab image.

1. **Make one team account.** On your laptop, go to <https://connect.raspberrypi.com>. Make one
   new Raspberry Pi ID for your team, and use it only for this course. Give it a new password that
   you do not use anywhere else. Do not use a team member's personal account. Only the Raspberry
   Pi ID that owns the device can open Screen sharing and Remote shell, so the whole team uses
   this one ID.
2. **Turn Connect on and sign in.** Run these two commands in a terminal on the Pi:

   ```bash
   rpi-connect on
   rpi-connect signin
   ```

   The second command prints a link, for example `https://connect.raspberrypi.com/verify/ABCD-1234`.
   Open this link in the browser on your laptop and sign in with the team's Raspberry Pi ID. In
   this way the password is never typed on the Pi. If the page asks for a device name, use your
   team name, for example `team-07`. (You can also turn Connect on with the Connect icon in the
   taskbar: *Turn On Raspberry Pi Connect*, then *Sign in*. But then you type the password in the
   browser on the Pi, so use the terminal way if you can.)
3. **Test from your laptop** — at connect.raspberrypi.com, select the device and open
   **Screen sharing** and **Remote shell**. A grey **Screen sharing** badge under the device name is
   normal: it means that screen sharing is available.

> **Pi Connect does not offer screen sharing?** Run `rpi-connect doctor` and read which check
> fails. If auto login is off, run `sudo raspi-config`, choose **System Options**, then
> **Auto Login**, and answer **Yes** to the question about logging in to the desktop
> automatically. If the desktop does not use Wayland, run `sudo raspi-config`, choose
> **Advanced Options**, then **Wayland**, and choose **labwc**. Restart the Pi after each change.
> If screen sharing is still not offered, show the output of `rpi-connect doctor` to a
> demonstrator.
>
> Docs: <https://www.raspberrypi.com/documentation/services/connect.html>

---

## E. First camera capture (sanity check)

Do this **before** you write any Python. The camera apps (`rpicam-apps`) are already installed,
and they are the simplest test of the camera. If they fail, the Picamera2 scripts will fail too.

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

**Focusing the 6 mm lens** (with the preview from command 1 running). Hold the camera with the
lens pointing away from you. The directions below (clockwise, anticlockwise) are as you see them
in this position.

1. **Aperture:** hold the front ring of the lens still and turn the middle ring. Clockwise makes
   the image darker, anticlockwise makes it brighter. When the brightness is good, tighten the
   small screw on the side of the lens.
2. **Focus:** tighten the screw of the inner ring marked NEAR and FAR. Then turn the two outer
   rings together until the image is sharp. Clockwise focuses on nearer objects, anticlockwise on
   more distant objects. This can take four or five whole turns.
3. Do not touch the back focus adjustment ring. For the 6 mm lens it stays screwed in fully. If it
   is loose, ask a demonstrator.
4. When the image is sharp, press `Ctrl+C` to stop the preview, and then run commands 2 and 3.
   Only one program can use the camera at a time.

Open `week6_test.jpg` and check that it is sharp. To check the video, double-click
`week6_test.mp4` in the file manager.

Confirm detection any time with:

```bash
rpicam-hello --list-cameras   # should list imx477
```

---

## F. Check before you continue

- [ ] The camera is detected: `rpicam-hello --list-cameras` lists `imx477`.
- [ ] You have captured a **sharp** `week6_test.jpg` and looked at it.
- [ ] Pi Connect **Screen sharing** works from your laptop.
- [ ] Pi Connect **Remote shell** works from your laptop.

**Next:** the Picamera2 tutorial, [`../week06_picamera2_tutorial/README.md`](../week06_picamera2_tutorial/README.md)
(folder `tutorials/week06_picamera2_tutorial`). If the session ends before you start it, shut down with
`sudo poweroff`, and return your kit as instructed.

---

## Trouble? Quick fixes

- **`No cameras available`:** power off the Pi. At both cable ends, pull the dark flap out until
  it stops and take the cable out. Then insert the cable straight with the metal contacts facing
  away from the flap, and push the flap back in. Use CAM/DISP 0. If it still fails, try a spare
  cable.
- **Black preview:** remove the lens cap and open the aperture.
- **eduroam does not connect:** there is no "accept the certificate" button on the Raspberry Pi.
  Check the values in section C. If you saved wrong values, clicking eduroam again only asks for
  the password. Delete the saved connection with `sudo nmcli connection delete eduroam` and
  connect again from the network icon. If this command says that there is no such connection, run
  `nmcli connection show` to see the name of the connection. If it still does not connect, ask a
  demonstrator.
- **The Pi shows an under-voltage warning or restarts by itself:** use the **27 W** USB-C power
  supply from the kit.
- **No picture on the monitor:** use the micro-HDMI port nearest the USB-C port, and reseat the
  cable.

---

**After the Picamera2 tutorial:** the next tutorials are calibration (Week 7), and then colour
segmentation and line fitting (Week 8). They lead to your contact-less airspeed sensor (demo on
11 December). · COMP0219, UCL.
