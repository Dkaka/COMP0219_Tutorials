# Week 6 (CM5 version): Compute Module 5 Development Kit and Camera Setup

COMP0219 · Start of Part 2 (contact-less, vision-based airspeed sensor)

**Work in your team of 3.** The session takes about 2 hours, and it has two parts: first this
setup, then the Picamera2 tutorial. Bring a laptop and your UCL login.

This checklist is for teams with the **Raspberry Pi Development Kit for Compute Module 5**.
Teams with a standard Raspberry Pi 5 use [`Lab_Handout.md`](Lab_Handout.md). The full
explanations for every step are in [`CM5_Dev_Kit_Guide.md`](CM5_Dev_Kit_Guide.md).

> **Today's goal:** assemble the development kit, attach the HQ Camera with the 6 mm lens,
> connect the kit to the network, set up **Raspberry Pi Connect** so that you can reach it from
> your laptop, and capture a first sharp image. Coding starts in the Picamera2 tutorial.

> **Before you touch anything:** handle the boards by their edges on the anti-static mat. The kit
> must always be **unplugged** when you connect or disconnect the camera cable. Connecting the
> camera while the power is on is the most common way to damage it.

> **The operating system is already installed.** The teaching team has installed Raspberry Pi OS
> on the CM5's built-in storage (eMMC). You do not need a microSD card, and you do not need to
> install the operating system. If your CM5 does not start, ask a demonstrator. Appendix A of the
> full guide explains how to install the operating system again if that is ever needed.

---

## A. Check your kit

From the Development Kit box:

- [ ] Compute Module 5 (CM5) with wireless, 4 GB RAM and eMMC storage, with Raspberry Pi OS
      already installed
- [ ] CM5 IO Board, fitted inside the metal IO Case (the case has a built-in fan)
- [ ] CM5 Cooler (aluminium heat sink)
- [ ] Antenna Kit (antenna and cable)
- [ ] 27 W USB-C power supply
- [ ] 2 × HDMI to HDMI cables and 1 × USB-A to USB-C cable

From the lab:

- [ ] HQ Camera (Sony IMX477, CS-mount)
- [ ] 6 mm wide-angle CS-mount lens (about 55° horizontal view)
- [ ] **Standard–Mini camera cable** (15-pin at one end, 22-pin at the other end). The cable that
      comes with the HQ Camera does **not** fit the CM5 IO Board.
- [ ] The Pi Hut mounting plate for the HQ Camera (Pi 4/5, Basic version) and its bag of screws,
      nuts and spacers, if the lab gives you one
- [ ] Tripod, or another camera holder that a demonstrator gives you
- [ ] Keyboard, mouse and monitor at the monitor station

---

## B. Assemble the development kit (power unplugged)

1. Remove the four side screws and lift the top of the case off.
2. On the IO Board, check that there is **no** jumper on pins 1–2 of **J2** (`nRPIBOOT`).
3. Press the CM5 onto the two connectors of the IO Board until it sits flat. If the CM5 is already
   on the IO Board, do not remove it. Check that it sits flat and continue with step 4. If the
   cooler is already fitted too, ask a demonstrator before you connect the antenna.
4. Press the antenna cable's small U.FL plug onto the socket on the CM5, next to its top-left
   mounting hole.
5. Remove the rubber plug from the antenna hole in the case. Push the antenna cable's threaded
   connector through the hole from the inside (flat side up). From the outside, put on the washer
   and then the hexagonal nut, and turn the nut clockwise until the connector is held firmly. Do
   not overtighten it. Then screw the antenna on.
6. Peel the paper off the cooler's silicone pad. Place the cooler on the CM5 with its cut-out over
   the CM5's antenna area and the U.FL socket.
7. Check that the fan cable is plugged into **FAN (J14)**.
8. Pass the **22-pin end** of the Standard–Mini cable through a slot at the back of the case, and
   connect it to **CAM/DISP 0**: pull the dark flap out until it stops, insert the cable straight
   with the metal contacts facing away from the flap, and push the flap back in.
9. Put the top back on. Check that no cable is trapped, then put the four screws back without
   overtightening them. If the top does not close over the cooler, do not force it. Your case may
   be the first version, which has no room for both the fan and the cooler. Ask a demonstrator
   before you remove anything.

---

## C. Mount the camera on the plate (power still unplugged)

The plate is made for a Raspberry Pi 4 or 5 board. The CM5 IO Board is much larger and stays in
its case, so the plate holds **only the camera**. The case sits next to the camera, connected by
the cable.

If the lab gives you no plate, do only steps 2, 6 and 7.

1. Peel the protective film off the acrylic plate.
2. Remove the original cable from the camera (it has the wide 15-pin plug at both ends): pull the
   dark flap out until it stops and slide the cable out. Put it aside, because you return it with
   the kit. Then connect the **15-pin end** of the Standard–Mini cable: insert it straight with the
   metal contacts facing away from the flap (towards the camera board), and push the flap back in.
3. Push the four **10 mm** nylon screws through the camera's corner holes from the lens side, and
   screw a nut onto each screw at the back of the camera board.
4. Put the plate onto the screws and fix it with the other four nuts. Tighten them gently and do
   not pinch the cable.
5. Keep the 5 mm screws and the 6 mm spacers in the bag. They are for a Raspberry Pi 4 or 5, which
   we do not use here.
6. If a thin threaded ring (the **C-CS adapter**) is fitted, unscrew it and put it in the kit bag.
   You need it later only for the 16 mm C-mount lens. Do not touch the back focus adjustment ring;
   if it is loose, ask a demonstrator. Screw the 6 mm lens clockwise all the way in. You will focus
   it in section F.
7. Screw the tripod into the 1/4-inch thread on the camera. If a demonstrator gives you a different
   holder, use that holder instead.

---

## D. First boot and network

1. Connect the monitor to an HDMI port, and the keyboard and mouse to the USB-A ports.
2. Plug in the 27 W USB-C power supply last. If the CM5 does not start, press the power button
   on the back of the case once.
3. If the CM5 asks you to log in, ask a demonstrator for the username and the password.
4. Enable the external antenna. Open a terminal and open the configuration file:

   ```bash
   sudo nano /boot/firmware/config.txt
   ```

   If the line `dtparam=ant2` is not already at the end of the file, add it there on a new line.
   Save with `Ctrl+O`, then `Enter`, and close the editor with `Ctrl+X`. Then restart:

   ```bash
   sudo reboot
   ```

5. Connect the CM5 to **eduroam** with your own UCL account. Click the network icon at the top
   right of the screen, then click **eduroam**. In the window "Wi-Fi Network Authentication
   Required", set these values:
   - Wi-Fi security: WPA & WPA2 Enterprise (already selected)
   - Authentication: change "Tunneled TLS" to **Protected EAP (PEAP)**
   - Anonymous identity: leave it empty
   - Domain: `ucl.ac.uk`
   - CA certificate: leave "(None)", and tick **No CA certificate is required**
   - PEAP version: Automatic
   - Inner authentication: **MSCHAPv2**
   - Username: your UCL user ID followed by `@ucl.ac.uk`, for example `abcd123@ucl.ac.uk`
   - Password: your UCL password

   Then click **Connect**. These values come from UCL's guide for Linux. Open it on your laptop or
   phone if you need it, because the CM5 has no internet yet:
   <https://www.ucl.ac.uk/isd/how-to/connecting-to-eduroam-wi-fi-linux>
6. Open a terminal and update the package lists:

   ```bash
   sudo apt update
   ```

---

## E. Remote access with Raspberry Pi Connect

Raspberry Pi Connect gives you the CM5's **screen** and a **remote shell** in a web browser,
without a monitor. It works on the CM5 in the same way as on a Raspberry Pi 5.

1. On your laptop, go to <https://connect.raspberrypi.com>. Make one new Raspberry Pi ID for your
   team, and use it only for this course. Give it a new password that you do not use anywhere
   else. Do not use a team member's personal account. Only the Raspberry Pi ID that owns the
   device can open Screen sharing and Remote shell, so the whole team uses this one ID.
2. On the CM5, run these commands in a terminal:

   ```bash
   rpi-connect on
   rpi-connect signin
   ```

   The second command prints a link, for example
   `https://connect.raspberrypi.com/verify/ABCD-1234`. Open this link in the browser on your
   laptop and sign in with the team's Raspberry Pi ID. In this way the password is never typed on
   the CM5. If the page asks for a device name, use your team name, for example `team-07`.
   (You can also click the Connect icon in the taskbar and choose *Turn On Raspberry Pi Connect*.
   But then you type the password in the browser on the CM5, so use the terminal way if you can.)
3. On your laptop, select the device and open **Screen sharing** and **Remote shell**. A grey
   **Screen sharing** badge under the device name is normal: it means that screen sharing is
   available.

> **Pi Connect does not offer screen sharing?** Run `rpi-connect doctor` and read which check
> fails. If auto login is off, run `sudo raspi-config`, choose **System Options**, then
> **Auto Login**, and answer **Yes** to the question about logging in to the desktop
> automatically. If the desktop does not use Wayland, run `sudo raspi-config`, choose
> **Advanced Options**, then **Wayland**, and choose **labwc**. Restart the CM5 after each change.
> If screen sharing is still not offered, show the output of `rpi-connect doctor` to a
> demonstrator. Documentation: <https://www.raspberrypi.com/documentation/services/connect.html>

---

## F. First camera capture (sanity check)

Do this section **before** you write any Python. If the official `rpicam-apps` tools fail, the
Picamera2 scripts will also fail. The camera apps (`rpicam-apps`) are already installed.

Run `rpicam-hello` and the Python scripts in a terminal on the CM5 desktop (lab monitor or Pi
Connect Screen sharing), not in Remote shell, because Remote shell has no desktop.

First check that the CM5 finds the camera. Run this command in that terminal:

```bash
rpicam-hello --list-cameras   # should list imx477
```

If no camera is listed, follow the last section of this checklist,
["If no camera is listed: tell the CM5 about the camera"](#if-no-camera-is-listed-tell-the-cm5-about-the-camera),
and then come back here.

When the camera is listed, take the first pictures with these commands:

```bash
# 1) Live preview until you press Ctrl+C. Use it to focus the lens.
rpicam-hello -t 0

# 2) Capture a still image
rpicam-still -o week6_test.jpg

# 3) Record a 10-second video at 1920 x 1080 pixels and 30 frames per second, as an MP4 file
rpicam-vid -t 10000 -o week6_test.mp4 --width 1920 --height 1080 --framerate 30
```

**Focusing:** while the preview of command 1 is open, turn the middle ring of the lens to set the
brightness (aperture), and lock it with its screw. Lock the inner NEAR/FAR ring with its screw,
then turn the two outer rings together until the image is sharp. This can take four or five
whole turns. Then press `Ctrl+C` to stop the preview before you run command 2, because only one
program can use the camera at a time.

Open `week6_test.jpg` and check that it is sharp. You also have the video `week6_test.mp4`.
Double-click it in the file manager to check it.

---

## G. Check before you continue

- [ ] The camera is detected: `rpicam-hello --list-cameras` lists `imx477`.
- [ ] You have captured and viewed a **sharp** `week6_test.jpg`.
- [ ] Pi Connect **Screen sharing** works from your laptop.
- [ ] Pi Connect **Remote shell** works from your laptop.
- [ ] The camera's original 15-pin to 15-pin cable (step C2) and, if you have a plate, its spare
      screws and spacers are with the kit, so that you return them with it.

**Next:** the Picamera2 tutorial, [`../week06_picamera2_tutorial/README.md`](../week06_picamera2_tutorial/README.md)
(the folder `tutorials/week06_picamera2_tutorial`). If the session ends before you start it, shut down
with `sudo poweroff`, and return your kit as instructed.

---

## Problems? Quick fixes

- **No camera listed:** follow the last section of this checklist,
  ["If no camera is listed: tell the CM5 about the camera"](#if-no-camera-is-listed-tell-the-cm5-about-the-camera).
  Check that the cable is in **CAM/DISP 0**. Power off, unplug, open the case, and reseat both
  cable ends.
- **Black preview:** remove the lens cap and open the aperture.
- **The CM5 does not start:** check that there is no jumper on J2 pins 1–2. Then ask a
  demonstrator. If the operating system needs to be installed again, follow Appendix A of
  [`CM5_Dev_Kit_Guide.md`](CM5_Dev_Kit_Guide.md#appendix-a-write-the-operating-system-to-the-emmc).
- **Weak Wi-Fi:** check the antenna and the U.FL plug, and check that `dtparam=ant2` is in
  `config.txt`.
- **eduroam does not connect:** there is no "accept the certificate" button on the Raspberry Pi.
  Check the values in step D5. If you saved wrong values, clicking eduroam again only asks for the
  password. Delete the saved connection with `sudo nmcli connection delete eduroam` and connect
  again from the network icon. If this command says that there is no such connection, run
  `nmcli connection show` to see the name of the connection. If it still does not connect, ask a
  demonstrator.
- **The top of the case does not close over the cooler:** do not force it. Ask a demonstrator
  before you remove anything.
- **Power warning or random restarts:** use the 27 W power supply from the kit.
- **No picture on the monitor:** try the other HDMI port and reseat the cable.

---

## If no camera is listed: tell the CM5 about the camera

Do this only if `rpicam-hello --list-cameras` in section F does not list `imx477`. On Compute
Modules, Raspberry Pi's documentation tells you to set up the camera by hand. We check first,
because the camera may also be found without this step. Open the configuration file:

```bash
sudo nano /boot/firmware/config.txt
```

1. Change the line `camera_auto_detect=1` to `camera_auto_detect=0`. If there is no such line,
   add `camera_auto_detect=0` at the end of the file. If it already says `=0`, leave it.
2. If the line `dtoverlay=imx477,cam0` is not already at the end of the file, add it there.
   (`cam0` means connector CAM/DISP 0.)
3. Save, close the editor, and restart with `sudo reboot`.
4. Check that the camera is detected:

   ```bash
   rpicam-hello --list-cameras   # should list imx477
   ```

5. If `imx477` is listed, go back to section F and take the first pictures. If it is still not
   listed, shut down with `sudo poweroff` and unplug the power. Open the case (step B1) and check
   that the cable is in **CAM/DISP 0**. If the camera is on the plate, remove the four nuts that
   hold the plate and lift the plate off, so that you can reach the connector on the camera (step
   C4). Reseat both ends of the cable. Then put the plate back on (if you removed it) and put the
   top of the case back on. Plug in the power again and repeat step 4. If the camera is still not
   listed, ask a demonstrator.

The full explanation of these steps, and what to do if the cable is in CAM/DISP 1, is in
[Appendix B of the full guide](CM5_Dev_Kit_Guide.md#appendix-b-tell-the-cm5-that-the-hq-camera-is-connected).

---

**After this week:** calibration (Week 7), then colour segmentation and line fitting (Week 8).
These build toward your contact-less airspeed sensor (demo on 11 December). · COMP0219, UCL.
