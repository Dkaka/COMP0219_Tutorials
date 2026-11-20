# Week 6 (CM5 version): Compute Module 5 Development Kit and Camera Setup

This guide is for teams who have the **Raspberry Pi Development Kit for Compute Module 5**
instead of a standard Raspberry Pi 5. If your team has a standard Raspberry Pi 5, follow
[`README.md`](README.md) instead.

This session starts **Part 2** of COMP0219, the *contact-less* (vision-based) airspeed sensor.
In this session you will assemble the development kit, attach the Raspberry Pi High Quality
(HQ) Camera with the 6 mm lens, connect the kit to the network, set up **Raspberry Pi Connect**
so that you can reach the kit from your laptop, and capture a first image.

> **Goal of Week 6:** at the end of the session, you have a working Compute Module 5 and camera
> that you can reach from your laptop without a monitor. The coding starts later in the same
> session, in the Picamera2 tutorial: see [`../week06_picamera2_tutorial`](../week06_picamera2_tutorial).

---

## How the CM5 kit differs from the Raspberry Pi 5

A Compute Module 5 (CM5) contains the same processor as a Raspberry Pi 5, but it has no
connectors of its own. It plugs into a larger carrier board, the **CM5 IO Board**, which provides
the connectors. Most of the steps are the same as in the Pi 5 guide. The important differences
are:

| Topic | Raspberry Pi 5 (main guide) | CM5 Development Kit (this guide) |
| --- | --- | --- |
| Where the operating system lives | On a microSD card | On storage built into the CM5, called **eMMC**. The microSD slot is not used. |
| Monitor cable | Micro-HDMI to HDMI | Full-size HDMI to HDMI (two cables are in the kit) |
| Camera detection | Automatic | You check it in section G. If the camera is not found, you add two lines to `/boot/firmware/config.txt` (Appendix B) |
| Camera cable | Supplied with the Pi 5 kit | **Not** in the development kit; the lab supplies it (section A) |
| Mounting plate | Not used in the Pi 5 guide | If the lab gives you one, it holds **only the camera**; the IO Board stays in its case (section D) |
| Wi-Fi antenna | Built into the board | An external antenna is in the kit; you enable it in `config.txt` (section E) |
| Power button | On the board | On the back of the metal case |

---

## A. What you receive (per team)

**From the Development Kit box.** According to the official product brief (September 2026
edition), the kit contains:

- **Raspberry Pi Compute Module 5** with wireless, 4 GB RAM and 64 GB eMMC (part number
  CM5104064). The teaching team has already installed Raspberry Pi OS on it (see section B).
- **Compute Module 5 IO Board.** In current kits the board is already fitted inside the case.
- **IO Case for Compute Module 5.** This is a metal case with a built-in fan.
- **Cooler for Compute Module 5.** This is a passive aluminium heat sink.
- **Antenna Kit.** This is an external Wi-Fi and Bluetooth antenna with a cable.
- **27 W USB-C power supply.** Use this power supply. A weaker supply can cause crashes.
- **2 × HDMI to HDMI cables** and **1 × USB-A to USB-C cable**.

The development kit does **not** contain a camera cable, a microSD card, a keyboard or a mouse.

**From the lab (in addition to the kit):**

- **Raspberry Pi HQ Camera.** It uses the Sony IMX477 sensor (12.3 megapixels, 4056 × 3040
  pixels) and has a CS-mount for the lens.
- **6 mm wide-angle CS-mount lens**: about 55° horizontal × 45° vertical field of view on the HQ
  camera, with manual focus and aperture. The 120 fps mode uses only the centre of the sensor, so
  its view is narrower (about 38° wide).
- **Camera cable, "Standard–Mini" type** (15-pin at one end, 22-pin at the other end). The CM5
  IO Board has the smaller 22-pin camera connectors, so the 15-pin to 15-pin cable that comes
  with the HQ Camera does **not** fit. Raspberry Pi sells the Standard–Mini cable in 200 mm,
  300 mm and 500 mm lengths.
- **The Pi Hut "Mounting Plate for High Quality Camera", Raspberry Pi 4/5, Basic version**, if the
  lab gives you one.
- **Tripod**, or another camera holder that a demonstrator gives you (section D, step 6).
- **Keyboard, mouse and monitor** at the lab monitor station.

> Handle the boards by their edges and work on the anti-static mat. The kit must be
> **unplugged** whenever you connect or disconnect the camera cable.

---

## B. The operating system is already installed

The CM5 in the lab's kits has built-in storage called **eMMC**. The CM5 starts from the eMMC, in
the same way as a Raspberry Pi 5 starts from its microSD card. The microSD card slot on the IO
Board is not used: a CM5 with eMMC ignores it, so inserting a card does not change anything.

The teaching team has already installed **Raspberry Pi OS** on the eMMC of every kit. You do not
need to install anything. Continue with section C.

If your CM5 does not start, ask a demonstrator first. If the eMMC turns out to be empty or the
operating system is damaged, you can install it again from another computer over USB. This is
described in [Appendix A](#appendix-a-write-the-operating-system-to-the-emmc).

---

## C. Assemble the development kit

Do this whole section with the power supply **unplugged**. You need a small Phillips (cross-head)
screwdriver. The steps follow the official case assembly instructions.

1. **Open the case.** Remove the four screws on the sides of the case (two on the left side and
   two on the right side). Lift the top of the case off the base. Keep the screws safe. In
   current kits the IO Board is already fitted in the base of the case. If it is not, place it in
   the base so that its connectors line up with the openings in the case, and fix it with four
   M2.5 screws in its corner holes.
2. **Check the J2 jumper block.** On the IO Board, find the jumper block labelled **J2**. For a
   normal start from the eMMC, there must be **no** jumper on pins 1–2 (labelled `nRPIBOOT`).
   A jumper on these pins makes the CM5 try to start from USB instead (this is only used in
   Appendix A).
3. **Fit the CM5 onto the IO Board.** Turn the CM5 so that the two connectors on its underside
   line up with the two matching connectors on the IO Board, and so that the four mounting holes
   in the corners of the CM5 line up with the holes on the board. Press down gently but firmly
   until the CM5 sits flat on the board.
   If the CM5 is already on the IO Board (the teaching team fitted it to install the operating
   system), do not remove it. Check that it sits flat and continue with step 4. If the cooler is
   already fitted too, ask a demonstrator before you connect the antenna.
4. **Connect the antenna cable to the CM5.** The antenna cable has a small round U.FL plug at
   one end. Press it onto the U.FL socket on the CM5, next to the CM5's top-left mounting hole,
   until it clicks. Do this **before** you fit the cooler, because the cooler makes the socket
   harder to reach.
5. **Fit the antenna to the case.** Remove the rubber plug from the antenna hole in the case.
   From the inside of the case, push the threaded (SMA) connector at the other end of the
   antenna cable through the hole, with the flat side of the connector facing up. From the
   outside, put the washer and then the hexagonal nut onto the connector and turn the nut
   clockwise until the connector is held firmly. Do not overtighten it. Then screw the antenna
   onto the connector by turning it clockwise. Finally, fold the antenna up to a 90 degree angle.
6. **Fit the cooler.** Peel the protective paper off the silicone pad on the bottom of the cooler.
   Place the cooler on top of the CM5 so that the cut-out in the cooler sits over the CM5's
   built-in antenna (the trapezoid-shaped area at one end of the module) and over the U.FL
   socket. Press it down. You can also fix it with screws in its four corners, but this is
   optional.
7. **Check the fan.** The fan is fitted inside the top of the case. Its cable must be plugged into
   the four-pin socket labelled **FAN (J14)** on the IO Board. Plug it in if it is not already
   connected. In the first version of the case, the fan and the cooler do not fit together. If
   the top of the case does not close over the cooler, see
   [The top of the case does not close over the cooler](#the-top-of-the-case-does-not-close-over-the-cooler)
   in the Troubleshooting section.
8. **Connect the camera cable to the IO Board.** Take the Standard–Mini camera cable. Pass its
   **narrow 22-pin end** from the outside through one of the slots at the back of the case. On
   the IO Board, find the two camera connectors labelled **CAM/DISP 0** and **CAM/DISP 1**. Use
   **CAM/DISP 0**, because it works without extra jumpers (CAM/DISP 1 needs two jumpers on the
   J6 block). Gently pull the dark plastic flap of the connector
   out until it stops, insert the cable straight and fully with its **metal contacts facing away
   from the flap**, and push the flap back in until it clicks. The wider 15-pin end stays outside
   the case for now. Do not bend the cable sharply.
9. **Close the case.** Put the top of the case back on the base. Line up the screw holes on both
   sides and the power button at the back. Check that the camera cable and the fan cable are not
   trapped between the two halves. Put the four screws back in without overtightening them.

Do **not** connect the power supply yet. First prepare the camera in section D, so that the
camera is never connected while the power is on.

---

## D. Mount the HQ Camera on the mounting plate

### Can the Pi 4/5 mounting plate hold the CM5 IO Board?

**No, not in the way The Pi Hut's assembly guide shows.** The plate is made for the Raspberry Pi
4 and Raspberry Pi 5 boards, which measure about 85 mm × 56 mm. The Pi Hut lists it as
compatible with the Raspberry Pi 3B, 4B, 5 and Zero 2, and does not mention the CM5 IO Board.
The CM5 IO Board measures 160 mm × 90 mm, so it is far larger than the plate. It also comes
fitted inside its metal case, which protects it, and we keep it in the case.

We therefore use the plate in a different way from The Pi Hut's guide:

- The plate holds **only the camera**. It protects the back of the camera board and gives you a
  flat surface to hold the camera.
- The camera stands on a tripod or on the rig. The tripod screws into the **1/4-inch (1/4"-20)
  tripod thread** on the HQ Camera itself, not into the plate.
- The CM5 IO Board stays in its case, next to the camera.
- The Standard–Mini cable runs from the camera, through the slot at the back of the case, to
  the CAM/DISP 0 connector. The camera can only be as far from the case as the cable allows, so
  place the case close to the camera.

Do **not** take the IO Board out of its case to fit it to the plate.

### What is in the plate pack

The Pi Hut lists these parts for the Basic version of the plate (the Pro version has a few extra
parts, which we do not use):

- 1 × clear acrylic mounting plate
- 4 × 10 mm M2.5 nylon screws (for the camera)
- 8 × M2.5 nylon nuts
- 8 × 5 mm M2.5 nylon screws (for fixing a Raspberry Pi to the spacers)
- 4 × 6 mm female-to-female nylon spacers (for holding a Raspberry Pi)

With the CM5 kit you only need the plate, the four 10 mm screws and the eight nuts. Keep the
5 mm screws and the spacers in the bag and return them with the kit.

### Assembly steps

These steps adapt The Pi Hut's assembly guide. Their guide shows the Zero version of the plate,
but they state that all versions are assembled in the same way, with the holes in slightly
different positions.

If the lab gives you no mounting plate, do only steps 2, 5, 6 and 7 below.

1. **Peel off the protective film.** The acrylic plate is covered with a thin protective film on
   both sides. Peel it off.
2. **Change the camera cable on the camera.** The HQ Camera comes with a 15-pin to 15-pin cable
   already attached, which does not fit the CM5 IO Board. The camera's cable connector is on the
   back of the camera board, on the opposite side from the lens. Gently pull the connector's flap
   out until it stops, slide the old cable out, and put it aside to return to the lab. Then
   insert the **wide 15-pin end** of the Standard–Mini cable (the other end of the cable that you
   connected in section C). The metal contacts must face away from the flap and towards the
   camera board. Push the flap back in until it clicks. Do this before you fix the camera to the
   plate, because the connector is hard to reach afterwards.
3. **Fit the screws to the camera.** Push the four 10 mm nylon screws through the four corner
   holes of the camera board, from the lens side. On the back of the camera board, screw one
   nylon nut onto each screw. These nuts act as small spacers. They keep the components on the
   back of the camera board away from the plate, so that they are not damaged.
4. **Fix the camera to the plate.** Place the plate on the four screws so that the screws pass
   through the four camera holes in the plate. Screw the remaining four nuts onto the screws on
   the other side of the plate. Make sure the camera cable comes out between the camera board and
   the plate without being pinched or bent sharply. Tighten the nuts gently. The nylon threads
   are easy to damage, and The Pi Hut warns that acrylic can crack under stress.
5. **Fit the lens.** The HQ Camera and the 6 mm lens are both CS-mount, so the lens screws
   directly onto the camera. Raspberry Pi's guide for this lens gives these steps:
   - If a thin threaded ring (the **C-CS adapter**) is fitted in the camera's lens mount, unscrew
     it and put it in the kit bag. The 6 mm lens does not focus correctly when the adapter is
     fitted. You need the adapter later only for the 16 mm C-mount lens.
   - Do not touch the **back focus adjustment ring** on the camera. For the 6 mm lens it stays
     screwed in fully. If it is loose, ask a demonstrator.
   - Screw the lens clockwise all the way in.
   - You will focus it in section G, where you can watch a live picture.
6. **Mount the camera on the tripod.** Screw the tripod into the 1/4-inch tripod thread on the
   camera body. If a demonstrator gives you a different holder, use that holder instead.
7. **Check the cable route.** The cable should run from the camera to the slot at the back of
   the case without being stretched, twisted or folded sharply.

---

## E. First boot and network

1. **Connect the monitor, keyboard and mouse.** Connect one of the IO Board's two full-size HDMI
   ports to the monitor with an HDMI cable from the kit. Connect the keyboard and the mouse to the
   USB-A ports.
2. **Connect the power last.** Plug the 27 W USB-C power supply into the USB-C port of the IO
   Board. The CM5 should start on its own. If nothing appears on the monitor after about a
   minute, press the power button on the back of the case once.
3. **Log in.** The desktop should log in automatically. If the CM5 asks for a username and a
   password, ask a demonstrator.
4. **Enable the external antenna.** The CM5 uses its small built-in antenna unless you tell it to
   use the external antenna. Because the case is made of metal, use the external antenna. Open a
   terminal and edit the configuration file:

   ```bash
   sudo nano /boot/firmware/config.txt
   ```

   Go to the very end of the file. If the line `dtparam=ant2` is already there, close the editor
   (press `Ctrl+X`) and continue with step 5. Otherwise, add this line on a new line:

   ```ini
   dtparam=ant2
   ```

   Save the file (press `Ctrl+O`, then `Enter`) and close the editor (press `Ctrl+X`). Restart
   the CM5:

   ```bash
   sudo reboot
   ```
5. **Connect to eduroam.** Connect the CM5 to **eduroam** with your own UCL account.
   1. Click the network icon at the top right of the screen, then click **eduroam**. A window
      called "Wi-Fi Network Authentication Required" opens.
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
   laptop or phone if you need it, because the CM5 has no internet yet. Its menu steps are for a
   different Linux desktop, but the values are the same.
6. **Update the package lists.** Open a terminal and run:

   ```bash
   sudo apt update
   ```

**About the power button.** The power button works in the same way as on a Raspberry Pi 5. A short
press opens the shutdown menu, and a second short press shuts the system down. A long press
forces the power off; avoid this, because it can damage files that are being written. When the
system is shut down, a short press starts it again.

---

## F. Remote access with Raspberry Pi Connect

Raspberry Pi Connect lets you see the CM5's **screen** and use a **remote shell** (a terminal)
from a web browser on your laptop. Once it is set up, you do not need a monitor. You will use it
in the project to start and stop your sensing software remotely.

Connect works on the CM5 in the same way as on a Raspberry Pi 5. We found no steps specific to
Compute Modules in the official documentation. Connect needs Raspberry Pi OS Bookworm or later,
and it is installed by default on Raspberry Pi OS with desktop. The service is switched off until
you turn it on.

1. **Make one Raspberry Pi ID for your team.** On your laptop, go to
   <https://connect.raspberrypi.com> and create the account there. Make one new Raspberry Pi ID
   for your team, and use it only for this course. Give it a new password that you do not use
   anywhere else. Do not use a team member's personal account. Only the Raspberry Pi ID that owns
   the device can open Screen sharing and Remote shell, so the whole team uses this one ID.
2. **Turn Connect on and sign in, on the CM5.** Run `rpi-connect on` and then
   `rpi-connect signin` in a terminal on the CM5:

   ```bash
   rpi-connect on
   rpi-connect signin
   ```

   The second command prints a link, for example
   `https://connect.raspberrypi.com/verify/ABCD-1234`. Open this link in the browser on your
   laptop and sign in with the team's Raspberry Pi ID. In this way the password is never typed on
   the CM5. If the page asks for a device name, use your team name, for example `team-07`.

   You can also use the **Connect icon** in the taskbar: click it and choose *Turn On Raspberry Pi
   Connect*, then sign in. This way opens the browser on the CM5, so you type the password on the
   CM5. If the CM5's browser offers to save the password, choose **Never**.
3. **Check the device name.** At <https://connect.raspberrypi.com>, open **Devices** and check
   that your device has your team name. To change the name, select your device, then
   **Settings**.
4. **Test it from your laptop.** At <https://connect.raspberrypi.com>, select your device. A grey
   **Screen sharing** badge under the device name is normal: it means that screen sharing is
   available. Open **Screen sharing** and check that you see the CM5 desktop. Then open
   **Remote shell** and check that you get a working terminal.
5. **Check the status from a terminal** at any time with:

   ```bash
   rpi-connect status
   ```

> **Where to run the scripts.** Programs that open a window must run in a terminal on the CM5
> desktop: either on the lab monitor, or inside Raspberry Pi Connect **Screen sharing**. This
> applies to `rpicam-hello` and to scripts 1, 3, 3.1, 3.2 and 4 of the Picamera2 tutorial. Raspberry Pi
> Connect **Remote shell** has no desktop, so these programs cannot open a window there.
> `rpicam-hello` then shows no picture, and the Python scripts stop with an error. `rpicam-still`,
> `rpicam-vid` and scripts 2, 5 and 5.1 do not need a window, so they also work in Remote shell. Script 3.2 also works there if you turn its window off (section 3.2 of the Picamera2 tutorial). A program that you start in Remote shell stops when you close the browser tab.

> **Screen sharing needs a Wayland desktop and an active desktop session.** Both are the default
> on current Raspberry Pi OS with desktop. When no monitor is connected, Raspberry Pi OS creates a
> virtual display, so screen sharing still works without a monitor. If screen sharing is not
> offered, see the Troubleshooting section.
>
> Each remote shell is a new session, in the same way as SSH. If you need a program to keep running
> after you close the browser tab, run it inside `tmux`. tmux is not installed by default, so
> install it once with `sudo apt install -y tmux`. Start a tmux session with `tmux new -s run`,
> and start your program in it. To leave the program running, press `Ctrl+B`, release both keys,
> and then press `D`. To return to it later, run `tmux attach -t run`.
>
> Documentation: <https://www.raspberrypi.com/documentation/services/connect.html>

At the end of the course, run `rpi-connect signout` on the CM5, and delete the device at
connect.raspberrypi.com (Devices, then your device, then Settings, then Delete device).

---

## G. First camera capture (sanity check)

Do this section **before** you write any Python. The command-line tools from the official
`rpicam-apps` package are the simplest test of the camera. If they fail, the Picamera2
scripts will also fail. The camera apps (`rpicam-apps`) are already installed.

Run `rpicam-hello` and the Python scripts in a terminal on the CM5 desktop (lab monitor or Pi
Connect Screen sharing), not in Remote shell, because Remote shell has no desktop.

First check that the CM5 finds the camera. Run this command in that terminal:

```bash
rpicam-hello --list-cameras
```

If the output lists a camera called `imx477`, continue below. If it says that no cameras are
available, follow [Appendix B](#appendix-b-tell-the-cm5-that-the-hq-camera-is-connected), which
adds two lines to the configuration file, and then come back here.

When the camera is listed, take the first pictures with these commands:

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

You should now have `week6_test.jpg` and `week6_test.mp4` in the folder where you ran the
commands (your home folder, if you opened a new terminal). Open the still image and check that it
is sharp. Double-click the video in the file manager to check it.

---

## H. Set up the coding environment

You need Python later in this session for the Picamera2 tutorial, so set up the environment now. It is
the same `comp0219` environment for all Part 2 tutorials, so you do this only once. The steps are
the same on the CM5 as on a Raspberry Pi 5. Follow section B of the Picamera2 tutorial:
[Set up the Python environment (on the Raspberry Pi)](../week06_picamera2_tutorial/README.md#b-set-up-the-python-environment-on-the-raspberry-pi).
Then come back here for section I.

---

## I. Check before you continue

- [ ] Camera detected (`rpicam-hello --list-cameras` lists `imx477`).
- [ ] A sharp `week6_test.jpg` captured.
- [ ] Raspberry Pi Connect **Screen sharing** *and* **Remote shell** both work from your laptop.
- [ ] The camera's original 15-pin to 15-pin cable (section D, step 2) and, if you have a plate,
  its spare screws and spacers are with the kit, so that you return them with it.

**Next:** the Picamera2 tutorial, [`../week06_picamera2_tutorial/README.md`](../week06_picamera2_tutorial/README.md).
If the session ends before you start it, shut down with `sudo poweroff`, and return your kit as
instructed.

The Picamera2 tutorial covers Picamera2 in Python (preview, stills, resolution and frame rate, the
120 fps mode, dual streams and H.264 recording). The next tutorials are calibration (Week 7), and
then colour segmentation and line fitting (Week 8). They lead to the contact-less airspeed project.

---

## Troubleshooting

### `rpicam-hello --list-cameras` says that no cameras are available
- If you have not done so yet, follow
  [Appendix B](#appendix-b-tell-the-cm5-that-the-hq-camera-is-connected). After that, check that
  `/boot/firmware/config.txt` contains `camera_auto_detect=0` and `dtoverlay=imx477,cam0`, and
  that you restarted after editing it.
- Check that the cable is in **CAM/DISP 0**. If it is in CAM/DISP 1, either move it (with the
  power off) or follow the CAM/DISP 1 note at the end of Appendix B.
- Shut down, unplug the power, and open the case (section C, step 1). If the camera is on the
  plate, remove the four nuts that hold the plate and lift the plate off, so that you can reach the
  connector on the camera (section D, step 4). Reseat **both** ends of the cable. The metal
  contacts must face away from the connector flap, and the cable must sit straight in the
  connector. Then put the plate back on (if you removed it) and close the case before you plug in
  the power again.
- Make sure you replaced the camera's original 15-pin to 15-pin cable with the Standard–Mini
  cable.
- If the camera is still not listed, ask a demonstrator.

### The preview is black
- Remove the lens cap.
- Open the aperture: hold the front ring of the lens and turn the middle ring anticlockwise. If
  the aperture is locked, loosen the small screw on the side of the lens first.
- A lens that is out of focus gives a blurred picture, not a black one, so do not change the
  focus to fix a black preview. Do not adjust the back focus adjustment ring. If it is loose, ask
  a demonstrator.

### The CM5 does not start from the eMMC
- Check that there is **no** jumper on pins 1–2 (`nRPIBOOT`) of the **J2** block. With this
  jumper fitted, the CM5 tries to start from USB.
- A microSD card is ignored by a CM5 with eMMC, so inserting one does not help.
- Ask a demonstrator. If the eMMC turns out to be empty or the operating system is damaged,
  install it again with [Appendix A](#appendix-a-write-the-operating-system-to-the-emmc).

### Weak or no Wi-Fi
- Check that the antenna is screwed on and that the U.FL plug is pressed fully onto the CM5.
- Check that `dtparam=ant2` is at the end of `/boot/firmware/config.txt`, and restart.

### Pi Connect does not offer screen sharing
- Make sure the full `rpi-connect` package is installed, not `rpi-connect-lite`.
- Run `rpi-connect doctor` and read which check fails. If auto login is off, run
  `sudo raspi-config`, choose **System Options**, then **Auto Login**, and answer **Yes** to the
  question about logging in to the desktop automatically. If the desktop does not use Wayland, run
  `sudo raspi-config`, choose **Advanced Options**, then **Wayland**, and choose **labwc**.
  Restart the CM5 after each change. If screen sharing is still not offered, show the output of
  `rpi-connect doctor` to a demonstrator.

### You cannot sign in to Connect
- Run `rpi-connect signin` again.
- Run `rpi-connect doctor`. Every line should show a tick. If a network line fails on eduroam,
  tell a demonstrator.

### eduroam does not connect
- There is no "accept the certificate" button on the Raspberry Pi. Check the values in the list
  in section E, step 5. If you saved wrong values, clicking eduroam again only asks for the
  password. Delete the saved connection with `sudo nmcli connection delete eduroam` and connect
  again from the network icon. If this command says that there is no such connection, run
  `nmcli connection show` to see the name of the connection. If it still does not connect, ask a
  demonstrator.

### There is no picture on the monitor
- Try the other HDMI port on the IO Board, and reseat the cable.

### The CM5 restarts by itself or shows a power warning
- Use the 27 W USB-C power supply from the kit. The IO Board shows a warning when the power supply
  cannot provide 5 A.

### The top of the case does not close over the cooler
- You may have the first version of the case, which has no room for both the fan and the cooler.
  Ask a demonstrator before you remove anything.

---

## Appendix A: Write the operating system to the eMMC

The teaching team has already installed Raspberry Pi OS on every kit, so you normally do not need
this appendix. It is here in case a CM5 does not start, for example because its eMMC is empty or
the operating system has been damaged. **Ask a demonstrator before you follow it**, because it
erases everything on the eMMC. The steps follow the official "Flash an image to a Compute Module"
instructions.

**You need:** another computer (Windows 11, macOS or Linux) with
[Raspberry Pi Imager](https://www.raspberrypi.com/software/) installed, a USB cable (the kit's
USB-A to USB-C cable, or a USB-C to USB-C cable), and a jumper for the J2 block.

1. **Unplug the power supply** from the IO Board and open the case (section C, step 1).
2. **Fit a jumper on pins 1–2 of J2** (labelled `nRPIBOOT`). This makes the CM5 wait for a
   computer to send it software over USB, instead of starting from the eMMC.
3. **Turn on RPIBOOT support in Imager.** Start Imager and open its *Debug Options* with
   `Ctrl+Alt+S` (Windows or Linux) or `Cmd+Option+S` (macOS). Scroll to *Advanced Features*,
   switch on *Enable Rpiboot/Fastboot Support*, and select *Apply*. Imager remembers this setting.
4. **Connect the IO Board to the computer.** Connect the IO Board's USB-C port (the same port
   that is normally used for power) to the computer. In this mode, the computer powers the board
   through the cable. If the computer cannot provide enough power, use a powered USB hub. If the
   computer does not find the CM5, connect it directly instead of through a hub.
5. **Choose the device, the operating system and the storage in Imager.**
   - Device: choose **Raspberry Pi 5**. Imager's Raspberry Pi 5 entry also covers the Compute
     Module 5.
   - Operating system: choose the same version of Raspberry Pi OS (64-bit) as on the other kits.
     The teaching team knows which version this is, so ask a demonstrator. To see the version on
     a working kit, type `grep VERSION_CODENAME /etc/os-release` in a terminal on that kit. It
     prints a line such as `VERSION_CODENAME=trixie`. In Imager, the first entry, **Raspberry Pi OS (64-bit)**, is
     the newer version (Trixie). The older version (Bookworm) is under **Raspberry Pi OS (other)**,
     as **Raspberry Pi OS (Legacy, 64-bit)**.
   - Storage: after a few seconds, the CM5's eMMC appears as a USB drive. Its size matches the
     eMMC size (for example, about 64 GB). Keep *Exclude system drives* switched on, so that you
     cannot erase your own computer's disk by mistake.
6. **Customise the settings** if Imager offers it. Set the hostname (for example `team-07-cm5`)
   and the same username and password as on the other kits (ask a demonstrator). Skip the Wi-Fi settings,
   because eduroam needs your personal UCL login, which you enter on the CM5 itself in section E.
7. **Write the image** and wait until Imager has finished writing and checking it.
8. **Disconnect the USB cable** from the computer, **remove the J2 jumper**, close the case, and
   continue with section E.

---

## Appendix B: Tell the CM5 that the HQ Camera is connected

Follow this appendix only if `rpicam-hello --list-cameras` in section G says that no cameras are
available.

On a standard Raspberry Pi 5, the camera is detected automatically. On Compute Modules,
Raspberry Pi's documentation tells you to configure the camera by hand: you switch off automatic
detection and name the camera's driver in the configuration file. This guide asks you to check
first in section G, because the camera may also be found without this step.

**Why we use CAM/DISP 0.** According to the CM5 IO Board datasheet, CAM/DISP 0 works with a
camera without any extra jumpers. CAM/DISP 1 needs two jumpers fitted on the J6 jumper block.
Using CAM/DISP 0 avoids this step, and it is also the connector used in the Pi 5 guide.

1. Open the configuration file:

   ```bash
   sudo nano /boot/firmware/config.txt
   ```

2. Find the line `camera_auto_detect=1` and change it to:

   ```ini
   camera_auto_detect=0
   ```

   If the line already says `camera_auto_detect=0`, leave it as it is. If the file has no
   `camera_auto_detect` line, add `camera_auto_detect=0` at the end of the file.
3. Go to the very end of the file. If the line `dtoverlay=imx477,cam0` is not already there, add
   it on a new line:

   ```ini
   dtoverlay=imx477,cam0
   ```

   `imx477` is the name of the HQ Camera's sensor driver, and `cam0` means connector CAM/DISP 0.
   Without `,cam0`, the system looks for the camera on connector 1 instead.
4. Raspberry Pi's Compute Module camera instructions also say that the line `dtparam=i2c_arm=on`
   should not be active. In the standard file this line starts with `#`, which means that it is
   switched off. If you find it without a `#`, add a `#` at the start of the line.
5. Save the file (`Ctrl+O`, then `Enter`), close the editor (`Ctrl+X`) and restart:

   ```bash
   sudo reboot
   ```

6. After the restart, check that the camera is detected:

   ```bash
   rpicam-hello --list-cameras
   ```

   The output should list a camera called `imx477`. If it does, go back to section G and take the
   first pictures. If it still says that no cameras are available, read the note below about
   CAM/DISP 1, and then follow the [Troubleshooting](#troubleshooting) section of this guide.

> **If you used CAM/DISP 1 instead:** the simplest fix is to shut down, unplug the power, open the
> case (section C, step 1), move the cable to **CAM/DISP 0**, and close the case again. Otherwise,
> ask a demonstrator for two jumpers. Shut down, unplug the power, open the case, fit the two
> jumpers on the **J6** block as printed on the board, and close the case. Then, in
> `/boot/firmware/config.txt`, change the line `dtoverlay=imx477,cam0` to `dtoverlay=imx477`
> (without `,cam0`), and restart.

---

## Sources

- Raspberry Pi, *Development Kit for Compute Module 5*, product page and product brief
  (RP-008185-DS, September 2026):
  <https://www.raspberrypi.com/products/cm5-dev-kit/> and
  <https://pip.raspberrypi.com/documents/RP-008185-DS>
- Raspberry Pi documentation, *Compute Module hardware* (CM5 IO Board, IO Case assembly, antenna,
  cooler, flashing the eMMC, attaching a camera):
  <https://www.raspberrypi.com/documentation/computers/compute-module.html>
- Raspberry Pi, *Compute Module 5 IO Board datasheet* (RP-008182-DS), sections on the MIPI
  connectors, the power input and the J2 and J6 jumpers:
  <https://pip.raspberrypi.com/documents/RP-008182-DS>
- Raspberry Pi documentation, *Camera* (installing a camera, cable types):
  <https://www.raspberrypi.com/documentation/accessories/camera.html>
- Raspberry Pi documentation, *Camera software* (`rpicam-apps` and the configuration section):
  <https://www.raspberrypi.com/documentation/computers/camera_software.html>
- Raspberry Pi, *Camera Cable* product page (Standard–Mini, 200/300/500 mm):
  <https://www.raspberrypi.com/products/camera-cable/>
- Raspberry Pi, *High Quality Camera* product page (tripod thread, supplied cable) and the
  6 mm CS-mount lens guide:
  <https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/> and
  <https://datasheets.raspberrypi.com/hq-camera/cs-mount-lens-guide.pdf>
- Raspberry Pi documentation, *Raspberry Pi Connect*:
  <https://www.raspberrypi.com/documentation/services/connect.html>
- Raspberry Pi documentation, *Getting started* (Raspberry Pi Imager):
  <https://www.raspberrypi.com/documentation/computers/getting-started.html>
- The Pi Hut, *Raspberry Pi Mounting Plate for High Quality Camera* (product page, Pi 4/5 Basic
  variant, kit contents and compatibility):
  <https://thepihut.com/products/mounting-plate-for-high-quality-camera>
- The Pi Hut, *Zero Mounting Plate for High Quality Camera Assembly Guide*:
  <https://thepihut.com/blogs/raspberry-pi-tutorials/zero-mounting-plate-for-high-quality-camera-assembly-guide>
