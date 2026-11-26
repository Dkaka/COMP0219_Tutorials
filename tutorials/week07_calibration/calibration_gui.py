"""Capture calibration photos with the Raspberry Pi camera, in a window with buttons.

    python calibration_gui.py

This is the window version of calibration_cap.py. Run it in a terminal on the Raspberry Pi desktop (on a monitor,
or in the screen sharing of Raspberry Pi Connect), in the comp0219 environment. SSH and the Pi Connect remote
shell have no desktop, so use calibration_cap.py there.

1. Press Start. The window shows the live picture of the camera.
2. Hold the chessboard in front of the camera. A photo can be saved only when the whole board is found and has
   been still for a moment (0.3 s), so that the photo is sharp. Press Space (or the Save button) to save it.
   Two switches next to the Save button change this:
   - "Motion check" (on at the start): untick it to save without waiting until the board is still. Then hold the
     board steady yourself, or the photos may be blurred.
   - "Auto capture" (off at the start): tick it, and the program saves a photo by itself each time you hold the
     board still at a new position, so you can hold the board with both hands. With the motion check off, it
     saves each new position, at most once a second.
3. Before it saves a photo, the program searches the full-size photo for the board in the same way as
   calibration.py. If this search fails, the photo is not saved. So calibration.py can use every saved photo.
4. The window shows where the board has been. The green dots on the live picture are the corners in the photos
   so far. The "Board positions" panel fills in each saved board in green, and it shows in red the parts of the
   image that no photo covers yet (usually the corners). Hold the board there next. From 4 photos on, the program
   also calibrates in the background and shows the focal length, the image centre and the RMS error so far. Tick
   "Undistorted view" to see the live picture corrected with this estimate.

The photos go to calibration_images/calibration_image_NN.jpg next to this file. Then calibrate with
    python calibration.py --images calibration_images
Keys: Space or c saves a photo, u removes the last photo saved since the program started (Undo), q quits.

Credit: this program is the capture GUI of the COMP0241 Week 3 calibration tutorial (w3_calibration_gui.py),
changed to use the Raspberry Pi camera. The green board positions, the corner dots, saving a still board by
itself, the running estimate and the undistorted live view follow OpenCV's interactive camera calibration
application (apps/interactive-calibration in the OpenCV source;
https://docs.opencv.org/4.x/d7/d21/tutorial_interactive_calibration.html). Showing the uncovered parts in red
comes from the COMP0241 GUI.
"""
import os
import re
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # Raspberry Pi OS Lite has no Tkinter; the desktop version has it
    sys.exit("Tkinter is missing. Install it with: sudo apt install -y python3-tk")

try:
    from picamera2 import Picamera2
    PICAMERA2_ERROR = None
except ImportError as error:
    Picamera2, PICAMERA2_ERROR = None, str(error)

# Parameters
# A calibration is only valid for the image size, the sensor mode and the lens focus of its photos. So use the
# size and the sensor mode of your wind-tunnel recordings, as described in calibration_cap.py and in the README.
# The default, 2028x1520 pixels, is the simplest setup that keeps the full field of view of the HQ camera.
# For the 120 fps mode of the week 6 tutorial: CAPTURE_SIZE = (1332, 990),
# SENSOR = {"output_size": (1332, 990), "bit_depth": 10}.
CAPTURE_SIZE = (2028, 1520)  # Size (width, height) of the saved photos
SENSOR = None  # Sensor mode of the camera; None lets Picamera2 choose it from CAPTURE_SIZE
CHESSBOARD_SIZE = (4, 7)  # Inner corners in the chessboard; must be the same as in calibration.py
SQUARE_SIZE = 34.0  # mm, as in calibration.py
TARGET = 17  # photos: the shot list in the README

# The live search runs on a small grey copy of each picture, which the camera makes at the same time as the
# full-size picture (the "lores" stream). It has the same shape as the photo and is LIVE_WIDTH pixels wide.
LIVE_WIDTH = 640
LIVE_SIZE = (LIVE_WIDTH, 2 * round(LIVE_WIDTH * CAPTURE_SIZE[1] / CAPTURE_SIZE[0] / 2))

# All distances below are in pixels of the full-size photo
# The limits for a hand-held board come from tests on the Pi. A hand that holds the board "still" moves its corners by
# about 2 to 5 px per picture and lets the board drift slowly; a board that is being moved goes 10 to 35 px per picture.
# Photos saved at up to 6 px per picture were as accurate as the others (indoor light, exposure time about 12 ms),
# and a slow drift does not blur a photo.
MAX_MOTION = 6.0  # px: the board counts as still if its corners moved less than this (median) since the last search
READY_TIME = 0.3  # s: a photo can be saved once the board has been still this long ...
MAX_DRIFT = 20.0  # px: ... and its corners drifted less than this (median) from where they were when it stopped
STILL_TIME = 0.6  # s: Auto capture saves once the board has been still this long ...
NEW_POSITION = 1 / 20  # ... and has moved (on average) at least this fraction of the image diagonal since the last photo
AUTO_INTERVAL = 1.0  # s: with the motion check off, Auto capture waits at least this long between photos
KEY_REPEAT_GAP = 0.6  # s: a save key pressed again within this time is a held key repeating, not a new press
MIN_COVERED = 1 / 3  # one of the nine parts of the image counts as covered when saved boards cover this much of it
MIN_ESTIMATE = 4  # photos before the running calibration estimate starts
PREVIEW_WIDTH = 640  # px: width of the live picture on the screen (smaller on short screens); larger is slower on the Pi
SIDE_WIDTH = 360  # px: width of the board-positions panel and its explanation

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'calibration_images')  # the folder that "calibration.py --images calibration_images" reads
GREEN, ORANGE, RED, GREY = '#1a7f37', '#c25e00', '#e30613', '#555'
FAST = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
SUBPIX_WINDOW = (5, 5)  # as in calibration.py
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # as in calibration.py
CAMERA_NAMES = {'imx477': 'HQ camera', 'imx708': 'Camera Module 3', 'imx219': 'Camera Module 2',
                'ov5647': 'Camera Module 1', 'imx296': 'Global Shutter camera', 'imx500': 'AI camera'}
PARTS = (('top left', 'top', 'top right'), ('left', 'centre', 'right'), ('bottom left', 'bottom', 'bottom right'))

OBJP = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)  # the chessboard corners in mm
OBJP[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE


def find_corners(gray):
    """The inner corners (N x 2) in a full-size grey photo, found as calibration.py finds them, or None."""
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA).reshape(-1, 2)


def find_corners_live(small, photo_size):
    """The inner corners in the small grey copy of a picture, in pixels of the full-size photo, or None.

    This quick search is only for the live picture. CALIB_CB_FAST_CHECK gives up quickly when there is no board,
    so the live picture stays smooth.
    """
    found, corners = cv2.findChessboardCorners(small, CHESSBOARD_SIZE, flags=FAST)
    if not found:
        return None
    scale = np.array([photo_size[0] / small.shape[1], photo_size[1] / small.shape[0]])
    return (corners.reshape(-1, 2) + 0.5) * scale - 0.5  # + 0.5 and - 0.5: from pixel centre to pixel centre


def corner_motion(now, before):
    """How far the corners moved between two searches (px, the median, so that one jittery corner does not
    count), or None if one search found no board. OpenCV may number the corners from the other end of the
    board, so both orders are tried."""
    if now is None or before is None:
        return None
    return float(min(np.median(np.linalg.norm(now - before, axis=1)),
                     np.median(np.linalg.norm(now[::-1] - before, axis=1))))


def calibrate(views, size):
    """(K, D, rms) from the saved photos, with the settings of calibration.py (k3 fixed at 0)."""
    obj = [OBJP] * len(views)
    img = [v['image'].reshape(-1, 1, 2).astype(np.float32) for v in views]
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, None, None, flags=cv2.CALIB_FIX_K3)
    return K, D, rms


def undistortion_maps(K, D, size, view_size):
    """Maps for cv2.remap that undistort the small live picture (view_size) with an estimate made for full-size
    photos (size). All of the picture is kept, with black borders."""
    sx, sy = view_size[0] / size[0], view_size[1] / size[1]
    K_view = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                       [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5],
                       [0, 0, 1]])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K_view, D, view_size, 1, view_size)
    return cv2.initUndistortRectifyMap(K_view, D, None, new_K, view_size, cv2.CV_16SC2)


def preview_size(size, max_height):
    """Size (width, height) of the live picture on the screen: PREVIEW_WIDTH wide, or smaller on a short screen."""
    w, h = size
    live_h = int(round(h * PREVIEW_WIDTH / w))
    if live_h <= max_height:
        return PREVIEW_WIDTH, live_h
    return int(round(w * max_height / h)), max_height


def ppm(image):
    """A BGR picture as PPM data, a format that Tk can show without extra packages."""
    return b'P6 %d %d 255 ' % (image.shape[1], image.shape[0]) + cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()


def camera_label(info):
    """A name for the camera list, such as '0: imx477 (HQ camera)'."""
    model = info.get('Model', '?')
    known = CAMERA_NAMES.get(model)
    return f"{info.get('Num', 0)}: {model}" + (f" ({known})" if known else "")


class PiCamera:
    """The Raspberry Pi camera, with two streams made from the same pictures: the full-size photo ("main") and a
    small grey copy for the live search ("lores")."""

    def __init__(self, number):
        self.picam2 = Picamera2(number)
        try:
            # "RGB888" gives arrays in B, G, R order, which is the order OpenCV expects.
            # In "YUV420", the first rows of the array are the grey (Y) picture.
            config = self.picam2.create_video_configuration(
                main={"size": CAPTURE_SIZE, "format": "RGB888"},
                lores={"size": LIVE_SIZE, "format": "YUV420"},
                sensor=SENSOR or {})
            self.picam2.configure(config)
            self.sensor_mode = self.picam2.camera_configuration()["sensor"]
            self.picam2.start()
        except Exception:
            self.picam2.close()
            raise

    def read(self):
        """(photo, small grey copy) of the next picture."""
        (photo, small), _ = self.picam2.capture_arrays(["main", "lores"])
        return photo, small[:LIVE_SIZE[1], :LIVE_SIZE[0]]

    def release(self):
        self.picam2.stop()
        self.picam2.close()


class Camera:
    """Reads the camera in one background thread, and searches the newest picture for the board in another.

    The two threads keep the live picture smooth: a search that fails can take 0.2 s on the Pi (for example when
    only part of the board is in view), but the camera still delivers 30 pictures per second to the window.
    """

    def __init__(self, number, max_preview_height):
        self.number = number
        self.max_preview_height = max_preview_height
        self.lock = threading.Lock()
        self.picture = None  # the newest picture: {'frame', 'small', 'preview', 'number'}
        self.latest = None  # the newest searched picture: {'frame', 'number', 'found', 'motion', 'still'}
        self.state = 'opening'  # then 'running', or a message that says what went wrong
        self.sensor_mode = None
        self.running = True
        self.new_picture = threading.Event()
        self.threads = [threading.Thread(target=self.read_pictures, daemon=True),
                        threading.Thread(target=self.search_pictures, daemon=True)]

    def start(self):
        for thread in self.threads:
            thread.start()

    def read_pictures(self):
        try:
            cam = PiCamera(self.number)
        except Exception as error:  # for example "Device or resource busy" when another program uses the camera
            self.state = (f"Could not open camera {self.number} ({error}). Only one program can use the camera at a "
                          "time: close the other one (for example rpicam-hello or calibration_cap.py), then press "
                          "Start again.")
            return
        self.sensor_mode = cam.sensor_mode
        print("Sensor mode in use:", cam.sensor_mode)
        number = 0
        while self.running:
            try:
                photo, small = cam.read()
            except Exception as error:
                self.state = f"The camera stopped sending pictures ({error})."
                break
            self.state = 'running'
            # Shrink the photo for the live picture here, not in the thread of the window: this takes about 10 ms on
            # the Pi, and OpenCV lets the other threads run in the meantime, so the window stays responsive.
            size = preview_size((photo.shape[1], photo.shape[0]), self.max_preview_height)
            preview = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.picture = {'frame': photo, 'small': small, 'preview': preview, 'number': number}
            self.new_picture.set()
            number += 1
        cam.release()

    def search_pictures(self):
        """Search the newest picture, again and again. The motion is measured from one search to the next, so
        when a search was slow and pictures were skipped, the still check is stricter, never weaker."""
        previous, done = None, -1
        while self.running:
            self.new_picture.wait(0.5)
            self.new_picture.clear()
            with self.lock:
                picture = self.picture
            if picture is None or picture['number'] == done:
                continue
            done = picture['number']
            photo = picture['frame']
            try:
                found = find_corners_live(picture['small'], (photo.shape[1], photo.shape[0]))
            except Exception as error:  # say so, instead of showing "No chessboard" for ever
                self.state = f"The search for the chessboard stopped ({error!r}). Press Start again."
                return
            motion = corner_motion(found, previous)
            previous = found
            with self.lock:
                self.latest = {'frame': photo, 'number': picture['number'], 'found': found,
                               'motion': motion, 'still': motion is not None and motion <= MAX_MOTION}

    def newest(self):
        """The newest searched picture (the one that a save uses), or None."""
        with self.lock:
            return self.latest

    def newest_picture(self):
        """The newest picture from the camera (for the live picture), or None."""
        with self.lock:
            return self.picture

    def stop(self):
        self.running = False
        self.new_picture.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=3)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calibration capture")
        self.cameras = Picamera2.global_camera_info()  # one entry per camera: {'Model': 'imx477', 'Num': 0, ...}
        self.camera = None  # the running Camera
        self.shown = -1  # number of the picture on the screen; -1: check the next picture against the folder
        self.views = []  # every photo in the folder with the board in it: {'name', 'image', 'size'}
        self.found_cache = {}  # (path, modification time) -> (corners or None, size or None)
        self.loading = False  # the photos already in the folder are being searched in the background
        self.load_ready = False
        self.session = []  # full paths of the photos saved since the program started, for Undo
        self.still_since = None
        self.anchor = None  # board corners when the board stopped moving, to catch a slow drift
        self.saved_still = None  # still_since of the last save: one photo per still period, even if a key is held
        self.refused_board = None  # corners of the last board that the full-size search refused
        self.saved_number = None  # camera picture number of the last save: a picture is never saved twice
        self.last_auto = 0.0  # time of the last save, for Auto capture without the motion check
        self.last_save_key = 0.0  # time of the last Space or c key event, to ignore a held key's repeats
        self.last_saved = None  # board of the last saved photo, so that Auto capture does not save the same view twice
        self.checking = False  # a photo is being checked and saved in the background
        self.save_ready = None  # the result of that check, left for the main loop
        self.frame_times = []
        self.estimate = None  # (K, D, rms, number of photos, size, job) of the running calibration
        self.estimate_job = 0  # increases with every change, so that an outdated background result is dropped
        self.estimate_ready = None  # (job, result) left by the background calibration for the main loop
        self.estimate_count, self.estimate_size = 0, None
        self.undistort = (None, None)  # (key, maps) for the live picture
        self.panel = (None, None, None)  # (key, board-positions panel without the board in view, coverage)
        self.side_shown = (None, 0.0)  # (saved photos, time) of the panel on the screen
        self.status_shown = None  # (text, colour) of the status line on the screen
        self.fps_text, self.fps_time = "", 0.0  # the fps number changes once per second, not with every picture
        self.closing = False
        self.build()
        self.load_folder()
        self.protocol('WM_DELETE_WINDOW', self.quit_app)
        for key in ('<space>', 'c', 'C'):
            self.bind(key, lambda e: self.keyboard(self.save))
        for key in ('u', 'U'):
            self.bind(key, lambda e: self.keyboard(self.undo))
        for key in ('q', 'Q', '<Escape>'):
            self.bind(key, lambda e: self.keyboard(self.quit_app))
        self.after(30, self.tick)

    # ------------------------------------------------------------------ layout
    def build(self):
        pad = {'padx': 6, 'pady': 4}
        top = ttk.Frame(self, padding=8)
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Camera").grid(row=0, column=0, sticky='w', **pad)
        self.camera_box = ttk.Combobox(top, values=[camera_label(c) for c in self.cameras], width=36,
                                       state='readonly')
        if self.cameras:
            self.camera_box.current(0)
        self.camera_box.grid(row=0, column=1, columnspan=2, sticky='w', **pad)
        self.camera_box.bind('<<ComboboxSelected>>', lambda e: self.focus_set())
        self.start_button = ttk.Button(top, text="Start", command=self.start_stop, takefocus=0)
        self.start_button.grid(row=0, column=3, sticky='w', **pad)
        self.show_undistorted = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Undistorted view", variable=self.show_undistorted, takefocus=0).grid(
            row=0, column=4, sticky='w', **pad)

        ttk.Label(top, text="Folder").grid(row=1, column=0, sticky='w', **pad)
        ttk.Label(top, text=FOLDER).grid(row=1, column=1, columnspan=3, sticky='w', **pad)
        ttk.Button(top, text="Open folder", command=self.open_folder, takefocus=0).grid(row=1, column=4, **pad)

        # The rows under the picture are packed first, from the bottom up: a short screen cuts the picture, not them
        wrap = PREVIEW_WIDTH + SIDE_WIDTH
        self.message = tk.Label(self, anchor='w', justify='left', padx=10, pady=4, fg=GREY, font=('Helvetica', 12),
                                wraplength=wrap)
        self.message.pack(side='bottom', fill='x')
        self.result = tk.Label(self, anchor='w', justify='left', padx=10, font=('Helvetica', 13), wraplength=wrap)
        self.result.pack(side='bottom', fill='x')
        self.status = tk.Label(self, font=('Helvetica', 14), anchor='w', padx=10, width=1)  # width=1: never widens
        self.status.pack(side='bottom', fill='x')
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side='bottom', fill='x')
        self.save_button = tk.Button(bottom, text="Save photo  (Space)", command=self.save,
                                     font=('Helvetica', 16, 'bold'), padx=18, pady=6, takefocus=0)
        self.save_button.pack(side='left')
        ttk.Button(bottom, text="Undo last (U)", command=self.undo, takefocus=0).pack(side='left', padx=10)
        # The two switches for saving: untick "Motion check" to save without waiting until the board is still,
        # and tick "Auto capture" to let the program save by itself
        self.need_still = tk.BooleanVar(value=True)
        tk.Checkbutton(bottom, text="Motion check", variable=self.need_still, command=self.auto_changed,
                       font=('Helvetica', 13), takefocus=0).pack(side='left', padx=6)
        self.auto = tk.BooleanVar(value=False)
        tk.Checkbutton(bottom, text="Auto capture", variable=self.auto, command=self.auto_changed,
                       font=('Helvetica', 13), takefocus=0).pack(side='left', padx=6)
        self.counter = tk.Label(bottom, font=('Helvetica', 16, 'bold'))
        self.counter.pack(side='left', padx=16)

        # The live picture and the board-positions panel are two pictures, because the panel changes less often:
        # sending only the live picture to the screen for most pictures keeps the window responsive on the Pi.
        pictures = tk.Frame(self, bg='black', highlightthickness=5, highlightbackground='black')
        pictures.pack(side='top', padx=8)
        self.photo = tk.PhotoImage(width=PREVIEW_WIDTH, height=PREVIEW_WIDTH * 3 // 4)  # blank until Start
        self.preview = tk.Label(pictures, image=self.photo, bg='black', borderwidth=0)
        self.preview.pack(side='left', anchor='n')
        self.side_photo = tk.PhotoImage(width=SIDE_WIDTH, height=PREVIEW_WIDTH * 3 // 4)
        self.side = tk.Label(pictures, image=self.side_photo, bg='black', borderwidth=0)
        self.side.pack(side='left', anchor='n')
        self.pictures = pictures
        for widget in (pictures, self.preview, self.side):
            widget.bind('<Button-1>', lambda e: self.focus_set())

        # Fit the live picture on short screens: it shrinks, the panel text stays readable
        self.update_idletasks()
        others = self.winfo_reqheight() - pictures.winfo_reqheight()
        self.max_live_h = max(240, self.winfo_screenheight() - others - 120)  # 120: title bar and taskbar

        if self.cameras:
            self.say("Press Start. Then hold the chessboard in front of the camera.")
        else:
            self.start_button.configure(state='disabled')
            self.say("No Raspberry Pi camera found. Check the camera cable, and check that the command "
                     "rpicam-hello --list-cameras lists the camera.", RED)

    def say(self, text, colour=GREY):
        self.message.configure(text=text, fg=colour)

    def keyboard(self, action):
        if action == self.save:  # a held key repeats itself: one save per press
            now = time.time()
            held = now - self.last_save_key < KEY_REPEAT_GAP
            self.last_save_key = now
            if held:
                return
        action()

    # ------------------------------------------------------------------ camera
    def start_stop(self):
        if self.camera is None:
            self.start()
        else:
            self.stop()

    def start(self):
        number = self.cameras[self.camera_box.current()]['Num']
        self.load_folder()  # photos may have been deleted by hand since the last check
        self.camera = Camera(number, self.max_live_h)
        self.camera.start()
        self.saved_number = None  # the new camera numbers its pictures from 0 again
        self.shown = -1
        self.start_button.configure(text="Stop")
        self.camera_box.configure(state='disabled')  # read at Start: press Stop to change it
        self.focus_set()
        self.say(f"Opening camera {camera_label(self.cameras[self.camera_box.current()])} ...")

    def stop(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.start_button.configure(text="Start")
        self.camera_box.configure(state='readonly')

    def auto_changed(self):
        self.still_since = None

    # ------------------------------------------------------------------ folder
    @staticmethod
    def photo_files():
        """The photos in the folder, as calibration.py reads them (not the pictures that calibration.py saves)."""
        if not os.path.isdir(FOLDER):
            return []
        return sorted(f for f in os.listdir(FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                      and '_corners_detected' not in f and '_undistorted' not in f)

    @staticmethod
    def cache_key(path):
        try:
            return path, os.path.getmtime(path)
        except OSError:  # deleted in the meantime
            return None

    def load_folder(self):
        """Search the photos already in the folder for the board, for the dots, the positions panel and the
        estimate. A full-size search takes up to about a second per photo on the Pi, so it runs in the background."""
        if self.loading:
            return
        self.session = [p for p in self.session if os.path.exists(p)]  # photos deleted by hand cannot be undone
        keys = [self.cache_key(os.path.join(FOLDER, name)) for name in self.photo_files()]
        todo = [key for key in keys if key is not None and key not in self.found_cache]
        if not todo:
            self.folder_loaded()
            return
        self.loading = True
        self.say(f"Reading the {len(todo)} photo{'s' if len(todo) != 1 else ''} already in the folder ...")

        def work():
            for key in todo:
                img = cv2.imread(key[0])  # read as calibration.py reads it: in colour, then made grey
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
                self.found_cache[key] = (None, None) if gray is None else (find_corners(gray), gray.shape[::-1])
            self.load_ready = True  # picked up by tick(): Tk may only be used from the main thread

        threading.Thread(target=work, daemon=True).start()

    def folder_loaded(self):
        views, files, unreadable = [], self.photo_files(), 0
        for name in files:
            key = self.cache_key(os.path.join(FOLDER, name))
            if key not in self.found_cache:  # deleted or changed during the search
                continue
            corners, size = self.found_cache[key]
            unreadable += size is None
            if corners is not None:
                views.append({'name': name, 'image': corners, 'size': size})
        self.views = views
        if self.loading:
            self.loading = False
            if unreadable:
                self.say(f"{unreadable} photo(s) in the folder could not be read.", ORANGE)
            elif self.camera is None:
                self.say(f"The folder already has {len(files)} photo{'s' if len(files) != 1 else ''}, and the board "
                         f"is found in {len(views)}. Press Start to add more.")
            if self.camera is not None:
                self.shown = -1  # check the running camera against the photos of the folder
        self.update_counter()
        self.start_estimate()

    def open_folder(self):
        try:
            os.makedirs(FOLDER, exist_ok=True)
            opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
            subprocess.Popen([opener, FOLDER])
        except OSError as error:
            self.say(f"Could not open {FOLDER}: {error.strerror}.", RED)

    def update_counter(self):
        files = len(self.photo_files())
        extra = f"  ({files - len(self.views)} without the board)" if files > len(self.views) else ""
        self.counter.configure(text=f"{len(self.views)} / {TARGET} photos{extra}")

    # ------------------------------------------------------------------ running estimate
    def start_estimate(self):
        """Calibrate in the background with the photos so far (the window stays responsive)."""
        self.estimate_job += 1
        job = self.estimate_job
        sizes = [v['size'] for v in self.views]
        size = max(set(sizes), key=sizes.count) if sizes else None  # one camera at one size, as calibration.py
        views = [v for v in self.views if v['size'] == size]
        self.estimate_count, self.estimate_size = len(views), size
        if len(views) < MIN_ESTIMATE:
            self.estimate = None
            self.show_estimate()
            return

        def work():
            try:
                K, D, rms = calibrate(views, size)
                result = (K, D, rms, len(views), size, job)
            except cv2.error:
                result = None
            self.estimate_ready = (job, result)  # picked up by tick(): Tk may only be used from the main thread

        threading.Thread(target=work, daemon=True).start()

    def estimate_done(self, job, result):
        if job != self.estimate_job:  # a newer estimate is on its way
            return
        self.estimate = result
        self.show_estimate()

    def show_estimate(self):
        other = len(self.views) - self.estimate_count
        note = (f"  {other} photo{'s' if other != 1 else ''} of another size {'are' if other != 1 else 'is'} left out "
                f"(a calibration needs one size: {self.estimate_size[0]}x{self.estimate_size[1]})." if other else "")
        if self.estimate is None:
            more = max(0, MIN_ESTIMATE - self.estimate_count)
            text = (f"Estimate: after {more} more photo{'s' if more != 1 else ''}." if more else
                    "Estimate: the calibration failed with these photos; add photos with the board in other places.")
            self.result.configure(text=text + note, fg=GREY)
            return
        K, D, rms, n, _, _ = self.estimate
        fx, fy = K[0, 0], K[1, 1]
        text = (f"Estimate from {n} photos: fx {fx:.0f}, fy {fy:.0f}, cx {K[0, 2]:.0f}, cy {K[1, 2]:.0f} px; "
                f"k1 {D.ravel()[0]:.3f}; RMS error {rms:.2f} px")
        if not (fx > 0 and fy > 0 and 0.9 < fx / fy < 1.1):
            # The pixels of the camera are square, so fx and fy must be almost equal. If they are not, the photos
            # are too similar (for example all facing the camera), and the calibration cannot separate the values.
            self.result.configure(text=f"{text}. Not reliable yet: fx and fy should be almost equal. Add photos with "
                                       f"the board tilted and near the corners of the image (see the shot list).{note}",
                                  fg=RED)
            return
        # the same limits as "Check your own calibration" in the README
        colour, verdict = ((GREEN, "good") if rms < 0.5 else
                           (ORANGE, "usable: retake blurred photos or photos of a bent board") if rms < 1.0 else
                           (RED, "too high: a bent board, motion blur, or a change of focus?"))
        self.result.configure(text=f"{text} ({verdict}).{note}", fg=colour)

    # ------------------------------------------------------------------ saving
    def next_name(self):
        """calibration_image_NN.jpg, numbered after the photos already there (any case, any image type)."""
        numbers = [int(m.group(1)) for m in (re.fullmatch(r'calibration_image_(\d+)\.(?:jpe?g|png)', f, re.IGNORECASE)
                                             for f in self.photo_files()) if m]
        n = max(numbers) + 1 if numbers else 0
        taken = {f.lower() for f in os.listdir(FOLDER)} if os.path.isdir(FOLDER) else set()
        while f"calibration_image_{n:02d}.jpg" in taken:
            n += 1
        return f"calibration_image_{n:02d}.jpg"

    def save(self, auto=False, item=None):
        """Check the newest picture and, if it passes, save it in the background (save_done() reports)."""
        if self.checking:
            if not auto:
                self.say("Wait: the last photo is still being checked.", ORANGE)
            return False
        if self.loading:
            self.say("Wait: the program is still reading the photos already in the folder.", ORANGE)
            return False
        if item is None:
            item = self.camera.newest() if self.camera is not None else None
        if item is None:
            self.say("Press Start first.", RED)
            return False
        if item['found'] is None:
            self.say("No chessboard found: the whole board, with some of its white border, must be in view.", RED)
            return False
        if self.need_still.get():
            if not item['still'] or self.still_since is None or time.time() - self.still_since < READY_TIME:
                self.say("The board is moving: hold it still for a moment (or rest it on something), then save "
                         "again. To save without this check, untick Motion check.", ORANGE)
                return False
            if self.saved_still == self.still_since:
                self.say("Already saved: move or tilt the board to a new position, then save again.", ORANGE)
                return False
        elif item['number'] == self.saved_number:
            self.say("Already saved: this camera picture is in the folder already.", ORANGE)
            return False
        self.checking = True
        self.last_auto = time.time()
        self.save_button.configure(state='disabled')
        self.say("Checking the full-size photo in the same way as calibration.py ...")
        still_since = self.still_since

        def work():
            # Make the JPEG file, read it back in colour and make it grey, as calibration.py will read it, and search
            # it in the same way. Only a photo in which calibration.py will find the board is saved.
            ok, jpeg = cv2.imencode('.jpg', item['frame'])
            corners = name = error = None
            if ok:
                corners = find_corners(cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY))
            if corners is not None:
                name = self.next_name()
                try:
                    os.makedirs(FOLDER, exist_ok=True)
                    with open(os.path.join(FOLDER, name), 'wb') as file:
                        file.write(jpeg.tobytes())
                except OSError as problem:
                    error = problem.strerror
            self.save_ready = (item, still_since, auto, corners, name, error)  # picked up by tick()

        threading.Thread(target=work, daemon=True).start()
        return True

    def save_done(self, item, still_since, auto, corners, name, error):
        self.checking = False
        self.save_button.configure(state='normal')
        self.last_auto = time.time()
        if corners is None:
            self.refused_board = item['found']  # Auto capture tries again only after the board has moved a little
            self.say("Not saved: calibration.py would not find the board in this full-size photo. The board may be "
                     "blurred, too small, too tilted, or partly hidden by a reflection. Move the board a little and "
                     "try again.", RED)
            return
        if error is not None:
            if auto:
                self.auto.set(False)  # so that it does not try again on every picture
            self.say(f"Could not save {name} in {FOLDER} ({error}). Check that the disk (the SD card) is not full"
                     f"{'; Auto capture is now off' if auto else ''}.", RED)
            return
        path = os.path.join(FOLDER, name)
        size = (item['frame'].shape[1], item['frame'].shape[0])
        key = self.cache_key(path)
        if key is not None:
            self.found_cache[key] = (corners, size)
        self.views.append({'name': name, 'image': corners, 'size': size})
        self.session.append(path)
        self.last_saved, self.saved_still = item['found'], still_since
        self.saved_number = item['number']
        empty = self.empty_parts(size)
        text = f"Saved {name}{' (Auto capture)' if auto else ''}."
        text += (f" More photos are needed in: {', '.join(empty)}." if empty else
                 " Every part of the image is covered now.")
        if len(self.views) >= TARGET and not empty:
            text += " You have enough photos: press q, then run python calibration.py --images calibration_images"
        self.say(text, GREEN)
        self.update_counter()
        self.start_estimate()
        self.flash()

    def undo(self):
        if self.checking:
            self.say("Wait: the last photo is still being checked.", ORANGE)
            return
        if not self.session:
            self.say("Nothing to undo: Undo only removes photos saved since the program started.", ORANGE)
            return
        path = self.session.pop()
        name = os.path.basename(path)
        if os.path.exists(path):
            os.remove(path)
        removed = [v for v in self.views if v['name'] == name]
        self.views = [v for v in self.views if v['name'] != name]
        if self.last_saved is None:  # otherwise keep the newest view, so that Auto capture does not save it straight back
            self.last_saved = removed[0]['image'] if removed else None
        self.saved_still = self.saved_number = None  # by hand, the view can be saved again
        self.say(f"Removed {name}.", ORANGE)
        self.update_counter()
        self.start_estimate()

    def flash(self):
        self.pictures.configure(highlightbackground='white')
        self.after(150, lambda: self.pictures.configure(highlightbackground='black'))

    def moved_since_refusal(self, found):
        """False while the board is (almost) where it was in the last photo that the full-size search refused,
        so that Auto capture does not try the same photo again and again."""
        motion = corner_motion(found, self.refused_board)
        return motion is None or motion > MAX_DRIFT

    def new_position(self, found, size):
        """True if the board moved away from where it was in the last saved photo."""
        motion = corner_motion(found, self.last_saved)
        return motion is None or motion > NEW_POSITION * np.hypot(*size)

    # ------------------------------------------------------------------ drawing
    def coverage(self, size):
        """(panel, parts) for the saved photos of this size. panel: every saved board filled in green, brighter
        where boards overlap (as in OpenCV's interactive calibration app), and the parts that no saved board
        covers yet in red. parts: 3 x 3 array, the fraction of each of the nine parts of the image that the saved
        boards cover. Both are made again only when the photos change."""
        w, h = size
        pw, ph = SIDE_WIDTH, int(round(h * SIDE_WIDTH / w))
        key = (size, tuple(v['name'] for v in self.views))
        if self.panel[0] != key:
            panel = np.zeros((ph, pw, 3), np.uint8)
            covered = np.zeros((ph, pw), np.uint8)
            for view in self.views:
                if view['size'] != size:
                    continue
                fill = np.zeros_like(panel)
                hull = cv2.convexHull((view['image'] * (pw / w, ph / h)).astype(np.int32))
                cv2.fillConvexPoly(fill, hull, (0, 255, 0), cv2.LINE_AA)
                cv2.fillConvexPoly(covered, hull, 255)
                panel = cv2.addWeighted(fill, 0.2, panel, 1.0, 0)
            panel[covered == 0] = (0, 0, 120)  # dark red: no board there yet, so the calibration knows nothing about it
            parts = np.array([[(covered[r * ph // 3:(r + 1) * ph // 3, c * pw // 3:(c + 1) * pw // 3] > 0).mean()
                               for c in range(3)] for r in range(3)])
            # the image split in thirds: a calibration needs the board in each of the nine parts
            for k in (1, 2):
                cv2.line(panel, (k * pw // 3, 0), (k * pw // 3, ph), (90, 90, 90), 1)
                cv2.line(panel, (0, k * ph // 3), (pw, k * ph // 3), (90, 90, 90), 1)
            self.panel = (key, panel, parts)
        return self.panel[1], self.panel[2]

    def positions_panel(self, size, current):
        """The board-positions panel, with the board in view now as a white outline."""
        panel = self.coverage(size)[0].copy()
        if current is not None:
            scale = (panel.shape[1] / size[0], panel.shape[0] / size[1])
            hull = cv2.convexHull((current * scale).astype(np.int32))
            cv2.polylines(panel, [hull], True, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    def empty_parts(self, size):
        """The names of the nine parts of the image (3 x 3) that saved boards do not cover enough yet."""
        parts = self.coverage(size)[1]
        return [PARTS[row][column] for row in range(3) for column in range(3) if parts[row, column] < MIN_COVERED]

    def tick(self):
        self.after(30, self.tick)
        if self.estimate_ready is not None:
            job, result = self.estimate_ready
            self.estimate_ready = None
            self.estimate_done(job, result)
        if self.load_ready:
            self.load_ready = False
            self.folder_loaded()
        if self.save_ready is not None:
            result, self.save_ready = self.save_ready, None
            self.save_done(*result)
        if self.camera is None:
            return
        if self.camera.state not in ('opening', 'running'):
            self.say(self.camera.state, RED)
            self.stop()
            return
        picture = self.camera.newest_picture()
        if picture is None or picture['number'] == self.shown:
            return
        item = self.camera.newest()  # the newest searched picture: the board and the still check come from it
        h, w = picture['frame'].shape[:2]
        if self.shown == -1 and not self.loading:  # first picture, or the folder was read again
            sizes = {v['size'] for v in self.views}
            if sizes and sizes != {(w, h)}:
                listed = ", ".join(f"{a}x{b}" for a, b in sorted(sizes))
                self.say(f"The photos in {FOLDER} are {listed}, but the camera gives {w}x{h}. A calibration needs "
                         "one camera at one size: move the photos of the other size out of the folder, then press "
                         "Start again.", RED)
                self.stop()
                return
            mode = self.camera.sensor_mode or {}
            mode_text = (f" in the {mode['output_size'][0]}x{mode['output_size'][1]} sensor mode"
                         if 'output_size' in mode else "")
            self.say(f"Capturing {w}x{h} photos{mode_text}. Use the calibration only for videos of this size and "
                     "sensor mode, and do not turn the focus ring from now on.")
        self.shown = picture['number']
        now = time.time()
        self.frame_times = [t for t in self.frame_times if now - t < 1.0] + [now]

        # Status and Auto capture
        found = item['found'] if item is not None else None
        corners = f"{len(found)} corners" if found is not None else ""
        can_auto = self.auto.get() and not self.checking and not self.loading
        if found is None:
            text, colour = "No chessboard", RED
            self.still_since = None
        elif not self.need_still.get():  # no motion check: hold the board steady yourself
            self.still_since = None
            text, colour = f"{corners} - ready (Motion check is off: hold the board steady)", GREEN
            if can_auto and now - self.last_auto >= AUTO_INTERVAL and self.moved_since_refusal(found):
                if self.new_position(found, (w, h)):
                    self.save(auto=True, item=item)
                else:
                    text = f"{corners} - move the board to a new position"
        elif not item['still']:
            moving = f" (moving {item['motion']:.1f} px per picture, needs under {MAX_MOTION:g})" if item['motion'] else ""
            text, colour = f"{corners} - hold still{moving}", ORANGE
            self.still_since = None
        else:
            drift = corner_motion(found, self.anchor)
            if self.still_since is None or drift is None or drift > MAX_DRIFT:
                self.still_since, self.anchor = now, found  # (re)start the still period here
            if now - self.still_since < READY_TIME:
                text, colour = f"{corners} - hold still", ORANGE
            elif self.saved_still == self.still_since:
                text, colour = f"{corners} - saved: move the board to a new position", GREEN
            else:
                text, colour = f"{corners} - ready", GREEN
            if (can_auto and now - self.still_since >= STILL_TIME and self.saved_still != self.still_since
                    and self.moved_since_refusal(found)):
                if self.new_position(found, (w, h)):
                    self.save(auto=True, item=item)
                else:
                    text = f"{corners} - move the board to a new position"
        empty = len(self.empty_parts((w, h)))
        coverage = (f"   ·   {empty} of 9 image parts need more photos" if empty else
                    "   ·   every part of the image is covered")
        if now - self.fps_time >= 1.0:
            self.fps_text, self.fps_time = f"{len(self.frame_times)} fps", now
        status = (f"{text}{coverage}   ·   {self.fps_text}", colour)
        if status != self.status_shown:  # changing a label costs time on the Pi, so only when the text changes
            self.status.configure(text=status[0], fg=status[1])
            self.status_shown = status

        # Live picture: the small copy that the camera thread made, with the saved corners as green dots and the
        # board in view with OpenCV's coloured corners (from the newest search, which can be a few pictures older
        # than the live picture)
        view = picture['preview'].copy()
        live_h, live_w = view.shape[:2]
        sx, sy = live_w / w, live_h / h
        for saved in self.views:
            if saved['size'] == (w, h):
                for x, y in saved['image']:
                    cv2.circle(view, (int(x * sx), int(y * sy)), 2, (0, 255, 0), 1, cv2.LINE_AA)
        if found is not None:
            cv2.drawChessboardCorners(view, CHESSBOARD_SIZE, (found * (sx, sy)).reshape(-1, 1, 2).astype(np.float32),
                                      True)
        if self.show_undistorted.get() and self.estimate is not None and self.estimate[4] == (w, h):
            key = (self.estimate[5], live_w, live_h)
            if self.undistort[0] != key:
                self.undistort = (key, undistortion_maps(self.estimate[0], self.estimate[1], (w, h), (live_w, live_h)))
            view = cv2.remap(view, *self.undistort[1], cv2.INTER_LINEAR)
            label = "Undistorted with the estimate"
        else:
            label = f"Camera {camera_label(self.cameras[self.camera_box.current()])}, {w}x{h}"
        cv2.rectangle(view, (0, 0), (live_w, 30), (0, 0, 0), -1)
        cv2.putText(view, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        self.photo = tk.PhotoImage(data=ppm(view), format='PPM')  # keep a reference, or Tk drops the picture
        self.preview.configure(image=self.photo)

        # The panel: at once when the saved photos change, otherwise at most 5 times per second (for the white outline)
        names = tuple(v['name'] for v in self.views)
        if names != self.side_shown[0] or now - self.side_shown[1] >= 0.2:
            self.side_shown = (names, now)
            panel = self.positions_panel((w, h), found)
            lines = ("Red: no saved photo covers it yet.", "Green: the saved photos (brighter",
                     "where they overlap). White: the board", "now. Move the board into the red,",
                     "especially the corners.")
            side = np.zeros((max(live_h, panel.shape[0] + 52 + 22 * len(lines)), SIDE_WIDTH, 3), np.uint8)
            side[:panel.shape[0]] = panel
            cv2.putText(side, "Board positions", (10, panel.shape[0] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            for i, line in enumerate(lines):
                cv2.putText(side, line, (10, panel.shape[0] + 52 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (190, 190, 190), 1, cv2.LINE_AA)
            self.side_photo = tk.PhotoImage(data=ppm(side), format='PPM')
            self.side.configure(image=self.side_photo)

    def quit_app(self):
        if self.closing:
            return
        self.closing = True
        self.stop()
        print(f"{len(self.views)} photos with the chessboard in {FOLDER}.")
        print("Next step: python calibration.py --images calibration_images")
        self.destroy()


def main():
    if Picamera2 is None:
        sys.exit(f"Picamera2 is not available ({PICAMERA2_ERROR}). This program needs the Raspberry Pi camera: run "
                 "it on the Raspberry Pi, in the comp0219 environment. On a laptop, practise with calibration_cap.py, "
                 "which uses the webcam.")
    try:
        app = App()
    except tk.TclError as error:  # no desktop, for example over SSH or in the Pi Connect remote shell
        sys.exit(f"Could not open a window ({error}). This program needs the Raspberry Pi desktop: run it in a "
                 "terminal on the Pi's monitor or in the screen sharing of Raspberry Pi Connect. Over SSH or in the "
                 "remote shell, use calibration_cap.py instead, which has a terminal mode.")
    # Ctrl+C in the terminal closes the window and releases the camera (tick() gives Python the chance to react)
    signal.signal(signal.SIGINT, lambda *_: app.after_idle(app.quit_app))
    app.mainloop()


if __name__ == '__main__':
    main()
