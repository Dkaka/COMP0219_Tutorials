"""
Wind Tunnel Logger - student edition (COMP0219 2026/2027).

Live plot and logging of a student airspeed sensor against the TecQuipment VDAS wind tunnel's own VDAS-F / AFA5
pressure cell as ground truth. Same as the developer edition (windspeed-vdas_GUI) without the scoring: runs
are saved as CSV files and scored later by the module team.

    wind-tunnel-logger                normal use after pip install (settings are in the window)
    python -m wind_tunnel_logger      same
    WindTunnelLogger.exe              Windows exe build (build_exe.ps1)
    wind-tunnel-logger --demo         simulated tunnel and simulated student (no hardware)

The tunnel's VDAS box is read directly with hidapi (vdas_afa5.py), which also sends the keep-alive the box needs
to stream: TecQuipment's VDAS software isn't needed (and should be closed). hidapi works on Windows, macOS and
Linux; it has been tested on Windows.

Student board: one speed in m/s per line ("3.42" or "...,3.42") on a USB serial port. The WindSpeedGiga
prototype's data line (seq,t_us,dp_raw_pa,dp_pa,v_mps,temp_c,flags) is also understood.

A WindSpeedGiga pitot tube (Arduino GIGA R1, see the windspeed-giga repo) can also be the ground truth: while one is
plugged in, "Ground truth" offers it. It boots with LEQ 0 (readings 50-60 % low), so its LEQ and C are sent after
every #HELLO and its readings are used only once its #CFG line confirms them. It zeroes itself on the board.
"""
import argparse
import bisect
import collections
import csv
import datetime
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from wind_tunnel_logger import __version__ as VERSION

APP_NAME = "Wind Tunnel Logger"
ORG = "UCL COMP0219"
HERE = Path(__file__).resolve().parent          # package folder (also inside the exe)
IS_WINDOWS = sys.platform == "win32"


def _settings_dir():
    if IS_WINDOWS:
        return Path(os.environ.get("APPDATA") or Path.home()) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "wind-tunnel-logger"


APPDATA = _settings_dir()
R_D = 287.05                 # specific gas constant of dry air, J/(kg K)
STALE_S = 2.0                # no sample for this long -> "NO DATA" in the status line
GIGA_VID = 0x2341
GIGA_HOLD, GIGA_OVR, GIGA_CLIP, GIGA_ZEROING, GIGA_RESYNC = 1, 8, 16, 64, 256
RATE_WINDOW_S = 5.0          # live student rate is averaged over this long
GT_COLOR = (0, 200, 220)
ST_COLOR = (230, 70, 230)
LOG_HEADER = [
    "timestamp_iso", "timestamp_epoch", "t_rel_s", "device_name", "port", "value_raw_mps",
    "value_clipped_mps", "expected_error_pct", "cov_lower_mps", "cov_upper_mps", "pa_raw", "pa_corrected",
]

DEFAULT_CONFIG = {
    "vdas": {"zero_seconds": 5.0, "max_offset_pa": 6.0, "fallback_offset_pa": 3.0, "dp_error_pa": 1.0,
             "cell": 1},
    "air": {"p_amb_pa": 101325.0, "air_temp_c": 20.0, "air_density": None},
    "student": {"name": "Student", "port": "", "baud": 115200, "rate_hz": 200.0},
    "giga": {"send_settings": True, "leq_m": 102.177, "c": 1.0, "rate_hz": 200, "zero_on_connect": True,
             "gt_rate_hz": 100, "gt_dp_error_pa": 0.08, "gt_speed_error_pct": 3.0},
    "ui": {"window_seconds": 15.0, "max_points": 3000, "refresh_ms": 30, "min_speed": -2.0,
           "max_speed": 100.0, "error_speed_floor": 0.2, "mean_seconds": 10.0},
    "logging": {"align_tolerance_ms": 150.0},
}

HELP_TEXT = f"""<h3>{APP_NAME} {VERSION}</h3>
<p>Compares your airspeed sensor with the wind tunnel's own reading and saves both as CSV files.</p>
<ol>
<li>With the <b>fan OFF</b>, plug the tunnel's ground truth into this laptop by USB and pick it as
    <i>Ground truth</i>: on the <b>white tunnel</b> its VDAS box (<i>Wind tunnel</i>), on the <b>wood tunnel</b> the
    GIGA (<i>GIGA pitot tube</i>). Nothing else is needed: if TecQuipment's VDAS software is open, close it.</li>
<li>When the tunnel data arrives (a few seconds), the program zeroes it: wait until the status line shows
    <i>tunnel offset</i> (or <i>GIGA zero</i>).</li>
<li>Enter today's <b>pressure</b> and <b>air temperature</b>, and your <b>team</b> name.</li>
<li>Plug in your board, pick its <b>port</b> (COMx on Windows, /dev/tty.usbmodem... on a Mac) and press
    <b>Connect</b>.
    Your board must print <b>one speed in m/s per line</b>, at <b>200 Hz</b>, <b>115200 baud</b>.
    The rate you achieve is shown under your reading.</li>
<li>Start the fan, press <b>Start Logging</b>, run your test, then <b>Stop &amp; Save</b>.</li>
<li><b>Plot run</b> shows any saved run. <b>Open logs folder</b> shows the files.</li>
</ol>
<p><b>No tunnel?</b> Set <i>Ground truth</i> to <i>Simulated (bench test)</i> to check your board's
connection, rate and logging at home. The simulated speed is not real wind.</p>
<p><b>GIGA pitot tube</b> (wood tunnel): offered while one is plugged in. The program sends it its calibration
(LEQ) and zeroes it on the board.</p>
"""


def resource(name):
    return HERE / name


def load_config(path):
    """Advanced settings. On first run a commented copy of the packaged config.yaml is put in the settings
    folder (%APPDATA%\\Wind Tunnel Logger on Windows)."""
    if not path.exists() and resource("config.yaml").exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resource("config.yaml"), path)
        except OSError:
            pass
    cfg = {section: dict(values) for section, values in DEFAULT_CONFIG.items()}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for section, values in user.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
    return cfg


def cmd_number(x):
    """Number as the GIGA command parser accepts it: plain decimal, no exponent."""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def align(gt, st, tol_s):
    """Pair each VDAS sample with the nearest student sample within tol_s.
    gt, st: lists of (epoch_s, mps). Returns [(epoch_s, gt_mps, student_mps or None)]."""
    st = sorted(st)
    st_t = [t for t, _ in st]
    out = []
    for t, v in gt:
        i = bisect.bisect_left(st_t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(st_t) and abs(st_t[j] - t) <= tol_s:
                if best is None or abs(st_t[j] - t) < abs(st_t[best] - t):
                    best = j
        out.append((t, v, st[best][1] if best is not None else None))
    return out


def iso(t):
    return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def run_base(path):
    """logs/<name> from any of the run's files (<name>_aligned.csv, _student.csv, _vdas.csv)."""
    p = Path(path)
    name = p.name[:-4] if p.name.lower().endswith(".csv") else p.name
    for suffix in ("_aligned", "_student", "_vdas"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return p.with_name(name)


def read_csv_columns(path, *cols):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [np.array([float(r[c]) if r.get(c, "") not in ("", None) else np.nan for r in rows]) for c in cols]


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")


def gt_device(path):
    """device_name of a saved run's ground truth: "VDAS" (the tunnel, or simulated) or "GIGA"."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return next(csv.DictReader(f), {}).get("device_name") or "VDAS"
    except OSError:
        return "VDAS"


def evt_fields(line):
    """key=value fields of a GIGA metadata line ("#EVT zero_ok zero_pa=0.0312 ..." -> {"zero_pa": "0.0312", ...})."""
    return dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)


def no_si_prefix(plot_item):
    """Keep axes in m/s and s: pyqtgraph would otherwise relabel small ranges as mm/s or ms.
    Call it before plotting: disabling later keeps whatever prefix was already chosen."""
    for side in ("left", "bottom"):
        plot_item.getAxis(side).enableAutoSIPrefix(False)


# ---------------------------------------------------------------- ground truth sources

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def vdas_software_running():
    """True if TecQuipment's VDAS software is open (it isn't needed, and it also drives the VDAS box)."""
    if not IS_WINDOWS:
        return False
    try:
        # explicit stdin: the windowless exe has no valid standard handles to inherit
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq VDAS.exe", "/NH"], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "vdas.exe" in out.lower()


class VdasSource(QtCore.QObject):
    """The tunnel's VDAS box (TecQuipment VDAS-F + AFA5 cell) over USB HID (vdas_afa5.GroundTruthReader, which keeps
    the box streaming with its keep-alive: no VDAS software needed). Reconnects after a USB drop or power cut."""
    reading = QtCore.Signal(float, float)   # epoch s, raw Pa (VDAS offset not removed)
    status = QtCore.Signal(str)
    port_name = "VDAS-HID"

    def __init__(self, parent=None, cell=1):
        super().__init__(parent)
        self._reader = None
        self._cell = int(cell)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._check)
        self._last_frame = 0.0              # last reading of our cell (reader thread writes, GUI thread reads)
        self._connected_at = 0.0

    def start(self):
        self._check()
        self._timer.start(2000)

    def _on_reading(self, r):   # called on the reader thread; the signal is queued to the GUI thread
        self._last_frame = time.time()
        if r["valid"]:
            self.reading.emit(r["t"], float(r["pa"]))

    def _explain_silence(self):
        """Box open but no readings of our cell: say why, as far as we can tell."""
        now = time.time()
        if now - self._last_frame < 3.0 or now - self._connected_at < 3.0:
            return                          # data is flowing, or the box was only just opened
        if now - self._reader.last_report < 3.0:
            self.status.emit(f"the VDAS box sends data, but none from pressure cell {self._cell} - check the "
                             f"AFA5 module's instance switch is set to {self._cell}")
        elif vdas_software_running():
            self.status.emit("the VDAS box sends no data - close TecQuipment's VDAS software (it isn't needed), "
                             "then unplug and replug the box's USB cable")
        else:
            self.status.emit("the VDAS box sends no data - check the tunnel's instrument power, then unplug and "
                             "replug the box's USB cable")

    def _check(self):
        if self._reader is not None:
            if self._reader.is_plugged():
                self._explain_silence()
                return
            self._close_reader()
            self.status.emit("tunnel's VDAS box unplugged - reconnecting")
        try:
            from wind_tunnel_logger.vdas_afa5 import GroundTruthReader
        except ImportError:
            self._timer.stop()
            self.status.emit("hidapi is missing - reinstall wind-tunnel-logger")
            return
        try:
            self._reader = GroundTruthReader(self._cell)
            self._reader.add_listener(self._on_reading)
            self._connected_at = time.time()
            self.status.emit("tunnel's VDAS box connected - waiting for data")
        except Exception as e:
            self._reader = None
            hint = " - close TecQuipment's VDAS software (it isn't needed)" if vdas_software_running() else ""
            self.status.emit(f"{e}{hint}; retrying - check its USB cable and the tunnel's instrument power")

    def _close_reader(self):
        try:
            self._reader.close()
        except Exception:
            pass
        self._reader = None

    def stop(self):
        self._timer.stop()
        if self._reader is not None:
            self._close_reader()


_DEMO_T0 = time.time()


def demo_speed(t):
    """Synthetic tunnel run: 8 s fan off (covers the zero), then 12 s steps with 3 s ramps, repeating."""
    t -= _DEMO_T0
    if t < 8:
        return 0.0
    levels = [3.0, 6.0, 9.0, 10.0, 6.0, 2.0]
    k, frac = divmod(t - 8, 12.0)
    k = int(k)
    cur = levels[k % len(levels)]
    prev = levels[(k - 1) % len(levels)] if k > 0 else 0.0
    return prev + (cur - prev) * frac / 3.0 if frac < 3.0 else cur


class FakeVdas(QtCore.QThread):
    """Simulated VDAS: whole-Pa readings at ~5 Hz, negative with the fan running, +2.5 Pa offset."""
    reading = QtCore.Signal(float, float)
    status = QtCore.Signal(str)
    port_name = "SIMULATED-VDAS"

    def __init__(self, parent=None):
        super().__init__(parent)
        global _DEMO_T0
        _DEMO_T0 = time.time()      # restart the profile, so the zero sees "fan off"
        self._stop = False

    def run(self):
        self.status.emit("SIMULATED tunnel (bench test, not real wind)")
        while not self._stop:
            t = time.time()
            v = demo_speed(t)
            pa = -0.5 * 1.204 * v * v + 2.5 + random.gauss(0, 0.6)
            self.reading.emit(t, float(round(pa)))
            self.msleep(200)

    def stop(self):
        self._stop = True
        self.wait(1000)


# ---------------------------------------------------------------- student sources

class SerialLines(QtCore.QThread):
    """Lines from a serial port, handed to _handle(); reopens the port if it drops.

    Reads in chunks rather than readline(): pyserial's readline() fetches one byte per call, which is slow
    enough at 200 Hz to fall behind and make a good board look like it drops samples."""
    status = QtCore.Signal(str)

    def __init__(self, port, baud, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = 115200 if int(baud) == 1200 else int(baud)     # 1200 baud reboots a GIGA into its bootloader
        self._stop = False

    def _sleep(self, s):
        end = time.time() + s
        while not self._stop and time.time() < end:
            self.msleep(50)

    def run(self):
        while not self._stop:
            try:
                ser = serial.Serial(self.port, self.baud, timeout=0.5)
            except Exception as e:
                self.status.emit(f"cannot open {self.port}: {e}")
                self._sleep(2.0)
                continue
            self.status.emit(f"{self.port} open")
            with ser:
                self._read(ser)
            self._sleep(1.0)

    def _read(self, ser):
        self._opened()
        buf = b""
        while not self._stop:
            try:
                self._tick(ser)
                chunk = ser.read(ser.in_waiting or 1)   # waits up to the timeout for the first byte
                if not chunk:
                    continue
                now = time.time()
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    self._handle(raw.decode(errors="ignore").strip(), now, ser)
            except Exception as e:
                self.status.emit(f"{self.port} lost: {e}")
                return
            if len(buf) > 4096:                         # no newline for 4 kB: garbage, resync
                buf = b""
                self._garbage()

    def _opened(self):
        """The port was (re)opened."""

    def _tick(self, ser):
        """Called before every read (at least every 0.5 s): time to write anything pending."""

    def _garbage(self):
        """4 kB without a newline were dropped."""

    def _handle(self, line, now, ser):
        raise NotImplementedError

    def stop(self):
        self._stop = True
        self.wait(1500)


class StudentSerial(SerialLines):
    """Student board on a serial port."""
    sample = QtCore.Signal(float, float)    # epoch s, m/s
    temp = QtCore.Signal(float)             # GIGA sensor temperature, degC

    def __init__(self, port, baud, giga_cmds, parent=None):
        super().__init__(port, baud, parent)
        self._giga_cmds = giga_cmds         # callable -> list of commands to send after a GIGA #HELLO
        self._zero_sent = False
        self.bad_lines = 0                  # non-empty lines that are not a number (read by the GUI thread)
        self.last_bad = ""
        self._last_seq = None

    def _opened(self):
        self._hello = self._asked_info = False

    def _garbage(self):
        self.bad_lines += 1

    def _handle(self, line, now, ser):
        if not line:
            return
        if line.startswith("#"):
            if line.startswith("#HELLO") and "GIGA" in line:
                self._hello = True
                self._last_seq = None
                self.status.emit(f"WindSpeedGiga on {self.port}")
                self._send_giga_settings(ser)
            elif line.startswith("#EVT zero_"):
                self.status.emit(f"GIGA {line[5:]}")
            return
        fields = line.split(",")
        try:
            if len(fields) == 7:        # WindSpeedGiga data line
                if not self._hello and not self._asked_info:
                    self._asked_info = True
                    ser.write(b"INFO\n")    # replays #HELLO so the settings get sent
                if int(fields[6]) & (GIGA_HOLD | GIGA_ZEROING):
                    return
                self.temp.emit(float(fields[5]))
                value = float(fields[4])
            else:
                value = float(fields[-1])
        except ValueError:
            self.bad_lines += 1
            self.last_bad = line[:40]
            return
        self.sample.emit(now, value)

    def _send_giga_settings(self, ser):
        cmds = list(self._giga_cmds())
        if cmds and cmds[-1] == "ZERO":
            if self._zero_sent:
                cmds.pop()
            self._zero_sent = True
        for c in cmds:
            ser.write((c + "\n").encode())
        if cmds:
            self.status.emit(f"WindSpeedGiga on {self.port}: sent {', '.join(cmds)}")


class GigaSource(SerialLines):
    """A WindSpeedGiga pitot tube (Arduino GIGA R1) as the ground truth (protocol: docs/protocol.md of the
    windspeed-giga repo). Its dp_pa column is the pressure after its zero, altitude and tube-loss (LEQ) corrections.

    The board keeps its settings in RAM only: after a power cut it boots with LEQ 0 and reads 50-60 % low. So the
    settings are sent after every #HELLO, and readings count only while its last #CFG shows our LEQ and C."""
    reading = QtCore.Signal(float, float, float, bool)  # epoch s, dp_raw_pa, dp_pa, usable (calibrated, not zeroing)
    zeroed = QtCore.Signal(bool, str)                   # zero accepted?, the #EVT line
    on_board_zero = True
    ZERO_RETRY_S = 6.0          # a zero takes 2.2 s: no answer by then, send ZERO again
    SETTLE_S = 0.3              # rows already queued under the old settings can follow the confirming #CFG
    RESEND_S = 2.0              # its #CFG still shows another LEQ or C this long after we sent ours: send again

    def __init__(self, port, settings, leq_m, c, parent=None):
        super().__init__(port, 115200, parent)
        self.port_name = port
        self._settings = settings           # callable -> commands to send after every #HELLO
        self.want = (round(float(leq_m), 3), round(float(c), 3))     # as its #CFG prints them
        self._calibration = [f"LEQ {cmd_number(leq_m)}", f"C {cmd_number(c)}"]
        self._sent_at = 0.0
        self.ready = False                  # the board confirmed our LEQ and C (read by the GUI thread)
        self.over_range = 0.0               # time of the last reading above the sensor's range
        self._ready_from = 0.0
        self._hello = self._asked_info = False
        self._outbox = collections.deque()  # commands from the GUI thread, written by this thread
        self._zero_pa = None                # last accepted zero: given back if the board resets
        self._want_zero = False
        self._zero_sent = 0.0

    def send(self, cmds):
        """Queue commands (GUI thread); they go out once the board has said #HELLO."""
        self._outbox.extend(cmds)

    def zero(self):
        self._want_zero = True
        self._zero_sent = 0.0

    @staticmethod
    def _write(ser, cmds):
        for c in cmds:
            ser.write((c + "\n").encode())

    def _opened(self):
        self._hello = self._asked_info = self.ready = False

    def _tick(self, ser):
        if not self._hello:                 # the board drops commands sent before its #HELLO
            return
        while self._outbox:
            self._write(ser, [self._outbox.popleft()])
        if self._want_zero and time.time() - self._zero_sent > self.ZERO_RETRY_S:
            self._write(ser, ["ZERO"])
            self._zero_sent = time.time()

    def _handle(self, line, now, ser):
        if line.startswith("#"):
            self._meta(line, now, ser)
            return
        fields = line.split(",")
        if len(fields) != 7:
            return
        try:
            flags, raw, pa = int(fields[6]), float(fields[2]), float(fields[3])
        except ValueError:
            return
        if not self._hello and not self._asked_info:
            self._asked_info = True
            ser.write(b"INFO\n")            # its #HELLO was missed: ask again, so the settings get sent
        if flags & GIGA_HOLD:
            return
        if flags & (GIGA_OVR | GIGA_CLIP):
            self.over_range = now
        usable = (self.ready and now >= self._ready_from
                  and not flags & (GIGA_ZEROING | GIGA_OVR | GIGA_CLIP))
        self.reading.emit(now, raw, pa, usable)

    def _meta(self, line, now, ser):
        if line.startswith("#HELLO"):
            self._hello, self.ready = True, False
            cmds = list(self._settings())
            if self._zero_pa is not None and not self._want_zero:     # the board reset: give it its zero back
                cmds.append(f"ZERO SET {cmd_number(self._zero_pa)}")
            self._write(ser, cmds)
            self._sent_at = now
            self.status.emit(f"GIGA pitot tube on {self.port}: sent {', '.join(cmds)}")
        elif line.startswith("#CFG"):
            f = evt_fields(line)
            try:
                have = (round(float(f["leq_m"]), 3), round(float(f["c"]), 3))
            except (KeyError, ValueError):
                return
            if have == self.want and not self.ready:
                self._ready_from = now + self.SETTLE_S
                self.status.emit(f"GIGA pitot tube on {self.port}: LEQ {have[0]:g} m, C {have[1]:g} set")
            elif have != self.want:
                self.status.emit(f"GIGA pitot tube on {self.port} has LEQ {have[0]:g} m, C {have[1]:g} - sending "
                                 f"LEQ {self.want[0]:g} m, C {self.want[1]:g}")
                if now - self._sent_at > self.RESEND_S:     # changed behind our back: put ours back
                    self._write(ser, self._calibration)
                    self._sent_at = now
            self.ready = have == self.want
        elif line.startswith("#EVT zero_ok") or line.startswith("#EVT zero_reject"):
            ok = line.startswith("#EVT zero_ok")
            if ok:
                try:
                    self._zero_pa = float(evt_fields(line)["zero_pa"])
                except (KeyError, ValueError):
                    pass
            self._want_zero = False
            self.zeroed.emit(ok, line[5:])
        elif line.startswith("#EVT sensor_fault") or (line.startswith("#SENSOR") and "state=missing" in line):
            self.status.emit(f"GIGA pitot tube: its pressure sensor isn't answering ({line[1:]}) - check the 4 "
                             "sensor wires")
        elif line.startswith("#ERR") and not line.startswith("#ERR ZERO reason=busy"):
            self.status.emit(f"GIGA pitot tube on {self.port} refused a command: {line[1:]}")


class FakeStudent(QtCore.QThread):
    """Simulated student sensor at rate_hz (only for --demo / the smoke test)."""
    sample = QtCore.Signal(float, float)
    temp = QtCore.Signal(float)
    status = QtCore.Signal(str)
    port = "SIMULATED-STUDENT"

    def __init__(self, parent=None, rate_hz=200.0):
        super().__init__(parent)
        self.rate_hz = rate_hz
        self._stop = False
        self.bad_lines = 0
        self.last_bad = ""

    def run(self):
        self.status.emit(f"SIMULATED student ({self.rate_hz:g} Hz)")
        v, sent, t_start = 0.0, 0, time.time()
        while not self._stop:
            self.msleep(10)             # Windows sleeps ~1-16 ms; emit however many samples are due
            due = int((time.time() - t_start) * self.rate_hz)
            for _ in range(due - sent):
                t = time.time()
                v += 0.05 * (0.95 * demo_speed(t) + 0.2 - v)
                self.sample.emit(t, v + random.gauss(0, 0.15))
            sent = due

    def stop(self):
        self._stop = True
        self.wait(1000)


# ---------------------------------------------------------------- saved-run plot

class RunPlotWindow(QtWidgets.QWidget):
    """A saved run: speed over time (VDAS + student), student vs VDAS, and error vs VDAS."""

    def __init__(self, base, student_name="Student"):
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Run {base.name}")
        self.resize(1150, 800)
        f_vdas, f_st, f_al = (base.with_name(base.name + s) for s in ("_vdas.csv", "_student.csv", "_aligned.csv"))
        if not f_al.exists():
            raise FileNotFoundError(f"not found: {f_al}")
        tv, vv = read_csv_columns(f_vdas, "timestamp_epoch", "value_raw_mps") if f_vdas.exists() else ([], [])
        ts, vs = read_csv_columns(f_st, "timestamp_epoch", "value_raw_mps") if f_st.exists() else ([], [])
        gt, st = read_csv_columns(f_al, "ground_truth_mps", "student_mps")
        t0 = min([a[0] for a in (tv, ts) if len(a)], default=0.0)
        ok = ~np.isnan(gt) & ~np.isnan(st)
        giga = gt_device(f_vdas) == "GIGA"
        ref = "GIGA pitot" if giga else "Tunnel"
        ref_lc = ref if giga else "tunnel"

        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel(f"{base}\n{len(tv)} {ref_lc} readings, {len(ts)} samples from "
                                f"{student_name.lower()}, {int(ok.sum())} paired")
        head.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(head)
        glw = pg.GraphicsLayoutWidget()
        lay.addWidget(glw, 1)
        dash = pg.mkPen((200, 200, 200), style=QtCore.Qt.PenStyle.DashLine)

        p1 = glw.addPlot(row=0, col=0, colspan=2, title="Speed over time")
        no_si_prefix(p1)
        p1.addLegend()
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setLabel("left", "Speed", units="m/s")
        p1.setLabel("bottom", "Time", units="s")
        if len(ts):
            p1.plot(np.asarray(ts) - t0, vs, pen=pg.mkPen(ST_COLOR, width=1), name=student_name)
        if len(tv):     # the tunnel's few readings a second as dots; the GIGA's 100 a second as a line
            dots = {} if giga else dict(symbol="o", symbolSize=4, symbolBrush=GT_COLOR, symbolPen=None)
            p1.plot(np.asarray(tv) - t0, vv, pen=pg.mkPen(GT_COLOR, width=2), name=f"{ref} ground truth", **dots)

        lim = max(float(np.nanmax(gt[ok])) if ok.any() else 10.0, float(np.nanmax(st[ok])) if ok.any() else 10.0) + 1
        p2 = glw.addPlot(row=1, col=0, title=f"{student_name} vs {ref_lc}")
        no_si_prefix(p2)
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel("left", student_name, units="m/s")
        p2.setLabel("bottom", ref, units="m/s")
        p2.addLegend()
        p2.plot([0, lim], [0, lim], pen=dash, name="1:1")
        p2.plot(gt[ok], st[ok], pen=None, symbol="o", symbolSize=5, symbolBrush=(*ST_COLOR, 180), symbolPen=None,
                name="paired samples")

        p3 = glw.addPlot(row=1, col=1, title=f"Error ({student_name} - {ref_lc}) vs {ref_lc}")
        no_si_prefix(p3)
        p3.showGrid(x=True, y=True, alpha=0.3)
        p3.setLabel("left", "Error", units="m/s")
        p3.setLabel("bottom", ref, units="m/s")
        p3.addLine(y=0, pen=dash)
        p3.plot(gt[ok], (st - gt)[ok], pen=None, symbol="o", symbolSize=5, symbolBrush=(*ST_COLOR, 180),
                symbolPen=None)
        for p in (p2, p3):
            p.setXRange(0, lim, padding=0)


# ---------------------------------------------------------------- window

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg, settings, log_dir, simulated=False, fake_student=False):
        super().__init__()
        self.cfg = cfg
        self.settings = settings                # QSettings: values the student sets in the window
        self.log_dir = Path(log_dir)
        ui = cfg["ui"]
        self.window_s = float(ui["window_seconds"])
        self.min_speed = float(ui["min_speed"])
        self.max_speed = float(ui["max_speed"])
        self.err_floor = float(ui["error_speed_floor"])
        self.mean_s = float(ui["mean_seconds"])
        self.target_hz = float(cfg["student"].get("rate_hz") or 200.0)
        self.student_name = cfg["student"].get("name") or "Student"
        self.t0 = time.time()

        # enough points for the whole plot window at the target rate
        n = max(int(ui["max_points"]), int(self.window_s * self.target_hz * 1.3))
        self.gt_t, self.gt_v = collections.deque(maxlen=n), collections.deque(maxlen=n)
        self.st_t, self.st_v = collections.deque(maxlen=n), collections.deque(maxlen=n)
        self.last_vdas = self.last_st = None
        self.giga_temp = None
        self.offset = None
        self.zeroing = False
        self.zero_samples = []
        self.zero_end = 0.0
        self.zero_note = ""
        self.logging = False
        self.log_start = 0.0
        self.rows_gt, self.rows_st = [], []
        self.pairs_gt, self.pairs_st = [], []
        self.vdas_status = ""
        self.student_status = ""
        self.student_reader = None
        self.student_port = ""
        self.last_run = None            # base path of the run saved last (default for Plot run)
        self._run_windows = []          # open Plot run windows (kept referenced so they stay open)
        self.vdas = None                # the ground truth source: VdasSource, FakeVdas or GigaSource
        self.gt_key = ""                # its key in cmb_source: "tunnel", "sim" or "giga:<port>"
        self.gt_readout = None          # (m/s, detail) for the readout, shown by refresh()

        self._build_ui()
        self.cmb_source.setCurrentIndex(1 if simulated else 0)
        self.set_source("sim" if simulated else "tunnel")
        self.cmb_source.currentIndexChanged.connect(lambda i: self.set_source(self.cmb_source.itemData(i)))
        self.giga_timer = QtCore.QTimer(self)       # offer a GIGA as ground truth as soon as it is plugged in
        self.giga_timer.timeout.connect(self.update_giga_choices)
        self.giga_timer.start(3000)

        if fake_student:
            for w in (self.cmb_port, self.btn_ports, self.btn_connect):
                w.setEnabled(False)
            self._start_student(FakeStudent(self, self.target_hz))
        elif self.settings.value("port", ""):
            self.toggle_student()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(ui["refresh_ms"]))

    # ---- layout

    def _build_ui(self):
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1250, 700)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 8, 8, 0)
        self.btn_log = QtWidgets.QPushButton("Start Logging")
        self.btn_log.clicked.connect(self.toggle_logging)
        self.lbl_log = QtWidgets.QLabel("Logging: OFF")
        self.lbl_log.setStyleSheet("color: #666;")
        self.btn_zero = QtWidgets.QPushButton("Re-zero tunnel (fan OFF)")
        self.btn_zero.clicked.connect(self.begin_zero)
        self.btn_plot_run = QtWidgets.QPushButton("Plot run…")
        self.btn_plot_run.setToolTip("Plot a saved run: the tunnel and your sensor over time, and against each other")
        self.btn_plot_run.clicked.connect(lambda: self.plot_saved_run())
        self.btn_open_logs = QtWidgets.QPushButton("Open logs folder")
        self.btn_open_logs.setToolTip(str(self.log_dir))
        self.btn_open_logs.clicked.connect(self.open_logs_folder)
        self.btn_help = QtWidgets.QPushButton("Help")
        self.btn_help.clicked.connect(self.show_help)
        for w in (self.btn_log, self.lbl_log, self.btn_zero, self.btn_plot_run, self.btn_open_logs, self.btn_help):
            bar.addWidget(w)
        bar.addStretch(1)
        self.cmb_port = QtWidgets.QComboBox()
        self.cmb_port.setMinimumWidth(280)
        self.btn_ports = QtWidgets.QPushButton("Refresh")
        self.btn_ports.clicked.connect(self.refresh_ports)
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_student)
        bar.addWidget(QtWidgets.QLabel("Your board:"))
        for w in (self.cmb_port, self.btn_ports, self.btn_connect):
            bar.addWidget(w)
        main.addLayout(bar)

        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(8, 4, 8, 0)
        self.cmb_source = QtWidgets.QComboBox()
        self.cmb_source.addItem("Wind tunnel", "tunnel")
        self.cmb_source.addItem("Simulated (bench test)", "sim")
        self.cmb_source.setToolTip("Simulated: no tunnel needed, to check your board's connection, rate and "
                                   "logging. The simulated speed is not real wind.\nGIGA pitot tube: offered while "
                                   "one is plugged in.")
        self.txt_team = QtWidgets.QLineEdit(str(self.settings.value("team", "")))
        self.txt_team.setPlaceholderText("e.g. team3")
        self.txt_team.setMaximumWidth(140)
        self.txt_team.editingFinished.connect(lambda: self.settings.setValue("team", self.txt_team.text().strip()))
        air = self.cfg["air"]
        self.spn_p = QtWidgets.QDoubleSpinBox()
        self.spn_p.setRange(800.0, 1100.0)
        self.spn_p.setDecimals(1)
        self.spn_p.setSuffix(" hPa")
        self.spn_p.setValue(float(air["p_amb_pa"]) / 100.0)
        self.spn_p.setToolTip("Today's air pressure (station pressure), from a barometer or a weather service")
        self.spn_p.valueChanged.connect(self._on_air_changed)
        self.spn_t = QtWidgets.QDoubleSpinBox()
        self.spn_t.setRange(-10.0, 45.0)
        self.spn_t.setDecimals(1)
        self.spn_t.setSuffix(" °C")
        t = air.get("air_temp_c")
        self.spn_t.setValue(float(t) if isinstance(t, (int, float)) else 20.0)
        self.spn_t.setToolTip("Air temperature in the lab")
        self.spn_t.valueChanged.connect(self._on_air_changed)
        row2.addWidget(QtWidgets.QLabel("Ground truth:"))
        row2.addWidget(self.cmb_source)
        row2.addSpacing(12)
        for label, w in (("Team:", self.txt_team), ("Air pressure:", self.spn_p), ("Air temperature:", self.spn_t)):
            row2.addWidget(QtWidgets.QLabel(label))
            row2.addWidget(w)
            row2.addSpacing(12)
        row2.addStretch(1)
        main.addLayout(row2)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(8, 8, 8, 0)
        self.lbl_gt_title, self.lbl_gt_val, self.lbl_gt_sub = self._readout(header, "Tunnel ground truth", GT_COLOR)
        _, self.lbl_st_val, self.lbl_st_sub = self._readout(header, "Your sensor", ST_COLOR)
        header.addStretch(1)
        main.addLayout(header)

        self.lbl_status = QtWidgets.QLabel("starting")
        self.lbl_status.setContentsMargins(8, 0, 8, 0)
        self.lbl_status.setWordWrap(True)
        main.addWidget(self.lbl_status)

        self.plot = pg.PlotWidget()
        main.addWidget(self.plot, 1)
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.getViewBox().enableAutoRange(x=False, y=False)
        self.plot.setLabel("left", "Speed", units="m/s")
        self.plot.setLabel("bottom", "Time", units="s")
        no_si_prefix(self.plot.getPlotItem())
        self.c_gt = self.plot.plot(pen=pg.mkPen(GT_COLOR, width=2), name="Tunnel ground truth")
        dash = pg.mkPen(GT_COLOR, style=QtCore.Qt.PenStyle.DashLine)
        self.c_lo = self.plot.plot(pen=dash)
        self.c_hi = self.plot.plot(pen=dash)
        self.plot.addItem(pg.FillBetweenItem(self.c_lo, self.c_hi, brush=pg.mkBrush(*GT_COLOR, 60)))
        self.c_st = self.plot.plot(pen=pg.mkPen(ST_COLOR, width=2), name="Your sensor")
        self.plot.setYRange(self.min_speed, self.min_speed + 10, padding=0)
        self.refresh_ports()

    def _readout(self, layout, title, color):
        w = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(w)
        box.setContentsMargins(8, 4, 24, 4)
        t = QtWidgets.QLabel(title)
        t.setStyleSheet("color: rgb(%d,%d,%d); font-size: 12pt; font-weight: 600;" % color)
        v = QtWidgets.QLabel("—")
        v.setStyleSheet("font-size: 24pt; font-weight: 600;")
        s = QtWidgets.QLabel("")
        s.setStyleSheet("color: #666;")
        for x in (t, v, s):
            box.addWidget(x)
        layout.addWidget(w)
        return t, v, s

    def _help_box(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"{APP_NAME} - help")
        box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        box.setText(HELP_TEXT + f"<p>Logs are saved in:<br>{self.log_dir}</p>")
        return box

    def show_help(self):
        self._help_box().exec()

    def open_logs_folder(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.log_dir)))

    # ---- settings set in the window

    def _on_air_changed(self):
        self.cfg["air"]["p_amb_pa"] = self.spn_p.value() * 100.0
        self.cfg["air"]["air_temp_c"] = self.spn_t.value()
        self.settings.setValue("p_amb_pa", self.cfg["air"]["p_amb_pa"])
        self.settings.setValue("air_temp_c", self.cfg["air"]["air_temp_c"])
        if isinstance(self.vdas, GigaSource):       # its altitude and tube-loss corrections use them too
            self.vdas.send(self._giga_air())

    # ---- ground truth source

    def _show_source(self, key):
        self.cmb_source.blockSignals(True)
        self.cmb_source.setCurrentIndex(max(0, self.cmb_source.findData(key)))
        self.cmb_source.blockSignals(False)

    def set_source(self, key):
        """key: "tunnel", "sim" or "giga:<port>"."""
        giga_port = key[5:] if key.startswith("giga:") else None
        if self.logging:                    # never switch mid-run
            self._show_source(self.gt_key)
            QtWidgets.QMessageBox.information(self, APP_NAME, "Stop logging before changing the ground truth.")
            return
        if giga_port and self.student_reader is not None and giga_port == self.student_port:
            self._show_source(self.gt_key)
            QtWidgets.QMessageBox.information(self, APP_NAME, f"{giga_port} is connected as Your board: "
                                                              "disconnect it first.")
            return
        if self.vdas is not None:
            self.vdas.stop()
        if giga_port:
            g = self.cfg["giga"]
            self.vdas = GigaSource(giga_port, self._giga_gt_settings, g["leq_m"], g["c"], self)
            self.vdas.reading.connect(self.on_giga)
            self.vdas.zeroed.connect(self._on_board_zero)
            self.gt_name, self.gt_the, self.gt_device = "GIGA", "the GIGA pitot tube", "GIGA"
            self.pitot_c = float(g["c"])
            self.dp_err, self.speed_err_pct = float(g["gt_dp_error_pa"]), float(g["gt_speed_error_pct"])
            title, legend = f"GIGA pitot tube ground truth ({giga_port})", "GIGA pitot ground truth"
        else:
            self.vdas = FakeVdas(self) if key == "sim" else VdasSource(self, int(self.cfg["vdas"].get("cell") or 1))
            self.vdas.reading.connect(self.on_vdas)
            self.gt_name, self.gt_the, self.gt_device = "tunnel", "the tunnel", "VDAS"
            self.pitot_c = 1.0
            self.dp_err, self.speed_err_pct = float(self.cfg["vdas"]["dp_error_pa"]), 0.0
            title = "SIMULATED ground truth (not real wind)" if key == "sim" else "Tunnel ground truth"
            legend = "Tunnel ground truth"
        self.vdas.status.connect(self._on_vdas_status)
        self.gt_key = key
        self.gt_t.clear()
        self.gt_v.clear()
        self.last_vdas = None
        self.offset = None
        self.vdas_status = ""
        self.gt_readout = None
        self.lbl_gt_val.setText("—")
        self.lbl_gt_sub.setText("")
        self.lbl_gt_title.setText(title)
        self.plot.getPlotItem().legend.getLabel(self.c_gt).setText(legend)
        self.btn_zero.setText(f"Re-zero {'GIGA' if giga_port else 'tunnel'} (fan OFF)")
        marks = {"sim": "   [SIMULATED GROUND TRUTH]", "tunnel": ""}
        self.setWindowTitle(f"{APP_NAME} {VERSION}" + marks.get(key, "   [GROUND TRUTH: GIGA PITOT TUBE]"))
        self.refresh_ports()                # the GIGA isn't also offered as Your board
        self.begin_zero()
        self.vdas.start()

    def update_giga_choices(self, ports=None):
        """Offer each plugged-in GIGA (unless it is Your board) as the ground truth. The selected one stays listed
        when unplugged, so its reader keeps retrying."""
        if ports is None:
            if self.logging:                # no change possible while logging: skip the 10-30 ms port scan
                return
            ports = serial.tools.list_ports.comports()
        board = self.student_port if self.student_reader is not None else None
        keys = [f"giga:{p.device}" for p in sorted(ports, key=lambda p: p.device)
                if p.vid == GIGA_VID and p.device != board]
        if self.gt_key.startswith("giga:") and self.gt_key not in keys:
            keys.append(self.gt_key)
        if keys == [self.cmb_source.itemData(i) for i in range(2, self.cmb_source.count())]:
            return
        self.cmb_source.blockSignals(True)
        while self.cmb_source.count() > 2:
            self.cmb_source.removeItem(2)
        for k in keys:
            self.cmb_source.addItem(f"GIGA pitot tube ({k[5:]})", k)
        self.cmb_source.blockSignals(False)
        self._show_source(self.gt_key)

    # ---- air density

    def rho(self):
        air = self.cfg["air"]
        if air.get("air_density"):
            return float(air["air_density"])
        t = air.get("air_temp_c")
        if not isinstance(t, (int, float)):
            t = self.giga_temp if self.giga_temp is not None else 20.0
        return float(air["p_amb_pa"]) / (R_D * (float(t) + 273.15))

    def speed_error(self, v):
        """Uncertainty of a ground-truth speed v (m/s, > 0; also arrays): its pressure uncertainty as speed, plus the
        GIGA's calibration uncertainty in % of the speed."""
        return self.dp_err / (self.rho() * v) + self.speed_err_pct / 100.0 * v

    # ---- student port

    def refresh_ports(self):
        current = self.cmb_port.currentData() or self.settings.value("port", "") or ""
        ground_truth = self.gt_key[5:] if self.gt_key.startswith("giga:") else None
        if current == ground_truth:
            current = ""
        ports = sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)
        self.cmb_port.clear()
        for p in ports:
            if p.device == ground_truth:
                continue
            tag = "  <- GIGA R1" if p.vid == GIGA_VID else ""
            self.cmb_port.addItem(f"{p.device} - {p.description}{tag}", p.device)
        if current and self.cmb_port.findData(current) < 0:
            self.cmb_port.addItem(f"{current} (not present)", current)
        i = self.cmb_port.findData(current)
        if i >= 0:
            self.cmb_port.setCurrentIndex(i)
        self.update_giga_choices(ports)

    def toggle_student(self):
        if self.student_reader is not None:
            self.student_reader.stop()
            self.student_reader = None
            self.btn_connect.setText("Connect")
            self.cmb_port.setEnabled(True)
            self.btn_ports.setEnabled(True)
            self.student_status = "disconnected"
            self.update_giga_choices()
            return
        port = self.cmb_port.currentData()
        if not port:
            self.student_status = "no port selected"
            return
        self.settings.setValue("port", port)
        self._start_student(StudentSerial(port, self.cfg["student"].get("baud", 115200), self._giga_commands, self))
        self.btn_connect.setText("Disconnect")
        self.cmb_port.setEnabled(False)
        self.btn_ports.setEnabled(False)
        self.update_giga_choices()

    def _start_student(self, reader):
        reader.sample.connect(self.on_student)
        reader.temp.connect(self._on_giga_temp)
        reader.status.connect(self._on_student_status)
        self.student_port = reader.port
        self.student_reader = reader
        self.st_t.clear()
        self.st_v.clear()
        self.last_st = None
        reader.start()

    # ---- GIGA settings

    def _giga_air(self):
        air = self.cfg["air"]
        t = air.get("air_temp_c")
        return [f"P {cmd_number(air['p_amb_pa'])}",
                f"TAIR {cmd_number(t)}" if isinstance(t, (int, float)) else "TAIR OFF"]

    def _giga_settings(self, rate_hz):
        """Settings a GIGA needs after every #HELLO. Always with LEQ: the board boots with LEQ 0 (no tube-loss
        correction), and then reads 50-60 % low."""
        g = self.cfg["giga"]
        p, tair = self._giga_air()
        cmds = [p, f"LEQ {cmd_number(g['leq_m'])}", f"C {cmd_number(g['c'])}", tair]
        if rate_hz:
            cmds.append(f"RATE {int(rate_hz)}")
        return cmds

    def _giga_commands(self):
        """For a GIGA as Your board."""
        g = self.cfg["giga"]
        if not g.get("send_settings"):
            return []
        cmds = self._giga_settings(g.get("rate_hz"))
        if g.get("zero_on_connect"):
            cmds.append("ZERO")
        return cmds

    def _giga_gt_settings(self):
        """For a GIGA as the ground truth (called on its reader thread). Its zero comes from begin_zero()."""
        return self._giga_settings(self.cfg["giga"].get("gt_rate_hz"))

    @QtCore.Slot(str)
    def _on_student_status(self, text):
        self.student_status = text

    @QtCore.Slot(str)
    def _on_vdas_status(self, text):
        self.vdas_status = text

    @QtCore.Slot(float)
    def _on_giga_temp(self, t):
        self.giga_temp = t

    # ---- ground truth zero

    def begin_zero(self):
        self.zeroing = True
        self.zero_samples = []
        self.zero_note = ""
        self.btn_zero.setEnabled(False)
        if getattr(self.vdas, "on_board_zero", False):
            self.zero_end = math.inf        # the GIGA zeroes itself: _on_board_zero() ends it
            self.vdas.zero()
        else:
            self.zero_end = time.time() + float(self.cfg["vdas"]["zero_seconds"])

    def _finish_zero(self):
        v = self.cfg["vdas"]
        if not self.zero_samples:           # no VDAS data yet: keep waiting
            self.zero_end = time.time() + float(v["zero_seconds"])
            return
        mean = sum(self.zero_samples) / len(self.zero_samples)
        if abs(mean) > float(v["max_offset_pa"]):
            self.offset = float(v["fallback_offset_pa"])
            self.zero_note = (f"The tunnel read {mean:+.1f} Pa during the zero: the fan was running. Using "
                              f"{self.offset:+g} Pa instead - stop the fan and press Re-zero tunnel.")
        else:
            self.offset = mean
        self.zeroing = False
        self.btn_zero.setEnabled(True)

    @QtCore.Slot(bool, str)
    def _on_board_zero(self, ok, text):
        """The GIGA's #EVT zero_ok / zero_reject (text without the "#EVT ")."""
        f = evt_fields(text)
        if ok:
            self.offset = float(f.get("zero_pa", 0.0))
        else:
            why = ("its pressure sensor didn't answer - check the 4 sensor wires" if f.get("reason") in
                   ("count", "sensor") else f"the air wasn't still (mean {f.get('mean_pa', '?')} Pa)")
            self.zero_note = (f"The GIGA rejected its zero: {why}. It keeps its previous zero (fine above about 4 "
                              "m/s) - stop the fan, or join its two tubes, and press Re-zero GIGA.")
        self.zeroing = False
        self.btn_zero.setEnabled(True)

    # ---- data

    def _clip(self, v):
        return min(max(v, self.min_speed), self.max_speed)

    def _row(self, t, name, port, value, gt, pa_raw="", pa=""):
        clipped = self._clip(value)
        err_pct = lo = hi = ""
        vabs = abs(value)
        if gt and vabs > 1e-12:
            dv = self.speed_error(vabs)
            err_pct = f"{dv / vabs * 100:.3f}"
            lo = f"{max(self.min_speed, clipped - dv):.6f}"
            hi = f"{clipped + dv:.6f}"
        return [iso(t), f"{t:.6f}", f"{t - self.t0:.6f}", name, port, f"{value:.6f}", f"{clipped:.6f}",
                err_pct, lo, hi, pa_raw, pa]

    @QtCore.Slot(float, float)
    def on_vdas(self, t, pa_raw):
        self.last_vdas = t
        if self.zeroing:
            if not self.zero_samples:       # the zero's 5 s start at the first reading: the VDAS software may
                self.zero_end = time.time() + float(self.cfg["vdas"]["zero_seconds"])   # be opened later
            self.zero_samples.append(pa_raw)
            return
        pa = pa_raw - self.offset
        self._ground_truth(t, pa, f"dp {pa:+.1f} Pa (raw {pa_raw:+.0f}, offset {self.offset:+.2f})",
                           f"{pa_raw:.0f}", f"{pa:.2f}")

    @QtCore.Slot(float, float, float, bool)
    def on_giga(self, t, pa_raw, pa, usable):   # ~100 times a second
        self.last_vdas = t
        if usable and not self.zeroing:
            self._ground_truth(t, pa, f"dp {pa:+.3f} Pa (raw {pa_raw:+.3f}, LEQ and zero applied on the GIGA)",
                               f"{pa_raw:.4f}", f"{pa:.4f}")

    def _ground_truth(self, t, pa, detail, pa_raw_text, pa_text):
        v = self.pitot_c * math.sqrt(2.0 * abs(pa) / self.rho())
        self.gt_t.append(t - self.t0)
        self.gt_v.append(self._clip(v))
        self.gt_readout = (self._clip(v), detail)
        if self.logging:
            self.rows_gt.append(self._row(t, self.gt_device, self.vdas.port_name, v, True, pa_raw_text, pa_text))
            self.pairs_gt.append((t, v))

    @QtCore.Slot(float, float)
    def on_student(self, t, v):         # up to 200+ times a second: labels are updated in refresh()
        self.last_st = t
        self.st_t.append(t - self.t0)
        self.st_v.append(self._clip(v))
        if self.logging:
            self.rows_st.append(self._row(t, self.student_name, self.student_port, v, False))
            self.pairs_st.append((t, v))

    # ---- plot + status

    def refresh(self):
        now = time.time()
        if self.zeroing and now >= self.zero_end:
            self._finish_zero()
        if self.gt_readout is not None:
            v, detail = self.gt_readout
            self.lbl_gt_val.setText(f"{v:.3f} m/s")
            self.lbl_gt_sub.setText(detail)
        tg, vg = np.array(self.gt_t), np.array(self.gt_v)
        ts, vs = np.array(self.st_t), np.array(self.st_v)
        self.c_gt.setData(tg, vg)
        if tg.size:
            dv = self.speed_error(np.maximum(np.abs(vg), self.err_floor))
            self.c_lo.setData(tg, np.maximum(vg - dv, self.min_speed))
            self.c_hi.setData(tg, vg + dv)
        else:
            self.c_lo.setData([], [])
            self.c_hi.setData([], [])
        self.c_st.setData(ts, vs)
        if ts.size:
            self._update_student_readout(now, ts, vs)
        latest = max((a[-1] for a in (tg, ts) if a.size), default=None)
        if latest is not None:
            start = max(0.0, latest - self.window_s)
            self.plot.setXRange(start, latest, padding=0)
            top = max((float(v[t >= start].max()) for t, v in ((tg, vg), (ts, vs)) if (t >= start).any()),
                      default=None)
            if top is not None:
                self.plot.setYRange(self.min_speed, max(top + 1.0, self.min_speed + 1.0), padding=0)
        self._update_status(now, tg, vg, ts, vs)

    def _update_student_readout(self, now, ts, vs):
        self.lbl_st_val.setText(f"{vs[-1]:.3f} m/s")
        t_rel = now - self.t0
        recent = ts[ts >= t_rel - RATE_WINDOW_S]
        span = min(RATE_WINDOW_S, t_rel - ts[0])
        text = self.student_port
        bad = False
        if self.giga_temp:
            text += f"   T {self.giga_temp:.1f} C"
        if span > 1.0:
            rate = len(recent) / span
            text += f"   {rate:.1f} Hz (target {self.target_hz:g})"
            bad = rate < 0.98 * self.target_hz
        r = self.student_reader
        if r is not None and r.bad_lines:
            text += f"   {r.bad_lines} lines not a number (last: {r.last_bad!r})"
            bad = True
        self.lbl_st_sub.setText(text)
        self.lbl_st_sub.setStyleSheet("color: #c00;" if bad else "color: #666;")

    def _update_status(self, now, tg, vg, ts, vs):
        lines, bad = [], False
        giga = self.vdas if isinstance(self.vdas, GigaSource) else None
        if self.last_vdas is None or now - self.last_vdas > STALE_S:
            hint = ""
            if self.gt_key == "tunnel" and self.cmb_source.count() > 2:
                hint = " - on the wood tunnel, pick Ground truth: GIGA pitot tube"
            lines.append(f"NO DATA from {self.gt_the} - {self.vdas_status or 'waiting'}{hint}")
            bad = True
        elif giga and not giga.ready:
            lines.append(f"The GIGA pitot tube hasn't confirmed LEQ {giga.want[0]:g} m and C {giga.want[1]:g} yet: its "
                         f"readings aren't used ({self.vdas_status or 'waiting'})")
            bad = True
        elif giga and now - giga.over_range < 1.0:
            lines.append("The GIGA pitot tube is above its sensor's range (125 Pa, about 14 m/s): those readings "
                         "aren't used")
            bad = True
        if self.zeroing and giga:
            lines.append("Waiting for the GIGA's data to zero it - keep the fan OFF" if self.last_vdas is None else
                         "Zeroing the GIGA pitot tube (on the board, about 2 s) - keep the fan OFF")
        elif self.zeroing and not self.zero_samples:
            lines.append("Waiting for the tunnel's data to zero it - keep the fan OFF")
        elif self.zeroing:
            lines.append(f"Zeroing the tunnel reading - keep the fan OFF ({max(0.0, self.zero_end - now):.1f} s left, "
                         f"{len(self.zero_samples)} readings)")
        elif self.zero_note:
            lines.append(self.zero_note)
            bad = True
        if self.student_reader is None:
            lines.append(f"Your board: not connected - pick its port and press Connect "
                         f"({self.student_status or 'idle'})")
        elif self.last_st is None or now - self.last_st > STALE_S:
            lines.append(f"NO DATA from your board - {self.student_status or 'waiting'}")
            bad = True
        cut = now - self.t0 - self.mean_s
        mg, ms = vg[tg >= cut], vs[ts >= cut]
        if mg.size and ms.size:
            g, s = mg.mean(), ms.mean()
            diff = f"{s - g:+.2f} m/s ({(s / g - 1) * 100:+.1f} %)" if g > 0.5 else "n/a"
            lines.append(f"last {self.mean_s:g} s mean:   {self.gt_name} {g:.2f} m/s   your sensor {s:.2f} m/s   "
                         f"difference {diff}")
        offset = "-" if self.offset is None else f"{self.offset:+.4f} Pa" if giga else f"{self.offset:+.2f} Pa"
        lines.append(f"air density {self.rho():.4f} kg/m3   {'GIGA zero' if giga else 'tunnel offset'} {offset}   "
                     f"your board: {self.student_status or '-'}")
        self.lbl_status.setText("\n".join(lines))
        self.lbl_status.setStyleSheet("color: #c00;" if bad else "color: #333;")

    # ---- logging

    @QtCore.Slot()
    def toggle_logging(self):
        if not self.logging:
            self.logging = True
            self.log_start = time.time()
            self.rows_gt, self.rows_st, self.pairs_gt, self.pairs_st = [], [], [], []
            self.btn_log.setText("Stop && Save")
            self.lbl_log.setText("Logging: ON")
            self.lbl_log.setStyleSheet("color: #0a0;")
            self.cmb_source.setEnabled(False)
            return
        self._stop_logging()
        self._save_logs()

    def _stop_logging(self):
        self.logging = False
        self.btn_log.setText("Start Logging")
        self.lbl_log.setText("Logging: OFF")
        self.lbl_log.setStyleSheet("color: #666;")
        self.cmb_source.setEnabled(True)

    def default_run_name(self):
        team = safe_name(self.txt_team.text()) or "run"
        kind = {"sim": "_simulated", "tunnel": ""}.get(self.gt_key, "_giga")    # never mistaken for tunnel data
        return f"{team}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.log_start))}{kind}"

    def _save_logs(self):
        if not self.rows_gt and not self.rows_st:
            self.lbl_log.setText("Logging: OFF (nothing recorded)")
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        default = self.log_dir / (self.default_run_name() + ".csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save log (writes <name>_vdas / _student / _aligned .csv: tunnel, yours, paired)", str(default), "CSV Files (*.csv)")
        if not path:
            self.lbl_log.setText("Logging: OFF (log discarded)")
            return
        try:
            files = self.write_logs(Path(path))
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))
            return
        box, btn_plot, btn_folder = self._saved_box(files)
        box.exec()
        if box.clickedButton() is btn_plot:
            self.plot_saved_run(self.last_run)
        elif box.clickedButton() is btn_folder:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(files[0].parent)))

    def _saved_box(self, files):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Log saved")
        box.setText(f"{len(self.rows_gt)} {self.gt_name} readings and {len(self.rows_st)} samples from your board saved:\n\n"
                    + "\n".join(f.name for f in files) + f"\n\nin {files[0].parent}")
        btn_plot = box.addButton("Plot this run", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        btn_folder = box.addButton("Open folder", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        return box, btn_plot, btn_folder

    def write_logs(self, path):
        """Write <base>_vdas.csv, <base>_student.csv and <base>_aligned.csv (VDAS samples paired with the nearest
        student sample; the format COMP0219 Grading/Score.py reads)."""
        base = path.with_suffix("") if path.suffix.lower() == ".csv" else path
        files = []
        for suffix, rows in (("_vdas", self.rows_gt), ("_student", self.rows_st)):
            if rows:
                f = base.with_name(base.name + suffix + ".csv")
                with open(f, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(LOG_HEADER)
                    w.writerows(rows)
                files.append(f)
        aligned = align(self.pairs_gt, self.pairs_st, float(self.cfg["logging"]["align_tolerance_ms"]) / 1000)
        if aligned:
            f = base.with_name(base.name + "_aligned.csv")
            with open(f, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["timestamp_iso", "timestamp_epoch", "ground_truth_mps", "student_mps"])
                for t, g, s in aligned:
                    w.writerow([iso(t), f"{t:.6f}", f"{g:.6f}", "" if s is None else f"{s:.6f}"])
            files.append(f)
        self.last_run = base
        return files

    # ---- saved runs

    def plot_saved_run(self, base=None):
        if base is None:
            start = self.last_run.with_name(self.last_run.name + "_aligned.csv") if self.last_run else self.log_dir
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Plot run - choose its _aligned.csv", str(start),
                                                            "Saved runs (*_aligned.csv);;All CSV files (*.csv)")
            if not path:
                return None
            base = run_base(path)
        try:
            win = RunPlotWindow(base, "Your sensor")
        except (OSError, ValueError, KeyError) as e:
            QtWidgets.QMessageBox.warning(self, "Plot run", f"Cannot plot {base.name}:\n{e}")
            return None
        # closed windows delete themselves (WA_DeleteOnClose): drop them by identity, never touch them again
        win.destroyed.connect(lambda *_, w=win: setattr(
            self, "_run_windows", [x for x in self._run_windows if x is not w]))
        self._run_windows.append(win)
        win.show()
        return win

    def closeEvent(self, event):
        if self.logging and (self.rows_gt or self.rows_st):
            ans = QtWidgets.QMessageBox.question(self, "Save log?", "Logging is still on. Save the log before quitting?")
            if ans == QtWidgets.QMessageBox.StandardButton.Yes:
                self._stop_logging()
                self._save_logs()
        self.timer.stop()
        if self.student_reader is not None:
            self.student_reader.stop()
        self.vdas.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------- start-up

def install_error_handler():
    """Show unexpected errors in a dialog (the exe has no console) and keep them in %APPDATA%\\...\\error.log."""
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            APPDATA.mkdir(parents=True, exist_ok=True)
            with open(APPDATA / "error.log", "a", encoding="utf-8") as f:
                f.write(f"--- {datetime.datetime.now().isoformat()} {APP_NAME} {VERSION}\n{text}\n")
        except OSError:
            pass
        if QtWidgets.QApplication.instance() is not None:
            QtWidgets.QMessageBox.critical(None, f"{APP_NAME} error",
                                           f"Something went wrong:\n\n{exc}\n\nDetails were saved to\n"
                                           f"{APPDATA / 'error.log'}\nPlease send that file to the TA.")
    sys.excepthook = hook


def smoke_test(app, w, out_dir, seconds):
    """Unattended self-check (used to test the built exe): simulated run, save, plot, report, quit."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / "smoke_result.txt"

    def finish():
        try:
            w._stop_logging()
            files = w.write_logs(out_dir / "smoke.csv")
            win = w.plot_saved_run(w.last_run)
            n = len(w.pairs_st)
            span = w.pairs_st[-1][0] - w.pairs_st[0][0] if n > 1 else 0
            try:                                # the real VDAS reader (hidapi) must load too, e.g. in the exe
                import hid
                from wind_tunnel_logger.vdas_afa5 import PRODUCT_ID, VENDOR_ID
                boxes = len(hid.enumerate(VENDOR_ID, PRODUCT_ID))
                reader = f"VDAS reader (hidapi) OK, VDAS boxes on USB: {boxes}"
            except Exception as e:
                reader = f"VDAS reader (hidapi) FAILED: {e}"
            result.write_text(f"OK {APP_NAME} {VERSION}\nfiles: {', '.join(f.name for f in files)}\n"
                              f"VDAS readings {len(w.pairs_gt)}, student samples {n}, "
                              f"rate {(n - 1) / span if span else 0:.1f} Hz, offset {w.offset}\n"
                              f"plot window: {win.windowTitle() if win else None}\n{reader}\n", encoding="utf-8")
        except Exception:
            result.write_text("FAIL\n" + traceback.format_exc(), encoding="utf-8")
        QtCore.QTimer.singleShot(500, app.quit)

    QtCore.QTimer.singleShot(int(w.cfg["vdas"]["zero_seconds"] * 1000) + 1500, w.toggle_logging)
    QtCore.QTimer.singleShot(int((w.cfg["vdas"]["zero_seconds"] + 1.5 + seconds) * 1000), finish)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=APPDATA / "config.yaml",
                    help="advanced settings file (default: config.yaml in the settings folder)")
    ap.add_argument("--demo", action="store_true", help="simulated tunnel and simulated student (no hardware)")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    ap.add_argument("--smoke-test", metavar="DIR", help=argparse.SUPPRESS)
    ap.add_argument("--smoke-seconds", type=float, default=10.0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    install_error_handler()
    QtCore.QCoreApplication.setOrganizationName(ORG)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    if IS_WINDOWS:          # its own taskbar button and icon, not Python's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UCL.COMP0219.WindTunnelLogger")
        except (AttributeError, OSError):
            pass
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(str(resource("icon.png"))))     # tools/make_logo.py draws it
    pg.setConfigOptions(antialias=True)

    cfg = load_config(args.config)
    settings = QtCore.QSettings()
    for key in ("p_amb_pa", "air_temp_c"):
        if settings.value(key) is not None:
            cfg["air"][key] = float(settings.value(key))
    docs = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
    log_dir = Path(docs or Path.home()) / APP_NAME / "logs"
    if args.smoke_test:
        cfg["vdas"]["zero_seconds"] = 3
        log_dir = Path(args.smoke_test)
        settings = QtCore.QSettings(str(Path(args.smoke_test) / "settings.ini"), QtCore.QSettings.Format.IniFormat)

    demo = args.demo or bool(args.smoke_test)
    w = MainWindow(cfg, settings, log_dir, simulated=demo, fake_student=demo)
    w.show()
    if args.smoke_test:
        smoke_test(app, w, args.smoke_test, args.smoke_seconds)
    signal.signal(signal.SIGINT, lambda *_: w.close())
    tick = QtCore.QTimer()
    tick.timeout.connect(lambda: None)   # lets Python see Ctrl-C while Qt runs
    tick.start(200)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
