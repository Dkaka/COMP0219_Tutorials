"""
TecQuipment VDAS-F / AFA5 differential pressure cell, read directly over USB HID with hidapi (Windows, macOS,
Linux). No VDAS software needed: this module keeps the VDAS-F streaming itself.

Protocol (from Alex's afa5_reader.py, COMP0219 2026/27):
  - The VDAS-F sends 64-byte HID reports, about 4 per second per device. Each report is one message:
        [0] message counter / part number
        [1] bits 0-2 interface input, bits 3-5 device instance, bit 7 adds 256 to the device id
        [2] device id (1 = AFA5 differential pressure cell, 20 = interface settings)
        [3] payload length
        [4..] payload
    AFA5: device 1 with a 2-byte payload; instance 1 = "DP Cell 1" (the tunnel's airspeed reference), 2 = "DP Cell 2".
    The payload is the pressure in Pa, 16-bit big-endian sign-magnitude (bit 15 set = negative, so 0x8001 is -1 Pa);
    0x7FFF means over range, 0xFFFF under range.
  - Keep-alive: the VDAS-F only streams while it gets a 2-byte output report (report id 0, then 0x01) about every
    600 ms; 0x00 tells it to stop. That is all the VDAS software did for us.
"""
import threading
import time

import hid

VENDOR_ID, PRODUCT_ID = 0x1781, 0x07EA
AFA5_DEVICE_ID = 1
HEARTBEAT = [0x00, 0x01]    # HID report id 0 + one data byte, as the VDAS software sends it
STOP = [0x00, 0x00]         # what the VDAS software sends when it shuts down
HEARTBEAT_S = 0.6
OVER_RANGE, UNDER_RANGE = 0x7FFF, 0xFFFF


def decode(report):
    """(cell, pressure_pa) for an AFA5 message, (cell, 'over'/'under') if out of range, else None."""
    if len(report) >= 65:        # some backends put the report-id byte first; this device has none
        report = report[1:]
    if len(report) < 6:
        return None
    b1 = report[1]
    device_id = report[2] + (256 if b1 & 0x80 else 0)
    cell = (b1 >> 3) & 7         # device instance
    if device_id != AFA5_DEVICE_ID or report[3] != 2:
        return None
    raw = (report[4] << 8) | report[5]
    if raw == OVER_RANGE:
        return cell, "over"
    if raw == UNDER_RANGE:
        return cell, "under"
    mag = raw & 0x7FFF
    return cell, (-mag if raw & 0x8000 else mag)


class GroundTruthReader:
    """Opens the VDAS-F, keeps it streaming with the keep-alive, and passes the readings of one AFA5 cell as
    {"t", "pa", "valid", "cell"} to listener callbacks (called on the reader thread). "valid" is False, and "pa"
    None, while the cell reports over or under range."""

    def __init__(self, cell=1):
        self.cell = int(cell)
        self.last_report = 0.0          # time of the last USB report of any kind (for diagnostics)
        self._dev = hid.device()
        try:
            self._dev.open(VENDOR_ID, PRODUCT_ID)
        except (OSError, IOError) as e:
            raise RuntimeError(f"tunnel's VDAS box not found ({e})") from None
        self._listeners = []
        self._stop = threading.Event()
        self._alive = True
        self._thread = threading.Thread(target=self._run, name="vdas-hid", daemon=True)
        self._thread.start()

    def _run(self):
        last_tx = 0.0
        try:
            while not self._stop.is_set():
                if time.time() - last_tx >= HEARTBEAT_S:
                    if self._dev.write(HEARTBEAT) < 0:
                        raise OSError("write failed")
                    last_tx = time.time()
                report = self._dev.read(64, 200)         # waits up to 0.2 s
                if not report:
                    continue
                now = time.time()
                self.last_report = now
                msg = decode(report)
                if msg is None or msg[0] != self.cell:
                    continue
                valid = not isinstance(msg[1], str)
                reading = {"t": now, "pa": msg[1] if valid else None, "valid": valid, "cell": msg[0]}
                for cb in list(self._listeners):
                    try:
                        cb(reading)
                    except Exception as e:
                        print(f"listener error: {e}")
        except (OSError, IOError, ValueError):
            self._alive = False             # unplugged or powered off: the GUI reopens it

    def add_listener(self, cb):
        self._listeners.append(cb)

    def remove_listener(self, cb):
        if cb in self._listeners:
            self._listeners.remove(cb)

    def is_plugged(self):
        return self._alive and self._thread.is_alive()

    def close(self):
        self._stop.set()
        self._thread.join(1.0)
        try:
            self._dev.write(STOP)           # tell the box we're done, as the VDAS software does
        except (OSError, IOError, ValueError):
            pass
        try:
            self._dev.close()
        except (OSError, IOError, ValueError):
            pass
