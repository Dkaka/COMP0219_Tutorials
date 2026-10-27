"""
Hardware check of Alex's VDAS-F keep-alive (tmp/afa5_reader.py): does the VDAS box stream to us with the TecQuipment
VDAS software CLOSED, only because we send the 2-byte keep-alive every 0.6 s?

    .venv\\Scripts\\python.exe -m pip install hidapi
    .venv\\Scripts\\python.exe tools\\check_vdas_keepalive.py

Three timed phases: A no keep-alive (expect silence), B keep-alive (expect ~4 readings/s), C after the stop byte
(expect silence again). Read-only apart from the same 2-byte messages the VDAS software itself sends.
"""
import subprocess
import sys
import time

try:
    import hid
    hid.device
except (ImportError, AttributeError):
    sys.exit("needs hidapi:  .venv\\Scripts\\python.exe -m pip install hidapi")

VID, PID = 0x1781, 0x07EA
HEARTBEAT, STOP = [0x00, 0x01], [0x00, 0x00]


def decode(report):
    """(cell, Pa) / (cell, 'over'|'under') for an AFA5 message, else None (same as Alex's afa5_reader.decode)."""
    if len(report) >= 65:
        report = report[1:]
    if len(report) < 6:
        return None
    b1 = report[1]
    device_id = report[2] + (256 if b1 & 0x80 else 0)
    if device_id != 1 or report[3] != 2:
        return None
    raw = (report[4] << 8) | report[5]
    if raw in (0x7FFF, 0xFFFF):
        return (b1 >> 3) & 7, "over" if raw == 0x7FFF else "under"
    mag = raw & 0x7FFF
    return (b1 >> 3) & 7, (-mag if raw & 0x8000 else mag)


def phase(dev, name, seconds, keepalive):
    reports, afa5, last_tx = 0, {}, 0.0
    end = time.time() + seconds
    while time.time() < end:
        if keepalive and time.time() - last_tx >= 0.6:
            dev.write(HEARTBEAT)
            last_tx = time.time()
        r = dev.read(64, 200)
        if r:
            reports += 1
            m = decode(r)
            if m:
                afa5.setdefault(m[0], []).append(m[1])
    cells = ", ".join(f"cell {c}: {len(v)} readings ({len(v) / seconds:.1f}/s), last {v[-3:]}"
                      for c, v in sorted(afa5.items())) or "no AFA5 readings"
    print(f"{name:34} {reports:3d} USB reports in {seconds:g} s | {cells}")
    return sum(len(v) for v in afa5.values())


def main():
    if sys.platform == "win32":
        tl = subprocess.run(["tasklist", "/FI", "IMAGENAME eq VDAS.exe", "/NH"], capture_output=True, text=True).stdout
        if "vdas.exe" in tl.lower():
            sys.exit("The TecQuipment VDAS software is running: close it first, so this tests the keep-alive alone.")
    dev = hid.device()
    try:
        dev.open(VID, PID)
    except (OSError, IOError) as e:
        sys.exit(f"VDAS box not found ({e}): plug it in and switch on the tunnel's instrument power.")
    print(f"opened {dev.get_manufacturer_string()} / {dev.get_product_string()}  (VDAS software closed)\n")
    try:
        a = phase(dev, "A  no keep-alive", 5, False)
        b = phase(dev, "B  keep-alive 00 01 every 0.6 s", 10, True)
        dev.write(STOP)
        c = phase(dev, "C  after stop byte 00 00", 5, False)
    finally:
        try:
            dev.write(STOP)
        except (OSError, IOError, ValueError):
            pass
        dev.close()
    print()
    if b >= 20 and c <= 3:
        print("PASS: the box streams only while it gets the keep-alive - no VDAS software needed.")
        if a:
            print("      (phase A had data: the box was probably still streaming from an earlier session)")
    else:
        print("FAIL: expected about 40 readings in phase B and silence in phase C - see the counts above.")


if __name__ == "__main__":
    main()
