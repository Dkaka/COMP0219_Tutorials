#!/usr/bin/env python3
import serial
import time
import math
import os
import ctypes

# -----------------------------
# Enable real-time scheduling
# -----------------------------
SCHED_FIFO = 1

class sched_param(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]

def enable_realtime():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        param = sched_param(70)   # RT priority 1–99
        result = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
        if result != 0:
            e = ctypes.get_errno()
            print(f"Warning: Could not enable RT scheduling: errno={e}")
        else:
            print("Real-time scheduling enabled (SCHED_FIFO, priority 70)")
    except Exception as e:
        print("Could not enable RT scheduling:", e)


# -----------------------------
# Main program
# -----------------------------
enable_realtime()

ser = serial.Serial(
    "/dev/ttyAMA0",
    baudrate=115200,
    timeout=1,
    write_timeout=2
)

print("Outputting sine wave: freq=0.33 Hz, range 0–8, step=50 ms\n")

# Sine parameters
freq = 0.33
two_pi = 2 * math.pi
amplitude = 4.0
offset = 4.0

# Absolute-timer loop setup
t_next = time.monotonic()
period = 0.050  # 50 ms

t_start = time.monotonic()

while True:
    t_now = time.monotonic()
    t_sec = t_now - t_start

    # Sine wave
    y = amplitude * math.sin(two_pi * freq * t_sec) + offset

    msg = f"{y:.4f}\r\n"
    ser.write(msg.encode("utf-8"))
    ser.flush()

    print("Sent:", msg.strip())

    # Absolute next wake-up
    t_next += period
    sleep_time = t_next - time.monotonic()
    if sleep_time > 0:
        time.sleep(sleep_time)
