**Wind Tunnel Logger 1.0.0** - COMP0219 2026/2027 wind tunnel software (Shengyang Zhuang).

Live plot and logging of your airspeed sensor against the wind tunnel's own airspeed reading. Each run is
saved as CSV files. **Read the user manual (PDF below) before your tunnel session.**

**Download one:**
- **Windows, no Python:** `WindTunnelLogger.exe`. Double-click it. If Windows warns you, click *More info*, then
  *Run anyway*.
- **Any system with Python 3.9+:** copy the link of the `.whl` file below, run
  `python -m pip install <link>`, then start the program with `wind-tunnel-logger`.

Nothing else to install: Wind Tunnel Logger reads the tunnel's VDAS box directly, so TecQuipment's VDAS software
isn't needed (close it if it's open). Tested on Windows; macOS and Linux should work with the pip version.

Your board must print one speed in m/s per line, at 200 Hz, 115200 baud.

A WindSpeedGiga pitot tube can also be the ground truth: it is offered under *Ground truth* while one is plugged
in (manual, section 6).
