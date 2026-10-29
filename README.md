<img src="src/wind_tunnel_logger/icon.png" alt="Wind Tunnel Logger logo" width="96" align="right">

# Wind Tunnel Logger (COMP0219 2026/2027 setup, by Shengyang Zhuang)

Live plot and logging of your airspeed sensor against the **wind tunnel's own airspeed reading**, which is the
ground truth. That reading comes from the tunnel's pressure sensor, through TecQuipment's *VDAS* box. Each run
is saved as CSV files for your report and for assessment.

**Full instructions with screenshots** (installing, zeroing the tunnel, running a test, troubleshooting):
[Wind_Tunnel_Logger_User_Manual.pdf](https://github.com/Dkaka/COMP0219_Tutorials/releases/latest/download/Wind_Tunnel_Logger_User_Manual.pdf), attached to each release.

## Install

Pick **one** of these. Both come from the
[Releases page](https://github.com/Dkaka/COMP0219_Tutorials/releases) (*Assets* of the latest *Wind Tunnel Logger* release).

**A. Windows, no Python needed.** Download `WindTunnelLogger.exe` and double-click it.
If Windows says *"Windows protected your PC"*, click **More info**, then **Run anyway**: the program isn't
code-signed.

**B. Any system (Windows, macOS, Linux) with Python 3.9 or newer.** Install from the release's `.whl` file:

```bash
python -m pip install https://github.com/Dkaka/COMP0219_Tutorials/releases/download/wind-tunnel-logger-v1.0.1/wind_tunnel_logger-1.0.1-py3-none-any.whl
```

(Or download the `.whl` file and run `python -m pip install wind_tunnel_logger-1.0.1-py3-none-any.whl` in that
folder.) On macOS/Linux, use `python3` instead of `python`. Then start the program:

```bash
wind-tunnel-logger
```

If that command isn't found, use `python -m wind_tunnel_logger` instead. To update to a newer release, run the
same `pip install` line with the new link.

Nothing else to install: you **don't need TecQuipment's VDAS software**. Wind Tunnel Logger talks to the
tunnel's VDAS box directly (close the VDAS software if it's open).

## Use

1. Keep the **fan OFF**. Plug the tunnel's ground truth into your laptop by USB: the **VDAS box** on the white
   (TecQuipment) tunnel, the **GIGA** on the wood tunnel.
2. Start this program and pick *Ground truth*: **Wind tunnel** on the white tunnel, **GIGA pitot tube (COMx)** on
   the wood tunnel. Within a few seconds the tunnel's data arrives and the program zeroes it.
3. Enter today's **air pressure**, **air temperature** and your **team** name.
4. Plug in your board, pick its port and press **Connect**.
5. Start the fan, press **Start Logging**, run your test, then press **Stop & Save**.

Until the tunnel reading arrives, the status line says what's missing (VDAS box not plugged in, tunnel power
off, ...).

**Plot run…** shows any saved run, and **Help** repeats these steps.

**No tunnel?** Set *Ground truth* to **Simulated (bench test)**. That lets you check your board's connection,
stream rate and logging at home; the simulated speed is not real wind.

**Using a Mac or Linux?** Install with pip (option B). Reading the tunnel should work there too, but it has
only been tested on Windows. If it doesn't, plug your board into a teammate's Windows laptop at the tunnel.

**Wood tunnel (GIGA pitot tube).** While its WindSpeedGiga (Arduino GIGA R1) is plugged in, *Ground truth* also
offers **GIGA pitot tube (COMx)**. The program sends it its calibration (LEQ 102.177 m, C 1; it forgets LEQ when it
loses power and would read 50–60 % low) and uses its readings only once it confirms them. Keep the fan OFF when you
pick it: it zeroes itself on the board. Its runs end in `_giga`.

### Your board must

- appear as a USB serial port (a NUCLEO does);
- print **one speed in m/s per line**, e.g. `printf("%.3f\r\n", v);`;
- stream at **200 Hz**, at **115200 baud**.

The rate you achieve is shown under your reading. It turns red if it's too slow or if a line isn't a number.

## Saved files

These go to `Documents/Wind Tunnel Logger/logs/` (**Open logs folder**), one set per run:

| File | Contents |
|---|---|
| `<team>_<date>_<time>_vdas.csv` | the tunnel's readings (speed, pressure) |
| `<team>_<date>_<time>_student.csv` | every sample from your board |
| `<team>_<date>_<time>_aligned.csv` | each tunnel reading paired with your nearest sample |

Runs with simulated ground truth end in `_simulated`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Red "NO DATA from the tunnel" | Follow the status line. *VDAS box not found* or *sends no data*: check its USB cable and the tunnel's instrument power, unplug and replug the cable, and close TecQuipment's VDAS software if it's open. *No pressure readings* or *sends pressure cells …*: ask your TA (the AFA5 pressure module). With one module, any cell number works: the status line shows which *pressure cell* is read |
| Linux: VDAS box not found although plugged in | Allow access: `echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1781", MODE="0666"' \| sudo tee /etc/udev/rules.d/99-vdas.rules`, then `sudo udevadm control --reload` and replug |
| Red "fan was running" | Stop the fan and press **Re-zero tunnel** |
| "cannot open COMx" | Close anything else using the port (serial monitor, PuTTY, CubeIDE terminal) |
| Your rate is red | Your board sends fewer than about 196 lines/s, or some lines aren't plain numbers |
| `DLL load failed while importing QtCore` (pip, Windows) | Old Anaconda Python. Install Python from python.org, or use the `.exe` |
| pip error mentioning *long path* (Windows) | Use a shorter folder, or tick *"Disable path length limit"* in the python.org installer |

Errors are also written to `error.log` in the settings folder (`%APPDATA%\Wind Tunnel Logger` on Windows).
Send that file to the TA.
