# Releasing a new version (for the TA)

Students download from the repo's GitHub **Releases** page. Each release carries three files built from this folder:

| Asset | For | Built by |
|---|---|---|
| `WindTunnelLogger.exe` | Windows students without Python | `build_exe.ps1` |
| `wind_tunnel_logger-<version>-py3-none-any.whl` | everyone (`pip install <link>`), any OS | `build_exe.ps1` |
| `Wind_Tunnel_Logger_User_Manual.pdf` | everyone (also in `docs\`) | `docs\build_manual.py` |

## 1. Build and check (Windows, in this folder)

1. Set the new version in `src/wind_tunnel_logger/__init__.py`, and delete the old files in `dist\`.
2. Build the exe and the wheel. The script also removes conda from `PATH` and clears stale build files itself.

   ```bash
   powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
   ```

3. Self-check the exe. A window opens for about 20 s; `OK` and `VDAS reader (hidapi) OK` should appear in
   `%TEMP%\wtl_check\smoke_result.txt`.

   ```bash
   .\dist\WindTunnelLogger.exe --smoke-test $env:TEMP\wtl_check
   ```

4. Rebuild the manual. It retakes the screenshots from the real program windows (simulated data; windows flash
   on screen for about a minute), then writes `docs\Wind_Tunnel_Logger_User_Manual.pdf`. Its version number and
   download links come from `__init__.py`.

   ```bash
   .\.venv\Scripts\python.exe -m pip install reportlab pypdfium2
   ```

   ```bash
   .\.venv\Scripts\python.exe docs\build_manual.py
   ```

5. At the tunnel (VDAS box plugged in, TecQuipment's VDAS software closed), check the keep-alive: it prints
   PASS when the box streams to us.

   ```bash
   .\.venv\Scripts\python.exe -m pip install hidapi
   ```

   ```bash
   .\.venv\Scripts\python.exe tools\check_vdas_keepalive.py
   ```

The logo (window, exe and manual) comes from `tools\make_logo.py`; run it after changing the design.

## 2. Publish on GitHub

Commit and push this folder first, so the tag holds the new manual.

**On the website:** open the repo's **Releases**, then **Draft a new release**.
1. Create the tag `wind-tunnel-logger-v<version>` (e.g. `wind-tunnel-logger-v1.0.0`) on `main` of
   Dkaka/COMP0219_Tutorials, whose `main` holds this folder's contents (last year's tutorials are on the branch
   `tutorials-2025-26`). The manual's pip link expects exactly that tag name.
2. Paste the text of `RELEASE_NOTES.md` as the description.
3. Drag in both files from `dist\` and `docs\Wind_Tunnel_Logger_User_Manual.pdf`, then press **Publish release**.

**Or with the GitHub CLI:**

```bash
gh release create wind-tunnel-logger-v1.0.0 dist\WindTunnelLogger.exe dist\wind_tunnel_logger-1.0.0-py3-none-any.whl docs\Wind_Tunnel_Logger_User_Manual.pdf --repo Dkaka/COMP0219_Tutorials --title "Wind Tunnel Logger 1.0.0" --notes-file RELEASE_NOTES.md
```

The release also gets an automatic "Source code" zip of the tagged commit; students ignore it. The developer
edition (with scoring) is not in this repo.

After publishing, check that the pip line in the manual works:

```bash
python -m pip install --dry-run https://github.com/Dkaka/COMP0219_Tutorials/releases/download/wind-tunnel-logger-v1.0.0/wind_tunnel_logger-1.0.0-py3-none-any.whl
```
