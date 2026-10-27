# Builds dist\WindTunnelLogger.exe: one file Windows students can double-click (no Python needed).
# The same code is also the pip package (pyproject.toml); this is just an extra download for Windows.
# Run from this folder:  powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# An active conda "(base)" puts Anaconda first on PATH, and PyInstaller then bundles Anaconda's old C runtime
# (ucrtbase.dll from 2018). Qt cannot use it: the exe dies with "DLL load failed while importing QtCore".
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch '(?i)anaconda|miniconda|\\conda' }) -join ';'
Remove-Item Env:CONDA_PREFIX, Env:CONDA_DEFAULT_ENV, Env:CONDA_SHLVL -ErrorAction SilentlyContinue

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.13 -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"

# setuptools reuses build\lib, so files from an old version or package name would end up in the wheel.
foreach ($stale in ".\build\lib", ".\build\bdist.win-amd64") {
    if (Test-Path $stale) { Remove-Item -Recurse -Force $stale }
}
& $py -m pip install --quiet . pyinstaller
& $py -m pip wheel . --no-deps -w dist --quiet          # the pip package (any OS)

# Qt / Python modules the GUI never uses are left out to keep the exe small.
$exclude = @("PyQt5", "PyQt6", "PySide2", "matplotlib", "tkinter", "scipy", "pandas", "IPython",
             "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick", "PySide6.QtQml",
             "PySide6.QtMultimedia", "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
             "PySide6.QtPdf", "PySide6.QtNetwork", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtBluetooth")
$excludeArgs = $exclude | ForEach-Object { "--exclude-module", $_ }

& $py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name WindTunnelLogger `
    --paths src `
    --icon src\wind_tunnel_logger\icon.ico `
    --add-data "src\wind_tunnel_logger\config.yaml;wind_tunnel_logger" `
    --add-data "src\wind_tunnel_logger\icon.png;wind_tunnel_logger" `
    --hidden-import wind_tunnel_logger.vdas_afa5 --hidden-import hid `
    @excludeArgs `
    run_gui.py

$conda = Select-String -Path .\build\WindTunnelLogger\*.toc -Pattern '(?i)anaconda|miniconda' -List
if ($conda) { throw "Anaconda files were bundled ($($conda[0].Path)): close conda and rebuild" }

Write-Host "Built: $(Resolve-Path .\dist\WindTunnelLogger.exe)"
Get-ChildItem .\dist\*.whl | ForEach-Object { Write-Host "Built: $($_.FullName)" }
Write-Host "Check it: .\dist\WindTunnelLogger.exe --demo"
