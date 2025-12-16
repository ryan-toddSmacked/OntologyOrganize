"""Setup script for building ClassifierOrganizer with cx_Freeze."""

import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but some modules need explicit inclusion
build_exe_options = {
    "packages": [
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PIL",
        "numpy",
        "scipy",
        "matplotlib",
        "cmcrameri",
        "pathlib",
        "io",
        "xml",
    ],
    "includes": [
        "scipy.ndimage",
        "scipy.stats",
        "matplotlib.pyplot",
        "cmcrameri.cm",
    ],
    "excludes": [
        "tkinter",
        "unittest",
        "email",
        "http",
        "PyQt5.QtQml",
        "PyQt5.QtQuick",
        "PyQt5.QtWebEngine",
        "PyQt5.QtWebEngineWidgets",
        "PyQt5.QtNetwork",
        "PyQt5.QtBluetooth",
    ],
    "include_files": [
        # Include any additional data files here if needed
        # ("data/", "data/"),
    ],
    "optimize": 2,
}

# Base for Windows GUI applications (no console window)
base = None
target_name = "ClassifierOrganizer"

if sys.platform == "win32":
    base = "Win32GUI"
    target_name = "ClassifierOrganizer.exe"
elif sys.platform == "linux":
    # For Linux, no base is needed (default console base is fine for GUI apps)
    target_name = "ClassifierOrganizer"

setup(
    name="ClassifierOrganizer",
    version="2.0",
    description="Image classification and organization tool",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main.py",
            base=base,
            target_name=target_name,
            icon=None,  # Add path to .ico or .png file if you have one
        )
    ],
)
