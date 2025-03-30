import platform
import subprocess

system = platform.system()
name = "Linear Programming Generator"

if system == "Windows":
    subprocess.run([
        "pyinstaller", "--onefile", "--noconsole",
        "--name", f"{name}-Windows",
        "--icon", "resources/win/icon.ico",
        "src/gui-app.py"
    ])
elif system == "Linux":
    subprocess.run([
        "pyinstaller", "--onefile",
        "--name", f"{name}-Linux",
        "--icon", "resources/linux/icon.xbm",
        "src/gui-app.py"
    ])
else:
    print("Unsupported OS")