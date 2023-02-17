import json
import os

setting_path = "result/planar"
setting = os.path.basename(os.path.normpath(setting_path))
print(setting)
print(os.listdir(setting_path))

log_folders = [
        os.path.join(setting_path, dI)
        for dI in os.listdir(setting_path)
        if os.path.isdir(os.path.join(setting_path, dI))
    ]
print(log_folders)