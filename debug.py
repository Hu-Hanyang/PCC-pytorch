import json
import os
import numpy as np

setting_path = "result/planar"
setting = os.path.basename(os.path.normpath(setting_path))

log_folders = ['result/planar/planar_1']
for log in log_folders:
    print(log)