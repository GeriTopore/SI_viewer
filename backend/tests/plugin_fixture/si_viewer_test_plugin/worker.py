"""The test plugin's worker: reports progress, then writes a small image."""

import json
import sys
import time

import numpy as np

out, steps, delay, fail = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), bool(int(sys.argv[4]))
print("a line that isn't JSON", flush=True)
for i in range(steps):
    time.sleep(delay)
    print(json.dumps({"progress": (i + 1) / steps, "message": f"Step {i + 1} of {steps}"}), flush=True)
if fail:
    sys.exit("worker failed on purpose")
y, x = np.mgrid[:64, :64]
np.save(out, np.sin(x / 4.0) * np.cos(y / 4.0))
print(json.dumps({"result": {"path": out}}), flush=True)
