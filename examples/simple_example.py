#!/usr/bin/env python3
"""Simple example to demonstrate importing and running bootmedian.
This script tries to import the package and runs a small bootstrap with
`nsimul=100` to keep runtime small. It prints helpful messages if imports
fail (missing dependencies).
"""
import sys
print("Running bootmedian example...")
try:
    import bootmedian as bm
except Exception as e:
    print("Import failed:", repr(e))
    print("\nInstall required dependencies: numpy, multiprocess, bottleneck, pandas, tqdm, astropy, matplotlib, miniutils")
    print("From the project root run: pip install -e .\n")
    sys.exit(1)

import numpy as np

data = np.array([1.0, 2.1, 2.3, np.nan, 3.5, 2.0])
print("Sample:", data)

res = bm.bootmedian(data, nsimul=100, errors=1, verbose=False, nthreads=1)
print("Result keys:", list(res.keys()))
print("median:", res.get("median"))
print("Done.")
