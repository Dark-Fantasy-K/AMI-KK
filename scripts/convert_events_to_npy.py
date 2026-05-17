import os, sys, ctypes
from pathlib import Path
import numpy as np
 
# Load hdf5_ecf plugin
os.environ["HDF5_PLUGIN_PATH"] = "/opt/hdf5_ecf/lib/hdf5/plugin"
ctypes.CDLL("/opt/hdf5_ecf/lib/libhdf5_ecf_codec.so")
import h5py
 
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/FRED")
 
for h5 in sorted(root.rglob("events.hdf5")):
    npy = h5.with_suffix(".npy")
    if npy.exists():
        print(f"skip {h5}")
        continue
    print(f"converting {h5} ...", end=" ", flush=True)
    with h5py.File(h5, "r") as f:
        events = f["CD/events"][:]
    np.save(npy, events)
    print(f"OK ({len(events):,} events, {npy.stat().st_size/1e6:.1f} MB)")
    os.chmod(npy, 0o666)
print("Done.")
 
