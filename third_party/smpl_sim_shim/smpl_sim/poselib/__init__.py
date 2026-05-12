import sys as _sys
import os as _os

# Add smpl_sim_shim/ to sys.path so `import poselib` resolves to the
# vendored copy at smpl_sim_shim/poselib/ — no external paths needed.
_shim_root = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
if _shim_root not in _sys.path:
    _sys.path.insert(0, _shim_root)
