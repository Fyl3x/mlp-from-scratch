import os

if os.environ.get("MLP_BACKEND") == "numpy":
    import numpy as np
    BACKEND = "numpy"
else:
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no CUDA device")
        np = cupy
        BACKEND = "cupy"
    except Exception:
        import numpy as np
        BACKEND = "numpy"
def to_numpy(a):
    if BACKEND == "cupy":
        return np.asnumpy(a)
    return np.asarray(a)