"""Optical-flow extrapolation baseline: estimate motion between t-1h and t with Farneback flow,
then advect the map at t forward 1, 2, 3 hours. Also usable as extra input channels for the U-Net."""
import sys, numpy as np, cv2
from scipy.ndimage import map_coordinates


def flow_between(a, b):
    a8 = np.clip(np.log1p(np.maximum(a, 0)) * 40, 0, 255).astype(np.uint8)
    b8 = np.clip(np.log1p(np.maximum(b, 0)) * 40, 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 9, 3, 5, 1.1, 0)  # (H,W,2) dx,dy


def advect(field, flow, steps):
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for s in range(1, steps + 1):
        # semi-Lagrangian backward: value at x comes from x - s*flow
        src_x = xx - s * flow[..., 0]; src_y = yy - s * flow[..., 1]
        out.append(map_coordinates(field, [src_y, src_x], order=1, mode="constant", cval=0.0))
    return np.stack(out)


def advect_batch(X, steps=3):
    """X: (N,3,H,W) -> (N,steps,H,W) advected forecasts of the last frame."""
    out = np.zeros((X.shape[0], steps) + X.shape[2:], np.float32)
    for i in range(X.shape[0]):
        fl = flow_between(X[i, 1], X[i, 2])
        out[i] = advect(X[i, 2], fl, steps)
    return out


if __name__ == "__main__":
    import pandas as pd
    from fss import report
    D = sys.argv[1]
    tr = np.load(f"{D}/train.npz"); ts = pd.to_datetime(tr["times"])
    val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01"))
    Xv, yv = tr["X"][val], tr["y"][val]
    adv = advect_batch(Xv)
    report("persistence", np.repeat(Xv[:, 2:3], 3, 1), yv)
    report("optical flow advection", adv, yv)
    for t in (0.7, 1.0, 1.5):
        report(f"advection thr {t}", (adv > t) * 10.0, yv)
    np.save(f"{D}/../outputs/advect_val.npy", adv)
