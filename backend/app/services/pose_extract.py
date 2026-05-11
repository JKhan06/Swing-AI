
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

video_path_str: Optional[str] = None

# Keep a focused set of golf-relevant landmarks to keep JSON small + meaningful.
# (We can expand later for front-on, hands, etc.)
KEEP_LANDMARKS = {
    # head
    "nose": mp.solutions.pose.PoseLandmark.NOSE,
    "left_eye": mp.solutions.pose.PoseLandmark.LEFT_EYE,
    "right_eye": mp.solutions.pose.PoseLandmark.RIGHT_EYE,
    "left_ear": mp.solutions.pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp.solutions.pose.PoseLandmark.RIGHT_EAR,
    # upper body
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_index": mp.solutions.pose.PoseLandmark.LEFT_INDEX,
    "right_index": mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
    # torso/legs
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
}


def _smooth_series(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Simple moving-average smoothing over time (frames).
    arr shape: (T, D)
    """
    if window <= 1 or arr.shape[0] < window:
        return arr

    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")

    smoothed_cols = []
    for i in range(arr.shape[1]):
        smoothed_cols.append(np.convolve(padded[:, i], kernel, mode="valid"))

    return np.stack(smoothed_cols, axis=1)


def extract_pose_from_video(
    video_path: Path,
    min_vis: float = 0.4,
    smooth_window: int = 5,
    view: str = "side_on",
) -> Dict[str, Any]:
    """
    Extract pose landmarks from a video using MediaPipe Pose.

    Output JSON shape:
    {
      "meta": {
        "video": "...",
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "frame_count": 123,
        "min_visibility": 0.4,
        "smooth_window": 5,
        "view": "side_on"
      },
      "frames": [
        {
          "frame_index": 0,
          "time_sec": 0.0,
          "landmarks": {
            "left_shoulder": {"x": 123.4, "y": 456.7, "vis": 0.91},
            ...
          }
        },
        ...
      ]
    }

    Notes:
    - x/y are PIXEL coordinates (not normalized).
    - If visibility < min_vis, x/y are None for that landmark on that frame.
    - Optional smoothing is applied only on frames where a landmark is valid.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_val) if fps_val and fps_val > 0 else 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mp_pose = mp.solutions.pose

    frames_out: List[Dict[str, Any]] = []
    raw_tracks: Dict[str, List[List[float]]] = {name: [] for name in KEEP_LANDMARKS.keys()}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB before processing.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Improve performance by passing by reference.
            rgb.flags.writeable = False
            results = pose.process(rgb)

            t = idx / fps if fps > 0 else 0.0
            frame_item: Dict[str, Any] = {
                "frame_index": idx,
                "time_sec": round(float(t), 4),
                "landmarks": {},
            }

            if results.pose_landmarks is None:
                # no detection this frame -> store missing points
                for name in KEEP_LANDMARKS.keys():
                    frame_item["landmarks"][name] = {"x": None, "y": None, "vis": 0.0}
                    raw_tracks[name].append([np.nan, np.nan])
                frames_out.append(frame_item)
                idx += 1
                continue

            lm = results.pose_landmarks.landmark

            for name, enum_idx in KEEP_LANDMARKS.items():
                p = lm[enum_idx.value]
                vis = float(p.visibility)

                if vis < min_vis or width == 0 or height == 0:
                    frame_item["landmarks"][name] = {"x": None, "y": None, "vis": round(vis, 3)}
                    raw_tracks[name].append([np.nan, np.nan])
                else:
                    x_px = float(p.x) * float(width)
                    y_px = float(p.y) * float(height)
                    frame_item["landmarks"][name] = {
                        "x": round(x_px, 2),
                        "y": round(y_px, 2),
                        "vis": round(vis, 3),
                    }
                    raw_tracks[name].append([x_px, y_px])

            frames_out.append(frame_item)
            idx += 1

    cap.release()

    # Optional smoothing: smooth per-landmark only over frames where the landmark is valid
    if smooth_window and smooth_window > 1:
        for name in KEEP_LANDMARKS.keys():
            arr = np.array(raw_tracks[name], dtype=float)  # (T,2) with NaNs
            valid_mask = ~np.isnan(arr).any(axis=1)

            if valid_mask.sum() < smooth_window:
                continue

            smoothed_valid = _smooth_series(arr[valid_mask], window=smooth_window)
            arr_sm = arr.copy()
            arr_sm[valid_mask] = smoothed_valid

            # Write smoothed x/y back into frames_out where x/y is not None
            for i in range(arr_sm.shape[0]):
                if not valid_mask[i]:
                    continue
                if frames_out[i]["landmarks"][name]["x"] is None:
                    continue
                frames_out[i]["landmarks"][name]["x"] = round(float(arr_sm[i, 0]), 2)
                frames_out[i]["landmarks"][name]["y"] = round(float(arr_sm[i, 1]), 2)

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": frame_count if frame_count > 0 else len(frames_out),
        "min_visibility": float(min_vis),
        "smooth_window": int(smooth_window),
        "view": view,
    }

    return {"meta": meta, "frames": frames_out}



def extract_pose_to_dict(
    video_path: str,
    *,
    view: str = "side_on",
    min_visibility: float = 0.4,
    smooth_window: int = 5,
) -> Dict[str, Any]:
    """
    Extract pose landmarks from a video and return a dict with `meta` and `frames`.

    This is a convenience wrapper around `extract_pose_from_video` that accepts a string path,
    and uses keyword args aligned with API usage.
    """
    path = Path(video_path).expanduser().resolve()
    return extract_pose_from_video(
        video_path=path,
        min_vis=min_visibility,
        smooth_window=smooth_window,
        view=view,
    )


def detect_swing_phases(
    pose: Dict[str, Any],
    *,
    wrist_preference: str = "auto",
    ball_xy: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Detect basic golf swing phases (address, top, impact) from pose time-series.

    Side-on v2 heuristic (more robust):
    1) Pick the most reliable wrist track (auto picks higher visibility coverage).
    2) Compute smoothed wrist (x,y) and smoothed speed to find the actual swing window
       (ignores setup time + waggles).
    3) ADDRESS: a stable low-speed segment immediately before swing start (else first valid).
    4) TOP: extremum of wrist displacement within the swing window, refined to the nearest
       direction-change (velocity sign flip) on the dominant axis.
    5) IMPACT: peak speed in the downswing window (top -> end), refined with ball/template
       detection and a kinematic fallback.

    Returns frame indices + timestamps + tempo metrics.
    Adds `impact_debug` with method info.
    """
    global video_path_str

    meta = pose.get("meta", {}) or {}
    frames = pose.get("frames", []) or []
    fps = float(meta.get("fps") or 60.0)
    min_vis = float(meta.get("min_visibility") or 0.4)
    smooth_window = int(meta.get("smooth_window") or 5)

    video_path_str = meta.get("video")

    def _detect_ball_center_near_address(address_frame: int) -> Optional[Tuple[float, float]]:
        """Best-effort golf ball detection near the bottom of the frame.

        Runs two strategies — color-mask contours and Hough circles — then merges
        results. When both agree (within 30 px) they are averaged for a more stable
        estimate; when only one fires it is used alone.
        """
        if not video_path_str:
            return None
        try:
            vp = Path(str(video_path_str)).expanduser().resolve()
            if not vp.exists():
                return None
        except Exception:
            return None

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, address_frame)))
            ok, frame = cap.read()
            if not ok or frame is None:
                return None

            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return None

            # ROI: bottom portion where ball typically sits.
            y0 = int(h * 0.58)
            roi = frame[y0:h, 0:w]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            roi_area = float(roi.shape[0] * roi.shape[1])
            min_a = max(25.0, 0.00010 * roi_area)
            max_a = max(min_a + 10.0, 0.00600 * roi_area)

            # --- Strategy 1: color mask (white + yellow) ---
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_white = cv2.inRange(hsv, (0, 0, 160), (180, 90, 255))
            mask_yellow = cv2.inRange(hsv, (15, 60, 120), (45, 255, 255))
            mask = cv2.bitwise_or(mask_white, mask_yellow)
            mask = cv2.medianBlur(mask, 5)
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color_best: Optional[Tuple[float, float]] = None
            color_best_score = -1e18

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_a or area > max_a:
                    continue
                per = float(cv2.arcLength(cnt, True))
                if per <= 1e-6:
                    continue
                circ = float(4.0 * math.pi * area / (per * per))
                if circ < 0.50:
                    continue
                m = cv2.moments(cnt)
                if abs(m.get("m00", 0.0)) < 1e-6:
                    continue
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                cx_i = int(round(cx))
                cy_i = int(round(cy))
                if cx_i < 0 or cy_i < 0 or cx_i >= gray_blur.shape[1] or cy_i >= gray_blur.shape[0]:
                    continue
                br = float(gray_blur[cy_i, cx_i])
                bottomness = float(cy_i) / float(max(1, gray_blur.shape[0]))
                score = 1800.0 * circ + 1.8 * br + 300.0 * bottomness + 0.02 * area
                if score > color_best_score:
                    color_best_score = score
                    color_best = (cx, cy)

            # --- Strategy 2: Hough circles ---
            hough_best: Optional[Tuple[float, float]] = None
            try:
                gray_hough = cv2.GaussianBlur(gray, (9, 9), 2)
                min_r = max(3, int((min_a / math.pi) ** 0.5))
                max_r = max(10, int((max_a / math.pi) ** 0.5))
                circles_raw = cv2.HoughCircles(
                    gray_hough, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=15,
                    param1=50, param2=25,
                    minRadius=min_r, maxRadius=max_r,
                )
                if circles_raw is not None:
                    hough_score = -1e18
                    for (hx, hy, _hr) in circles_raw[0]:
                        hx, hy = float(hx), float(hy)
                        if hy < roi.shape[0] * 0.05:
                            continue
                        hi_i = int(round(hy))
                        wi_i = int(round(hx))
                        if hi_i < 0 or wi_i < 0 or hi_i >= gray.shape[0] or wi_i >= gray.shape[1]:
                            continue
                        br = float(gray[hi_i, wi_i])
                        bottomness = float(hy) / float(max(1, gray.shape[0]))
                        sc = 1.8 * br + 300.0 * bottomness
                        if sc > hough_score:
                            hough_score = sc
                            hough_best = (hx, hy)
            except Exception:
                pass

            # --- Merge: average when both agree, else use whichever fired ---
            best: Optional[Tuple[float, float]]
            if color_best is not None and hough_best is not None:
                dx = color_best[0] - hough_best[0]
                dy = color_best[1] - hough_best[1]
                if (dx * dx + dy * dy) ** 0.5 <= 30.0:
                    best = ((color_best[0] + hough_best[0]) / 2.0, (color_best[1] + hough_best[1]) / 2.0)
                else:
                    best = color_best
            elif color_best is not None:
                best = color_best
            elif hough_best is not None:
                best = hough_best
            else:
                return None

            bx, by = best
            return (float(bx), float(by + y0))
        finally:
            cap.release()

    def get_xy_series(name: str) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        vis: List[Optional[float]] = []
        for fr in frames:
            lm = (fr.get("landmarks") or {}).get(name) or {}
            x = lm.get("x")
            y = lm.get("y")
            v = lm.get("vis")
            xs.append(float(x) if x is not None else None)
            ys.append(float(y) if y is not None else None)
            vis.append(float(v) if v is not None else None)
        return xs, ys, vis

    rx, ry, rvis = get_xy_series("right_wrist")
    lx, ly, lvis = get_xy_series("left_wrist")

    def count_good(xs: List[Optional[float]], ys: List[Optional[float]], vis: List[Optional[float]], thr: float) -> int:
        c = 0
        for xx, yy, vv in zip(xs, ys, vis):
            if xx is None or yy is None or vv is None:
                continue
            if vv >= thr and math.isfinite(float(xx)) and math.isfinite(float(yy)):
                c += 1
        return c

    if wrist_preference == "right":
        wrist = "right_wrist"
        x_raw, y_raw, v_raw = rx, ry, rvis
    elif wrist_preference == "left":
        wrist = "left_wrist"
        x_raw, y_raw, v_raw = lx, ly, lvis
    else:
        right_good = count_good(rx, ry, rvis, min_vis)
        left_good = count_good(lx, ly, lvis, min_vis)
        if right_good >= left_good:
            wrist = "right_wrist"
            x_raw, y_raw, v_raw = rx, ry, rvis
        else:
            wrist = "left_wrist"
            x_raw, y_raw, v_raw = lx, ly, lvis

    x: List[Optional[float]] = []
    y: List[Optional[float]] = []
    vis_ok: List[bool] = []
    for xx, yy, vv in zip(x_raw, y_raw, v_raw):
        if xx is None or yy is None or vv is None or float(vv) < min_vis:
            x.append(None)
            y.append(None)
            vis_ok.append(False)
        else:
            x.append(float(xx))
            y.append(float(yy))
            vis_ok.append(True)

    idxs = [i for i, ok in enumerate(vis_ok) if ok]
    if not idxs:
        return {"ok": False, "reason": "No reliable wrist track found", "wrist": wrist}

    x_s = x[:]
    y_s = y[:]
    if smooth_window and smooth_window > 1:
        half = smooth_window // 2

        def _smooth_list(vals: List[Optional[float]]) -> List[Optional[float]]:
            out: List[Optional[float]] = [None] * len(vals)
            for i in range(len(vals)):
                if vals[i] is None:
                    continue
                s = 0.0
                n = 0
                for j in range(max(0, i - half), min(len(vals), i + half + 1)):
                    if vals[j] is None:
                        continue
                    s += float(vals[j])
                    n += 1
                out[i] = (s / n) if n else None
            return out

        x_s = _smooth_list(x)
        y_s = _smooth_list(y)

    vx: List[Optional[float]] = [None] * len(x_s)
    vy: List[Optional[float]] = [None] * len(y_s)
    speed: List[Optional[float]] = [None] * len(x_s)
    for i in range(1, len(x_s)):
        if x_s[i] is None or y_s[i] is None or x_s[i - 1] is None or y_s[i - 1] is None:
            continue
        dx = float(x_s[i] - x_s[i - 1])
        dy = float(y_s[i] - y_s[i - 1])
        vx[i] = dx
        vy[i] = dy
        speed[i] = float((dx * dx + dy * dy) ** 0.5)

    speed_s = _moving_average(speed, max(1, min(9, smooth_window + 2)))

    finite_speeds = [float(s) for s in speed_s if s is not None and math.isfinite(float(s))]
    if not finite_speeds:
        return {"ok": False, "reason": "No reliable wrist speed found", "wrist": wrist}

    finite_sorted = sorted(finite_speeds)
    p60 = finite_sorted[int(0.60 * (len(finite_sorted) - 1))] if len(finite_sorted) > 1 else finite_sorted[0]
    thr = max(2.5, float(p60))

    above = [i for i, s in enumerate(speed_s) if s is not None and float(s) >= thr]
    if not above:
        swing_start = idxs[0]
        swing_end = idxs[-1]
    else:
        sustain = max(6, int(round(0.12 * fps)))
        swing_start = above[0]
        run = 1
        for k in range(1, len(above)):
            if above[k] == above[k - 1] + 1:
                run += 1
            else:
                run = 1
            if run >= sustain:
                swing_start = above[k - run + 1]
                break

        swing_end = above[-1]
        swing_end = min(len(frames) - 1, max(swing_end, idxs[-1]))
        swing_start = max(swing_start, idxs[0])
        swing_end = min(swing_end, idxs[-1])

    first_valid = idxs[0]
    pre_end = max(first_valid, swing_start - 1)
    lookback = max(12, int(round(0.90 * fps)))
    search_start = max(first_valid, pre_end - lookback + 1)
    stable_thr = max(1.5, thr * 0.35)

    stable_candidates = [
        i
        for i in range(search_start, pre_end + 1)
        if speed_s[i] is not None and float(speed_s[i]) <= stable_thr and vis_ok[i]
    ]

    address = first_valid

    def _wrist_y(i: int) -> float:
        # Higher y value = wrist closer to ground = more likely to be address position
        v = y_s[i]
        return float(v) if v is not None else -1e9

    if stable_candidates:
        # Among slow frames before swing, pick the one with wrist nearest the ground.
        # This rules out momentary pauses mid-backswing where wrists are already elevated.
        address = int(max(stable_candidates, key=_wrist_y))
    else:
        candidates = [i for i in idxs if i <= pre_end]
        if candidates:
            address = int(max(candidates, key=_wrist_y))
        else:
            address = first_valid

    window_idxs = [i for i in idxs if swing_start <= i <= swing_end]
    if not window_idxs:
        window_idxs = [i for i in idxs if i >= address]

    ax = x_s[address]
    ay = y_s[address]
    if ax is None or ay is None:
        address = first_valid
        ax = x_s[address]
        ay = y_s[address]

    disp_x: Dict[int, float] = {}
    disp_y: Dict[int, float] = {}
    for i in window_idxs:
        if x_s[i] is None or y_s[i] is None or ax is None or ay is None:
            continue
        disp_x[i] = float(x_s[i] - ax)
        disp_y[i] = float(y_s[i] - ay)

    if not disp_x:
        return {"ok": False, "reason": "Insufficient wrist samples in swing window", "wrist": wrist}

    dx_vals = list(disp_x.values())
    dy_vals = list(disp_y.values())
    dx_spread = (max(dx_vals) - min(dx_vals)) if dx_vals else 0.0
    dy_spread = (max(dy_vals) - min(dy_vals)) if dy_vals else 0.0
    axis = "x" if dx_spread >= dy_spread else "y"
    v_series = vx if axis == "x" else vy

    post_start = [i for i in window_idxs if i >= swing_start]
    if not post_start:
        post_start = window_idxs[:]

    impact0 = max(post_start, key=lambda i: float(speed_s[i] or -1.0))

    def _sign(v: Optional[float]) -> int:
        if v is None:
            return 0
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    search_lo = max(address + 2, swing_start)
    search_hi = max(search_lo + 1, impact0 - 2)

    flip_candidates: List[int] = []
    for i in range(search_lo + 1, min(search_hi, len(v_series) - 1)):
        s1 = _sign(v_series[i - 1])
        s2 = _sign(v_series[i])
        if s1 != 0 and s2 != 0 and s1 != s2:
            flip_candidates.append(i)

    if flip_candidates:
        top = int(flip_candidates[-1])
    else:
        bounded = [i for i in window_idxs if i <= impact0]
        if not bounded:
            bounded = window_idxs[:]
        if axis == "x":
            top0 = max(bounded, key=lambda i: abs(disp_x.get(i, 0.0)))
        else:
            top0 = max(bounded, key=lambda i: abs(disp_y.get(i, 0.0)))

        r = max(6, int(round(0.12 * fps)))
        neigh = [i for i in bounded if abs(i - top0) <= r and speed_s[i] is not None]
        if neigh:
            top = int(min(neigh, key=lambda i: float(speed_s[i] or 1e9)))
        else:
            top = int(top0)

    top = max(int(address), int(top))
    if top >= impact0:
        top = max(int(address), int(impact0) - 1)

    impact_bound = int(impact0)

    downswing_idxs = [i for i in window_idxs if i >= top]
    if not downswing_idxs:
        downswing_idxs = [i for i in idxs if i >= top]

    downswing_speed = [(i, float(speed_s[i])) for i in downswing_idxs if speed_s[i] is not None]
    if not downswing_speed:
        return {"ok": False, "reason": "No speed samples in downswing", "wrist": wrist}

    # Prefer peak speed among frames where wrist has returned near address height.
    # At impact the wrist is back near the ground; in follow-through it rises — this
    # naturally excludes false peaks caused by tracking noise in follow-through.
    address_y = y_s[address]
    if address_y is not None:
        # Keep frames where wrist_y >= 80% of address y (within 20% below ground level)
        near_ground = [(i, s) for i, s in downswing_speed
                       if y_s[i] is not None and float(y_s[i]) >= float(address_y) * 0.80]
        if near_ground:
            peak_i, _ = max(near_ground, key=lambda t: t[1])
        else:
            peak_i, _ = max(downswing_speed, key=lambda t: t[1])
    else:
        peak_i, _ = max(downswing_speed, key=lambda t: t[1])

    ball_xy_use = ball_xy if (ball_xy is not None) else _detect_ball_center_near_address(address)
    impact_method = "unknown"

    if ball_xy_use is not None:
        peak_i = int(peak_i)
        win_pre = max(3, int(round(0.10 * fps)))
        win_post = max(6, int(round(0.25 * fps)))

        start = max(top, peak_i - win_pre)
        end = min(downswing_idxs[-1], peak_i + win_post)

        # Run all three ball-based methods and vote; median of successful results wins.
        ball_candidates: List[Tuple[str, int]] = []

        r1 = _impact_from_ball_template_match(ball_xy_use, start_frame=start, end_frame=end, roi_scale=0.12)
        if r1 is not None:
            ball_candidates.append(("ball_template", int(r1)))

        r2 = _impact_from_ball_roi_motion(ball_xy_use, start_frame=start, end_frame=end, roi_scale=0.10)
        if r2 is not None:
            ball_candidates.append(("ball_motion", int(r2)))

        r3 = _detect_ball_movement_frame(ball_xy_use, start_frame=start, end_frame=end, roi_scale=0.12)
        if r3 is not None:
            ball_candidates.append(("ball_movement", int(r3)))

        if len(ball_candidates) >= 2:
            vals = sorted([c[1] for c in ball_candidates])
            impact = vals[len(vals) // 2]
            impact_method = "consensus(" + ",".join(c[0] for c in ball_candidates) + ")"
        elif len(ball_candidates) == 1:
            impact = ball_candidates[0][1]
            impact_method = ball_candidates[0][0]
        else:
            # All ball methods failed — peak wrist speed is the best single-frame proxy
            impact_method = "kinematic_fallback"
            impact = int(peak_i)
    else:
        # No ball detected at all — same fallback
        impact_method = "kinematic_fallback"
        impact = int(peak_i)

    if impact < top:
        impact = int(peak_i)

    if (impact + 1) <= downswing_idxs[-1]:
        y0 = y_s[impact]
        y1 = y_s[impact + 1]
        if y0 is not None and y1 is not None and float(y1) > float(y0) + 2.0:
            impact = int(impact + 1)

    def time_of(frame_index: int) -> float:
        return float(frame_index) / fps if fps > 0 else 0.0

    address_t = time_of(address)
    top_t = time_of(top)
    impact_t = time_of(impact)

    backswing = max(0.0, top_t - address_t)
    downswing = max(1e-6, impact_t - top_t)
    tempo_ratio = backswing / downswing if downswing > 0 else None

    coverage = float(len(idxs)) / float(len(frames) or 1)
    motion_peak = float(max(finite_speeds)) if finite_speeds else 0.0
    motion_clear = 1.0 if motion_peak >= (thr * 1.5) else 0.6
    conf = max(0.0, min(1.0, 0.6 * min(1.0, coverage / 0.75) + 0.4 * motion_clear))

    return {
        "ok": True,
        "wrist": wrist,
        "address": {"frame": int(address), "time_sec": round(address_t, 4)},
        "top": {"frame": int(top), "time_sec": round(top_t, 4)},
        "impact": {"frame": int(impact), "time_sec": round(impact_t, 4)},
        "impact_debug": {"ball_xy_used": ball_xy_use, "impact_method": impact_method},
        "metrics": {
            "backswing_sec": round(backswing, 4),
            "downswing_sec": round(downswing, 4),
            "tempo_ratio": round(tempo_ratio, 3) if tempo_ratio is not None else None,
            "fps": fps,
        },
        "confidence": {
            "overall": round(conf, 3),
            "coverage": round(coverage, 3),
            "motion_threshold": round(thr, 3),
            "dominant_axis": axis,
        },
    }


def write_json(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks from a video using MediaPipe Pose."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Path to input video file (e.g. swing.mp4).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Path to output JSON file (e.g. backend/outputs/pose.json).",
    )
    parser.add_argument(
        "--min_vis",
        type=float,
        default=0.4,
        help="Minimum landmark visibility (0..1). Below this, x/y become None.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Smoothing window (frames). Use 1 to disable smoothing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    video_path = Path(args.video).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    data = extract_pose_from_video(
        video_path=video_path,
        min_vis=args.min_vis,
        smooth_window=args.smooth,
        view="side_on",
    )
    write_json(data, out_path)

    print(f"✅ Wrote pose landmarks to: {out_path}")
    print(
        f"Frames: {len(data['frames'])} | FPS: {data['meta']['fps']:.2f} | View: {data['meta']['view']}"
    )


if __name__ == "__main__":
    main()


def _moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return values[:]
    out: List[Optional[float]] = [None] * len(values)
    half = window // 2
    for i in range(len(values)):
        s = 0.0
        n = 0
        for j in range(max(0, i - half), min(len(values), i + half + 1)):
            v = values[j]
            if v is None:
                continue
            s += float(v)
            n += 1
        out[i] = (s / n) if n else None
    return out


def _finite_indices(values: List[Optional[float]]) -> List[int]:
    return [i for i, v in enumerate(values) if v is not None and math.isfinite(float(v))]


def _impact_from_ball_template_match(
    ball_xy: Tuple[float, float],
    *,
    start_frame: int,
    end_frame: int,
    roi_scale: float = 0.15,  # Increased from 0.12
) -> Optional[int]:
    """Estimate impact as the frame **right before** the ball stops matching its pre-impact appearance.
    
    IMPROVEMENTS:
    - More lenient correlation thresholds
    - Better baseline establishment
    - Longer baseline sampling period
    """
    if not video_path_str:
        return None
    try:
        vp = Path(str(video_path_str)).expanduser().resolve()
        if not vp.exists():
            return None
    except Exception:
        return None

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return None

    try:
        start = max(0, int(start_frame))
        end = max(start + 3, int(end_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

        ok, first = cap.read()
        if not ok or first is None:
            return None

        h, w = first.shape[:2]
        if h <= 0 or w <= 0:
            return None

        bx, by = float(ball_xy[0]), float(ball_xy[1])

        # ROI around the ball
        r = int(round(min(h, w) * float(roi_scale)))
        r = max(50, min(200, r))  # Increased minimum

        def crop(img: "np.ndarray") -> Tuple[Optional["np.ndarray"], Tuple[int, int, int, int]]:
            y0 = max(0, int(round(by - r)))
            y1 = min(h, int(round(by + r)))
            x0 = max(0, int(round(bx - r)))
            x1 = min(w, int(round(bx + r)))
            if (y1 - y0) < 30 or (x1 - x0) < 30:  # Increased minimum
                return None, (x0, y0, x1, y1)
            return img[y0:y1, x0:x1], (x0, y0, x1, y1)

        # Build template with larger size for stability
        tpl_half = max(15, int(round(r * 0.22)))  # Increased
        tpl_half = min(tpl_half, 35)

        roi0, _ = crop(first)
        if roi0 is None:
            return None

        def to_gray(bgr: "np.ndarray") -> "np.ndarray":
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (7, 7), 0)  # Slightly more blur
            return g

        g0 = to_gray(roi0)
        H0, W0 = g0.shape[:2]
        cx0 = int(round(W0 / 2.0))
        cy0 = int(round(H0 / 2.0))

        # Find brightest spot near bottom (likely the ball)
        try:
            y_scan0 = int(H0 * 0.50)  # Search lower portion
            scan = g0[y_scan0:H0, :]
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(scan)
            cx0 = int(maxloc[0])
            cy0 = int(maxloc[1] + y_scan0)
        except Exception:
            pass

        x0 = max(0, cx0 - tpl_half)
        x1 = min(W0, cx0 + tpl_half)
        y0 = max(0, cy0 - tpl_half)
        y1 = min(H0, cy0 + tpl_half)
        if (x1 - x0) < 20 or (y1 - y0) < 20:
            return None

        template = g0[y0:y1, x0:x1].copy()
        template = cv2.equalizeHist(template)

        # CRITICAL: Sample baseline over MORE frames (15 instead of 8)
        scores: List[Optional[float]] = [None]
        locs: List[Optional[Tuple[int, int]]] = [None]
        base_samples: List[Tuple[int, int, float]] = []

        prev_roi_g = g0
        cur = start

        baseline_frames = 15  # INCREASED from 8

        while cur <= end:
            if cur == start:
                roi = roi0
            else:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                roi, _ = crop(fr)
                if roi is None:
                    scores.append(None)
                    locs.append(None)
                    cur += 1
                    continue

            g = to_gray(roi)
            g = cv2.equalizeHist(g)

            if g.shape[0] < template.shape[0] + 2 or g.shape[1] < template.shape[1] + 2:
                scores.append(None)
                locs.append(None)
                cur += 1
                continue

            res = cv2.matchTemplate(g, template, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)

            scores.append(float(maxv))
            locs.append((int(maxloc[0]), int(maxloc[1])))

            # Longer baseline sampling
            if (cur - start) <= baseline_frames:
                base_samples.append((int(maxloc[0]), int(maxloc[1]), float(maxv)))

            prev_roi_g = g
            cur += 1

        T = len(scores)
        if T < 6 or len(base_samples) < 5:  # Need more baseline samples
            return None

        # Use median for more stable baseline
        bx0 = int(round(np.median([p[0] for p in base_samples])))
        by0 = int(round(np.median([p[1] for p in base_samples])))
        s0 = float(np.median([p[2] for p in base_samples]))

        # MORE LENIENT THRESHOLDS (this is key!)
        drop_thr = max(0.12, 0.25 * (1.0 - s0))  # Was 0.08/0.18
        corr_min = max(0.25, s0 - 0.30)          # Was 0.35/0.22
        disp_thr = max(7.0, 0.055 * float(r))   # Was 5.0/0.045

        # Require MORE persistence to avoid false triggers
        bad_run = 0
        persist = 3  # Increased from 2

        for t in range(baseline_frames + 1, T):  # Start checking AFTER baseline
            sc = scores[t]
            lc = locs[t]
            if sc is None or lc is None:
                bad_run += 1
            else:
                dx = float(lc[0] - bx0)
                dy = float(lc[1] - by0)
                disp = float((dx * dx + dy * dy) ** 0.5)

                bad = (sc < (s0 - drop_thr)) or (sc < corr_min) or (disp >= disp_thr)
                if bad:
                    bad_run += 1
                else:
                    bad_run = 0

            if bad_run >= persist:
                # Return frame right before the match went bad
                impact_local = max(0, t - persist)
                return int(start + impact_local)

        return None
    finally:
        cap.release()


def _impact_from_ball_roi_motion(
    ball_xy: Tuple[float, float],
    *,
    start_frame: int,
    end_frame: int,
    roi_scale: float = 0.12,
) -> Optional[int]:
    """Estimate impact when ball component starts moving or disappears.
    
    IMPROVEMENTS:
    - More lenient area constraints (catches balls at different distances)
    - Better circularity threshold
    - More robust baseline
    """
    if not video_path_str:
        return None
    try:
        vp = Path(str(video_path_str)).expanduser().resolve()
        if not vp.exists():
            return None
    except Exception:
        return None

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return None

    try:
        start = max(0, int(start_frame))
        end = max(start + 3, int(end_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

        ok, first = cap.read()
        if not ok or first is None:
            return None

        h, w = first.shape[:2]
        if h <= 0 or w <= 0:
            return None

        bx, by = float(ball_xy[0]), float(ball_xy[1])
        r = int(round(min(h, w) * float(roi_scale)))
        r = max(40, min(160, r))

        def crop(img: "np.ndarray") -> Tuple[Optional["np.ndarray"], Tuple[int, int, int, int]]:
            y0 = max(0, int(round(by - r)))
            y1 = min(h, int(round(by + r)))
            x0 = max(0, int(round(bx - r)))
            x1 = min(w, int(round(bx + r)))
            if (y1 - y0) < 20 or (x1 - x0) < 20:
                return None, (x0, y0, x1, y1)
            return img[y0:y1, x0:x1], (x0, y0, x1, y1)

        def ball_component(roi_bgr: "np.ndarray") -> Optional[Tuple[float, float, float, float]]:
            """Return (cx, cy, area, circularity) for the most ball-like component."""
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            # Detect BOTH white and yellow balls
            mask_white = cv2.inRange(hsv, (0, 0, 160), (180, 90, 255))
            mask_yellow = cv2.inRange(hsv, (15, 50, 100), (45, 255, 255))
            mask = cv2.bitwise_or(mask_white, mask_yellow)
            
            mask = cv2.medianBlur(mask, 5)
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            H, W = mask.shape[:2]
            roi_area = float(H * W)

            # MORE LENIENT area bounds
            min_a = max(15.0, 0.00012 * roi_area)  # Was 18.0/0.00015
            max_a = max(min_a + 15.0, 0.01500 * roi_area)  # Was +10.0/0.01000

            best = None
            best_score = -1e9
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_a or area > max_a:
                    continue
                per = float(cv2.arcLength(cnt, True))
                if per <= 1e-6:
                    continue
                circ = float(4.0 * math.pi * area / (per * per))
                if circ < 0.40:  # MORE LENIENT (was 0.45)
                    continue

                m = cv2.moments(cnt)
                if abs(m.get("m00", 0.0)) < 1e-6:
                    continue
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])

                # Prefer larger + more circular
                score = 2.5 * area + 1500.0 * circ  # Increased weights
                if score > best_score:
                    best_score = score
                    best = (cx, cy, area, circ)

            return best

        roi0, _ = crop(first)
        if roi0 is None:
            return None

        prev_g = cv2.cvtColor(roi0, cv2.COLOR_BGR2GRAY)
        prev_g = cv2.GaussianBlur(prev_g, (5, 5), 0)

        comps: List[Optional[Tuple[float, float, float, float]]] = [ball_component(roi0)]
        energy: List[Optional[float]] = [None]

        cur_frame = start + 1
        while cur_frame <= end:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            roi, _ = crop(fr)
            if roi is None:
                comps.append(None)
                energy.append(None)
                cur_frame += 1
                continue

            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            diff = cv2.absdiff(g, prev_g)
            e = float(np.mean(diff))

            comps.append(ball_component(roi))
            energy.append(e)

            prev_g = g
            cur_frame += 1

        T = len(comps)
        if T < 4:
            return None

        # Establish baseline over MORE frames
        base_samples: List[Tuple[float, float]] = []
        for c in comps[: min(12, T)]:  # Was 8
            if c is None:
                continue
            base_samples.append((float(c[0]), float(c[1])))
        if len(base_samples) < 3:
            return None

        bx0 = float(np.median([p[0] for p in base_samples]))
        by0 = float(np.median([p[1] for p in base_samples]))

        # Movement threshold - slightly more lenient
        thr_px = max(5.5, 0.04 * float(r))  # Was 4.5/0.03
        move_persist = 3  # Was 2
        miss_persist = 3  # Was 2

        # Energy threshold
        e_vals = [float(e) for e in energy[1: min(T, 12)] if e is not None and math.isfinite(float(e))]
        mu = float(np.mean(e_vals)) if e_vals else 0.0
        sd = float(np.std(e_vals)) if len(e_vals) >= 2 else 0.0
        thr_e = max(mu + 2.5 * sd, mu + 5.0, 8.0)  # Slightly more lenient

        moved_run = 0
        miss_run = 0

        for t in range(0, T):
            c = comps[t]
            e = energy[t]

            if c is None:
                if e is not None and math.isfinite(float(e)) and float(e) >= thr_e:
                    miss_run += 1
                else:
                    miss_run = 0
                moved_run = 0
            else:
                cx, cy = float(c[0]), float(c[1])
                dist = float(((cx - bx0) ** 2 + (cy - by0) ** 2) ** 0.5)
                if dist >= thr_px:
                    moved_run += 1
                else:
                    moved_run = 0
                miss_run = 0

            if moved_run >= move_persist or miss_run >= miss_persist:
                impact_local = max(0, t - 1)
                return int(start + impact_local)

        return None
    finally:
        cap.release()


def _detect_ball_movement_frame(
    ball_xy: Tuple[float, float],
    *,
    start_frame: int,
    end_frame: int,
    roi_scale: float = 0.12,
) -> Optional[int]:
    """
    Detect the exact frame when the ball first starts moving.
    Returns the absolute frame index or None if not found.
    """
    if not video_path_str:
        return None
    try:
        vp = Path(str(video_path_str)).expanduser().resolve()
        if not vp.exists():
            return None
    except Exception:
        return None

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return None

    try:
        start = max(0, int(start_frame))
        end = max(start + 3, int(end_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

        ok, first = cap.read()
        if not ok or first is None:
            return None

        h, w = first.shape[:2]
        if h <= 0 or w <= 0:
            return None

        bx, by = float(ball_xy[0]), float(ball_xy[1])

        # ROI sized to track ball movement
        r = int(round(min(h, w) * float(roi_scale)))
        r = max(45, min(180, r))

        def crop(img: "np.ndarray") -> Tuple[Optional["np.ndarray"], Tuple[int, int, int, int]]:
            y0 = max(0, int(round(by - r)))
            y1 = min(h, int(round(by + r)))
            x0 = max(0, int(round(bx - r)))
            x1 = min(w, int(round(bx + r)))
            if (y1 - y0) < 25 or (x1 - x0) < 25:
                return None, (x0, y0, x1, y1)
            return img[y0:y1, x0:x1], (x0, y0, x1, y1)

        def find_ball_center(roi_bgr: "np.ndarray") -> Optional[Tuple[float, float]]:
            """Find ball center (ROI coords) using generous color + shape heuristics."""
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            mask_white = cv2.inRange(hsv, (0, 0, 150), (180, 100, 255))
            mask_yellow = cv2.inRange(hsv, (12, 45, 90), (50, 255, 255))
            mask = cv2.bitwise_or(mask_white, mask_yellow)

            mask = cv2.medianBlur(mask, 5)
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            H, W = mask.shape[:2]
            roi_area = float(H * W)

            min_a = max(12.0, 0.00010 * roi_area)
            max_a = max(min_a + 20.0, 0.02000 * roi_area)

            best_ball = None
            best_score = -1e9

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_a or area > max_a:
                    continue
                per = float(cv2.arcLength(cnt, True))
                if per <= 1e-6:
                    continue
                circ = float(4.0 * math.pi * area / (per * per))
                if circ < 0.35:
                    continue

                m = cv2.moments(cnt)
                if abs(m.get("m00", 0.0)) < 1e-6:
                    continue
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])

                score = 3.0 * area + 2000.0 * circ
                if score > best_score:
                    best_score = score
                    best_ball = (cx, cy)

            return best_ball

        ball_positions: List[Optional[Tuple[float, float]]] = []
        cur_frame = start

        while cur_frame <= end:
            if cur_frame == start:
                roi = crop(first)[0]
            else:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                roi = crop(fr)[0]

            if roi is None:
                ball_positions.append(None)
            else:
                ball_center = find_ball_center(roi)
                ball_positions.append(ball_center)

            cur_frame += 1

        T = len(ball_positions)
        if T < 5:
            return None

        baseline_samples: List[Tuple[float, float]] = []
        baseline_window = min(20, int(T * 0.40))
        for i in range(baseline_window):
            if ball_positions[i] is not None:
                baseline_samples.append(ball_positions[i])

        if len(baseline_samples) < 3:
            return None

        baseline_x = float(np.median([p[0] for p in baseline_samples]))
        baseline_y = float(np.median([p[1] for p in baseline_samples]))
        baseline_std_x = float(np.std([p[0] for p in baseline_samples]))
        baseline_std_y = float(np.std([p[1] for p in baseline_samples]))

        threshold_x = max(3.5, 3.0 * baseline_std_x, 0.04 * float(r))
        threshold_y = max(3.5, 3.0 * baseline_std_y, 0.04 * float(r))

        movement_persistence = 2
        movement_count = 0

        for i in range(baseline_window, T):
            pos = ball_positions[i]
            if pos is None:
                movement_count += 1
            else:
                cx, cy = pos
                dx = abs(cx - baseline_x)
                dy = abs(cy - baseline_y)
                if dx > threshold_x or dy > threshold_y:
                    movement_count += 1
                else:
                    movement_count = 0

            if movement_count >= movement_persistence:
                impact_frame = max(0, i - movement_persistence + 1)
                return int(start + impact_frame)

        return None
    finally:
        cap.release()

def _get_xy(pose: Dict[str, Any], frame_index: int, name: str) -> Tuple[Optional[float], Optional[float]]:
    frames = pose.get("frames", []) or []
    if frame_index < 0 or frame_index >= len(frames):
        return None, None
    lm = (frames[frame_index].get("landmarks") or {}).get(name) or {}
    x = lm.get("x")
    y = lm.get("y")
    if x is None or y is None:
        return None, None
    try:
        return float(x), float(y)
    except Exception:
        return None, None


def _midpoint(
    a: Tuple[Optional[float], Optional[float]],
    b: Tuple[Optional[float], Optional[float]],
) -> Tuple[Optional[float], Optional[float]]:
    ax, ay = a
    bx, by = b
    if ax is None or ay is None or bx is None or by is None:
        return None, None
    return (ax + bx) / 2.0, (ay + by) / 2.0



def _dist(
    a: Tuple[Optional[float], Optional[float]],
    b: Tuple[Optional[float], Optional[float]],
) -> Optional[float]:
    ax, ay = a
    bx, by = b
    if ax is None or ay is None or bx is None or by is None:
        return None
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


# Angle helper: angle ABC in degrees, where B is the vertex.
def _angle_between_deg(
    a: Tuple[Optional[float], Optional[float]],
    b: Tuple[Optional[float], Optional[float]],
    c: Tuple[Optional[float], Optional[float]],
) -> Optional[float]:
    """
    Angle ABC in degrees, where B is the vertex.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    if ax is None or ay is None or bx is None or by is None or cx is None or cy is None:
        return None

    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by

    n1 = (v1x * v1x + v1y * v1y) ** 0.5
    n2 = (v2x * v2x + v2y * v2y) ** 0.5
    if n1 <= 1e-9 or n2 <= 1e-9:
        return None

    dot = v1x * v2x + v1y * v2y
    cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
    return float(math.degrees(math.acos(cosv)))


def _angle_deg_from_vertical(
    a: Tuple[Optional[float], Optional[float]],
    b: Tuple[Optional[float], Optional[float]],
) -> Optional[float]:
    """
    Angle of vector a->b relative to vertical axis (degrees).
    0° means perfectly vertical (b below/above a), positive/negative indicate tilt.
    """
    ax, ay = a
    bx, by = b
    if ax is None or ay is None or bx is None or by is None:
        return None
    dx = bx - ax
    dy = by - ay
    # atan2(dx, dy) gives angle vs vertical (swap params vs usual atan2(dy,dx))
    return math.degrees(math.atan2(dx, dy))


def _angle_deg_shoulder_line(
    l_sh: Tuple[Optional[float], Optional[float]],
    r_sh: Tuple[Optional[float], Optional[float]],
) -> Optional[float]:
    """
    Shoulder line angle vs horizontal (degrees). 0° means level shoulders.
    """
    lx, ly = l_sh
    rx, ry = r_sh
    if lx is None or ly is None or rx is None or ry is None:
        return None
    dx = rx - lx
    dy = ry - ly
    return math.degrees(math.atan2(dy, dx))


def compute_golf_metrics(
    pose: Dict[str, Any],
    phases: Dict[str, Any],
    *,
    handedness: str = "right",
) -> Dict[str, Any]:
    """
    Compute a small set of reliable side-on golf metrics from pose + phases.
    Returns metrics with None values when a metric can't be computed robustly.
    """
    if not phases.get("ok"):
        return {"ok": False, "reason": phases.get("reason", "phase_detection_failed")}

    address_f = int(phases["address"]["frame"])
    top_f = int(phases["top"]["frame"])
    impact_f = int(phases["impact"]["frame"])

    # Reference scale: shoulder width at address
    l_sh_a = _get_xy(pose, address_f, "left_shoulder")
    r_sh_a = _get_xy(pose, address_f, "right_shoulder")
    shoulder_w = _dist(l_sh_a, r_sh_a)
    if shoulder_w is None or shoulder_w <= 1e-6:
        shoulder_w = None

    # Head sway (nose x shift) address -> impact (normalized by shoulder width)
    nose_a = _get_xy(pose, address_f, "nose")
    nose_i = _get_xy(pose, impact_f, "nose")
    head_sway_px = None
    head_sway_norm = None
    if nose_a[0] is not None and nose_i[0] is not None:
        head_sway_px = float(nose_i[0] - nose_a[0])
        if shoulder_w is not None:
            head_sway_norm = head_sway_px / shoulder_w

    # Hip sway (hip midpoint x shift) address -> impact (normalized)
    l_hip_a = _get_xy(pose, address_f, "left_hip")
    r_hip_a = _get_xy(pose, address_f, "right_hip")
    l_hip_i = _get_xy(pose, impact_f, "left_hip")
    r_hip_i = _get_xy(pose, impact_f, "right_hip")
    hip_mid_a = _midpoint(l_hip_a, r_hip_a)
    hip_mid_i = _midpoint(l_hip_i, r_hip_i)
    hip_sway_px = None
    hip_sway_norm = None
    if hip_mid_a[0] is not None and hip_mid_i[0] is not None:
        hip_sway_px = float(hip_mid_i[0] - hip_mid_a[0])
        if shoulder_w is not None:
            hip_sway_norm = hip_sway_px / shoulder_w

    # Spine angle change (shoulder-mid -> hip-mid tilt vs vertical), address vs impact
    sh_mid_a = _midpoint(l_sh_a, r_sh_a)
    sh_mid_i = _midpoint(
        _get_xy(pose, impact_f, "left_shoulder"),
        _get_xy(pose, impact_f, "right_shoulder"),
    )
    hip_mid_a2 = hip_mid_a
    hip_mid_i2 = hip_mid_i

    spine_a = _angle_deg_from_vertical(sh_mid_a, hip_mid_a2)
    spine_i = _angle_deg_from_vertical(sh_mid_i, hip_mid_i2)
    spine_angle_change_deg = None
    if spine_a is not None and spine_i is not None:
        spine_angle_change_deg = abs(float(spine_i - spine_a))

    # Shoulder "turn" proxy: change in shoulder-line angle from address -> top
    sh_line_a = _angle_deg_shoulder_line(l_sh_a, r_sh_a)
    sh_line_t = _angle_deg_shoulder_line(
        _get_xy(pose, top_f, "left_shoulder"),
        _get_xy(pose, top_f, "right_shoulder"),
    )
    shoulder_turn_proxy_deg = None
    if sh_line_a is not None and sh_line_t is not None:
        shoulder_turn_proxy_deg = abs(float(sh_line_t - sh_line_a))

    # Wrist angle proxy at impact (elbow -> wrist -> index). Experimental but useful for comparisons.
    # For right-handed golfers: lead = left, trail = right. For left-handed: lead = right, trail = left.
    if handedness.lower().startswith("l"):
        lead_elbow = _get_xy(pose, impact_f, "right_elbow")
        lead_wrist = _get_xy(pose, impact_f, "right_wrist")
        lead_index = _get_xy(pose, impact_f, "right_index")
        trail_elbow = _get_xy(pose, impact_f, "left_elbow")
        trail_wrist = _get_xy(pose, impact_f, "left_wrist")
        trail_index = _get_xy(pose, impact_f, "left_index")
    else:
        lead_elbow = _get_xy(pose, impact_f, "left_elbow")
        lead_wrist = _get_xy(pose, impact_f, "left_wrist")
        lead_index = _get_xy(pose, impact_f, "left_index")
        trail_elbow = _get_xy(pose, impact_f, "right_elbow")
        trail_wrist = _get_xy(pose, impact_f, "right_wrist")
        trail_index = _get_xy(pose, impact_f, "right_index")

    lead_wrist_angle_deg = _angle_between_deg(lead_elbow, lead_wrist, lead_index)
    trail_wrist_angle_deg = _angle_between_deg(trail_elbow, trail_wrist, trail_index)

    # Tempo metrics are already in phases
    tempo_ratio = (phases.get("metrics") or {}).get("tempo_ratio")
    backswing_sec = (phases.get("metrics") or {}).get("backswing_sec")
    downswing_sec = (phases.get("metrics") or {}).get("downswing_sec")

    return {
        "ok": True,
        "handedness": handedness,
        "tempo_ratio": tempo_ratio,
        "backswing_sec": backswing_sec,
        "downswing_sec": downswing_sec,
        "head_sway_px": round(head_sway_px, 2) if head_sway_px is not None else None,
        "head_sway_norm": round(head_sway_norm, 4) if head_sway_norm is not None else None,
        "hip_sway_px": round(hip_sway_px, 2) if hip_sway_px is not None else None,
        "hip_sway_norm": round(hip_sway_norm, 4) if hip_sway_norm is not None else None,
        "spine_angle_change_deg": round(spine_angle_change_deg, 2) if spine_angle_change_deg is not None else None,
        "shoulder_turn_proxy_deg": round(shoulder_turn_proxy_deg, 2) if shoulder_turn_proxy_deg is not None else None,
        "lead_wrist_angle_deg": round(lead_wrist_angle_deg, 2) if lead_wrist_angle_deg is not None else None,
        "trail_wrist_angle_deg": round(trail_wrist_angle_deg, 2) if trail_wrist_angle_deg is not None else None,
        "scale_ref_shoulder_width_px": round(shoulder_w, 2) if shoulder_w is not None else None,
    }


def generate_coaching_feedback(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert metrics into human-readable coaching tips.
    Status values: good | ok | needs_work | unknown
    """
    tips: List[Dict[str, Any]] = []

    # Tempo (common coaching target ~3:1, but we keep it as ranges)
    tempo = metrics.get("tempo_ratio")
    if tempo is None:
        tips.append({"title": "Tempo", "status": "unknown", "tip": "Could not estimate tempo reliably from this video."})
    else:
        if 2.6 <= float(tempo) <= 3.4:
            tips.append({"title": "Tempo", "status": "good", "tip": f"Tempo looks balanced (≈{tempo}:1). Keep the same rhythm."})
        elif float(tempo) < 2.6:
            tips.append({"title": "Tempo", "status": "needs_work", "tip": f"Downswing may be too quick (≈{tempo}:1). Try a smoother transition—count ‘1-2-3’ back, ‘1’ down."})
        else:
            tips.append({"title": "Tempo", "status": "ok", "tip": f"Backswing may be a bit long (≈{tempo}:1). Keep it athletic—avoid over-slowing at the top."})

    # Head sway (normalized)
    head_norm = metrics.get("head_sway_norm")
    if head_norm is None:
        tips.append({"title": "Head Stability", "status": "unknown", "tip": "Could not measure head stability reliably (head landmark missing in key frames)."})
    else:
        a = abs(float(head_norm))
        if a <= 0.06:
            tips.append({"title": "Head Stability", "status": "good", "tip": "Nice—head stayed centered through impact."})
        elif a <= 0.12:
            tips.append({"title": "Head Stability", "status": "ok", "tip": "Slight head movement—aim to stay a bit more centered while still shifting pressure."})
        else:
            tips.append({"title": "Head Stability", "status": "needs_work", "tip": "Noticeable head sway. Focus on turning around your spine rather than sliding laterally."})

    # Spine angle change
    spine = metrics.get("spine_angle_change_deg")
    if spine is None:
        tips.append({"title": "Posture (Spine Angle)", "status": "unknown", "tip": "Could not measure spine angle consistency reliably."})
    else:
        if float(spine) <= 4.0:
            tips.append({"title": "Posture (Spine Angle)", "status": "good", "tip": "Great posture retention—spine angle stayed stable into impact."})
        elif float(spine) <= 8.0:
            tips.append({"title": "Posture (Spine Angle)", "status": "ok", "tip": "Some posture change into impact. Think ‘chest down’ and maintain your hip hinge through the strike."})
        else:
            tips.append({"title": "Posture (Spine Angle)", "status": "needs_work", "tip": "Big posture change (possible early extension). Work on keeping your hips back and maintaining your hinge into impact."})

    # Hip sway (normalized)
    hip_norm = metrics.get("hip_sway_norm")
    if hip_norm is None:
        tips.append({"title": "Hip Sway", "status": "unknown", "tip": "Could not measure hip sway reliably."})
    else:
        a = abs(float(hip_norm))
        if a <= 0.06:
            tips.append({"title": "Hip Sway", "status": "good", "tip": "Lower body looks stable—nice control through impact."})
        elif a <= 0.12:
            tips.append({"title": "Hip Sway", "status": "ok", "tip": "A bit of hip slide—aim for more rotation while keeping the pelvis centered."})
        else:
            tips.append({"title": "Hip Sway", "status": "needs_work", "tip": "Large hip slide. Try feeling pressure shift without your hips drifting too far toward the ball/target."})

    # Shoulder turn proxy
    sh_turn = metrics.get("shoulder_turn_proxy_deg")
    if sh_turn is None:
        tips.append({"title": "Shoulder Turn", "status": "unknown", "tip": "Could not estimate shoulder turn from this angle reliably."})
    else:
        if float(sh_turn) >= 35.0:
            tips.append({"title": "Shoulder Turn", "status": "good", "tip": "Solid shoulder turn to the top—good range of motion."})
        elif float(sh_turn) >= 25.0:
            tips.append({"title": "Shoulder Turn", "status": "ok", "tip": "Decent turn—if you want more power, add a touch more rotation without swaying."})
        else:
            tips.append({"title": "Shoulder Turn", "status": "needs_work", "tip": "Turn looks limited. Try making a fuller shoulder rotation (lead shoulder under chin) while keeping balance."})

    # Wrist angle proxy (experimental)
    lead_w = metrics.get("lead_wrist_angle_deg")
    if lead_w is None:
        tips.append({"title": "Lead Wrist (Impact) [Experimental]", "status": "unknown", "tip": "Could not estimate lead wrist angle at impact (need clear hand landmarks)."})
    else:
        tips.append({"title": "Lead Wrist (Impact) [Experimental]", "status": "ok", "tip": f"Lead wrist proxy angle at impact: {lead_w}°. Use this to compare swings (higher often means more ‘released’ hands)."})

    return tips


def build_swing_report(
    pose: Dict[str, Any],
    phases: Dict[str, Any],
    *,
    handedness: str = "right",
) -> Dict[str, Any]:
    """
    End-to-end: phases -> metrics -> feedback.
    """
    metrics = compute_golf_metrics(pose, phases, handedness=handedness)
    feedback = generate_coaching_feedback(metrics if metrics.get("ok") else {})
    return {
        "phases": phases,
        "metrics": metrics,
        "feedback": feedback,
    }


# -----------------------------
# Option 1: Visual validation (keyframe images)
# -----------------------------

# Simple skeleton connections (subset of MediaPipe pose) using our KEEP_LANDMARKS names.
SKELETON_EDGES: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _lm_xy_vis(frame_item: Dict[str, Any], name: str) -> Tuple[Optional[int], Optional[int], float]:
    lm = (frame_item.get("landmarks") or {}).get(name) or {}
    x = lm.get("x")
    y = lm.get("y")
    vis = float(lm.get("vis") or 0.0)
    if x is None or y is None:
        return None, None, vis
    try:
        return int(round(float(x))), int(round(float(y))), vis
    except Exception:
        return None, None, vis


def draw_pose_overlay(
    image_bgr: "np.ndarray",
    frame_item: Dict[str, Any],
    *,
    min_vis: float = 0.4,
    label: Optional[str] = None,
) -> "np.ndarray":
    """
    Draw a lightweight pose overlay (skeleton + joints) on a single BGR frame.
    Returns a new image array (does not mutate the input).
    """
    out = image_bgr.copy()

    # Draw edges
    for a, b in SKELETON_EDGES:
        ax, ay, av = _lm_xy_vis(frame_item, a)
        bx, by, bv = _lm_xy_vis(frame_item, b)
        if ax is None or ay is None or bx is None or by is None:
            continue
        if av < min_vis or bv < min_vis:
            continue
        cv2.line(out, (ax, ay), (bx, by), (0, 255, 0), 2)

    # Draw joints
    for name in KEEP_LANDMARKS.keys():
        x, y, v = _lm_xy_vis(frame_item, name)
        if x is None or y is None:
            continue
        if v < min_vis:
            continue
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)

    # Label
    if label:
        cv2.putText(
            out,
            str(label),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(label),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def _read_frame_at(video_path: Path, frame_index: int) -> Optional["np.ndarray"]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _refine_phase_frame(
    frames: List[Dict[str, Any]],
    target: int,
    *,
    window: int = 4,
    min_vis: float = 0.4,
) -> int:
    """
    Nudge a phase frame to the nearest frame (within ±window) that has the best
    landmark visibility. This reduces cases where the chosen phase frame has
    missing wrists/hips and the key image looks wrong.
    """
    if not frames:
        return target

    lo = max(0, target - window)
    hi = min(len(frames) - 1, target + window)

    best_idx = target
    best_score = (-1.0, -1.0, 0)  # (visible_count, vis_sum, negative_distance)

    for i in range(lo, hi + 1):
        lm = frames[i].get("landmarks") or {}
        visible = 0
        vis_sum = 0.0
        for name in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            v = (lm.get(name) or {}).get("vis")
            x = (lm.get(name) or {}).get("x")
            y = (lm.get(name) or {}).get("y")
            if v is None or x is None or y is None:
                continue
            if float(v) >= min_vis:
                visible += 1
                vis_sum += float(v)

        # Prefer more visible points, then higher total visibility, then closeness to target
        score = (visible, vis_sum, -abs(i - target))
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)


def save_keyframe_images(
    pose: Dict[str, Any],
    phases: Dict[str, Any],
    *,
    out_dir: Path,
    prefix: str = "swing",
) -> Dict[str, str]:
    """
    Save 3 key images (address/top/impact) with pose overlay.
    Returns file paths as strings (useful for API responses).
    """
    meta = pose.get("meta", {}) or {}
    video_str = meta.get("video")
    if not video_str:
        raise ValueError("pose.meta.video missing; cannot render visuals")

    if not phases.get("ok"):
        raise ValueError(f"phase detection failed: {phases.get('reason', 'unknown')}")

    video_path = Path(str(video_str)).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Frame indices
    address_f = int(phases["address"]["frame"])
    top_f = int(phases["top"]["frame"])
    impact_f = int(phases["impact"]["frame"])

    frames = pose.get("frames", []) or []
    if address_f >= len(frames) or top_f >= len(frames) or impact_f >= len(frames):
        raise ValueError("phase frame index out of bounds for pose.frames")

    min_vis = float(meta.get("min_visibility") or 0.4)

    # Refine phase frames to nearest high-visibility frames for cleaner key images
    address_f = _refine_phase_frame(frames, address_f, window=5, min_vis=min_vis)
    top_f = _refine_phase_frame(frames, top_f, window=5, min_vis=min_vis)
    impact_f = _refine_phase_frame(frames, impact_f, window=5, min_vis=min_vis)

    # Read raw frames
    address_img = _read_frame_at(video_path, address_f)
    top_img = _read_frame_at(video_path, top_f)
    impact_img = _read_frame_at(video_path, impact_f)

    if address_img is None or top_img is None or impact_img is None:
        raise RuntimeError("Could not read one or more key frames from the video")

    # Overlay
    address_out = draw_pose_overlay(address_img, frames[address_f], min_vis=min_vis, label="ADDRESS")
    top_out = draw_pose_overlay(top_img, frames[top_f], min_vis=min_vis, label="TOP")
    impact_out = draw_pose_overlay(impact_img, frames[impact_f], min_vis=min_vis, label="IMPACT")

    # Save
    address_path = out_dir / f"{prefix}_address.jpg"
    top_path = out_dir / f"{prefix}_top.jpg"
    impact_path = out_dir / f"{prefix}_impact.jpg"

    cv2.imwrite(str(address_path), address_out)
    cv2.imwrite(str(top_path), top_out)
    cv2.imwrite(str(impact_path), impact_out)

    return {
        "address_image": str(address_path),
        "top_image": str(top_path),
        "impact_image": str(impact_path),
    }


# -----------------------------
# Option 2: Full annotated video (pose overlay on every frame)
# -----------------------------
def save_annotated_video(
    pose: Dict[str, Any],
    phases: Dict[str, Any],
    *,
    out_dir: Path,
    prefix: str = "swing",
    include_phase_labels: bool = False,
) -> Dict[str, Any]:
    """
    Save an annotated MP4 with pose overlay on every frame.
    Returns:
      {
        "video": "<path>",
        "frame_count_written": <int>,
        "fps": <float>,
        "width": <int>,
        "height": <int>
      }
    """
    meta = pose.get("meta", {}) or {}
    video_str = meta.get("video")
    if not video_str:
        raise ValueError("pose.meta.video missing; cannot render annotated video")

    video_path = Path(str(video_str)).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found for annotation: {video_path}")

    frames_pose = pose.get("frames", []) or []
    if not frames_pose:
        raise ValueError("pose.frames is empty; cannot render annotated video")

    out_dir.mkdir(parents=True, exist_ok=True)

    fps = float(meta.get("fps") or 30.0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    min_vis = float(meta.get("min_visibility") or 0.4)

    # Phase frame indices (optional labels)
    address_f = int(phases["address"]["frame"]) if phases.get("ok") else None
    top_f = int(phases["top"]["frame"]) if phases.get("ok") else None
    impact_f = int(phases["impact"]["frame"]) if phases.get("ok") else None

    # Show labels for a short window so they’re visible (default ~0.3s)
    label_window_frames = max(10, int(round(fps * 0.30)))

    # Output writer
    # We write a "raw" MP4 first. Browsers often require H.264, so we may transcode via ffmpeg.
    raw_path = out_dir / f"{prefix}_annotated_raw.mp4"
    out_path = out_dir / f"{prefix}_annotated.mp4"

    # Try a more compatible codec first (avc1), then fall back to mp4v.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for output: {raw_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video for reading: {video_path}")

    written = 0
    try:
        i = 0
        # Read sequentially; overlay pose for corresponding frame index.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= len(frames_pose):
                break

            label = None
            if include_phase_labels and phases.get("ok"):
                # Persist labels for a window around each key frame so users can actually see them.
                # Priority: IMPACT > TOP > ADDRESS (in case windows overlap)
                if impact_f is not None and abs(i - impact_f) <= label_window_frames:
                    label = "IMPACT"
                elif top_f is not None and abs(i - top_f) <= label_window_frames:
                    label = "TOP"
                elif address_f is not None and abs(i - address_f) <= label_window_frames:
                    label = "ADDRESS"

            annotated = draw_pose_overlay(
                frame,
                frames_pose[i],
                min_vis=min_vis,
                label=label,
            )

            writer.write(annotated)
            written += 1
            i += 1
    finally:
        cap.release()
        writer.release()

    # If ffmpeg is available, transcode to H.264 for best browser compatibility.
    # If not, fall back to the raw file we wrote.
    final_video_path = raw_path
    transcode_info: Dict[str, Any] = {"attempted": False, "ok": False, "encoder": None, "stderr_tail": None}

    ffmpeg_path = shutil.which("ffmpeg")

    # On macOS, uvicorn may run with a limited PATH that doesn't include Homebrew.
    # Try common Homebrew locations as a fallback.
    if not ffmpeg_path:
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if Path(candidate).exists():
                ffmpeg_path = candidate
                break
    if ffmpeg_path:
        transcode_info["attempted"] = True

        def _run_ffmpeg(encoder: str) -> Tuple[bool, str]:
            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                str(raw_path),
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(out_path),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stderr = proc.stderr or ""
            ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
            return ok, stderr

        # Try software H.264 first, then fall back to Apple's hardware encoder (common on macOS).
        ok, stderr = _run_ffmpeg("libx264")
        if not ok:
            ok, stderr2 = _run_ffmpeg("h264_videotoolbox")
            if ok:
                transcode_info["encoder"] = "h264_videotoolbox"
                stderr = stderr2
        else:
            transcode_info["encoder"] = "libx264"

        # Record a short tail of stderr for debugging
        if stderr:
            transcode_info["stderr_tail"] = stderr.splitlines()[-20:]

        if ok:
            transcode_info["ok"] = True
            final_video_path = out_path
            # Clean up raw file
            try:
                raw_path.unlink()
            except OSError:
                pass
        else:
            # Print a hint in server logs so it's obvious why we fell back to raw.
            print("⚠️ ffmpeg transcode failed; serving raw annotated video. See stderr tail below:")
            if transcode_info["stderr_tail"]:
                for line in transcode_info["stderr_tail"]:
                    print(line)

    return {
        "video": str(final_video_path),
        "frame_count_written": int(written),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "transcode": transcode_info,
    }