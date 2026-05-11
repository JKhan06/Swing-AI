"""
Reference swing matching: compare a golfer's swing against a database
of stored reference pose snapshots and return per-phase similarity scores.

Algorithm
---------
1. At each key phase (address, top, impact) extract a normalized pose vector.
   Normalization: translate to hip-centre origin, scale by shoulder width,
   mirror x-axis for left-handers so all poses live in "right-handed space".
2. Compare to every reference swing in the DB via cosine similarity.
3. Average per-phase similarities, weight them, and map to a 0-100 score.

The DB is a plain JSON file stored at backend/reference_db.json.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DB_PATH = Path(__file__).resolve().parents[2] / "reference_db.json"

# Landmarks used for comparison — everything MediaPipe reliably tracks from side-on
COMPARE_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]

PHASE_WEIGHTS = {"address": 0.25, "top": 0.40, "impact": 0.35}

# How many of the 13 landmarks must be present to compute a valid vector
MIN_VALID_LANDMARKS = 8


def _normalize_frame(landmarks: Dict[str, Any], handedness: str = "right") -> Optional[np.ndarray]:
    """
    Convert a frame's pixel-coordinate landmarks to a scale/position-invariant vector.

    Returns a float32 array of shape (len(COMPARE_LANDMARKS)*2,), or None if
    anchors are missing or too few landmarks are visible.
    """
    def get_xy(name: str) -> Optional[tuple]:
        lm = landmarks.get(name, {})
        x, y = lm.get("x"), lm.get("y")
        return (float(x), float(y)) if x is not None and y is not None else None

    l_sh  = get_xy("left_shoulder")
    r_sh  = get_xy("right_shoulder")
    l_hip = get_xy("left_hip")
    r_hip = get_xy("right_hip")

    # Need at least one hip to centre, at least one shoulder for scale
    if l_sh is None and r_sh is None:
        return None
    if l_hip is None and r_hip is None:
        return None

    # Hip midpoint → translation anchor
    if l_hip is not None and r_hip is not None:
        cx = (l_hip[0] + r_hip[0]) / 2.0
        cy = (l_hip[1] + r_hip[1]) / 2.0
    elif l_hip is not None:
        cx, cy = l_hip
    else:
        cx, cy = r_hip  # type: ignore[misc]

    # Shoulder width → scale reference
    if l_sh is not None and r_sh is not None:
        scale = math.dist(l_sh, r_sh)
    elif l_sh is not None and r_sh is None:
        scale = math.dist(l_sh, (cx, cy)) * 2.0
    else:
        scale = math.dist(r_sh, (cx, cy)) * 2.0  # type: ignore[arg-type]

    if scale < 1e-6:
        return None

    # Count usable landmarks
    valid = sum(1 for n in COMPARE_LANDMARKS if get_xy(n) is not None)
    if valid < MIN_VALID_LANDMARKS:
        return None

    vec = []
    for name in COMPARE_LANDMARKS:
        xy = get_xy(name)
        if xy is not None:
            nx = (xy[0] - cx) / scale
            ny = (xy[1] - cy) / scale
        else:
            nx, ny = 0.0, 0.0   # missing → hip-centre placeholder

        # Mirror right→left so all poses are comparable regardless of handedness
        if handedness == "left":
            nx = -nx
        vec.extend([nx, ny])

    return np.array(vec, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Public helpers ─────────────────────────────────────────────────────────────

def extract_phase_vectors(
    pose: Dict[str, Any],
    phases: Dict[str, Any],
    handedness: str = "right",
) -> Dict[str, Optional[List[float]]]:
    """
    Extract normalized pose vectors at address, top, and impact.
    Returns {phase: list[float] | None}.
    """
    frames = pose.get("frames", [])
    result: Dict[str, Optional[List[float]]] = {}

    for phase_name in ("address", "top", "impact"):
        info = phases.get(phase_name)
        if not info:
            result[phase_name] = None
            continue
        fi = info.get("frame")
        if fi is None or fi >= len(frames):
            result[phase_name] = None
            continue
        lms = frames[fi].get("landmarks", {})
        vec = _normalize_frame(lms, handedness)
        result[phase_name] = vec.tolist() if vec is not None else None

    return result


def score_against_reference(
    query_vectors: Dict[str, Optional[List[float]]],
    db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare query phase vectors to every reference swing in db.

    Returns:
        {
          "overall":  float 0–100 or None,
          "phases":   {"address": float|None, "top": float|None, "impact": float|None},
          "n_references": int
        }
    """
    swings = db.get("swings", [])
    if not swings:
        return {
            "overall": None,
            "phases": {p: None for p in PHASE_WEIGHTS},
            "n_references": 0,
        }

    phase_sims: Dict[str, List[float]] = {p: [] for p in PHASE_WEIGHTS}

    for ref in swings:
        ref_vecs = ref.get("phase_vectors", {})
        for phase in PHASE_WEIGHTS:
            q = query_vectors.get(phase)
            r = ref_vecs.get(phase)
            if q is None or r is None:
                continue
            qa = np.array(q, dtype=np.float32)
            ra = np.array(r, dtype=np.float32)
            if qa.shape != ra.shape:
                continue
            phase_sims[phase].append(_cosine_sim(qa, ra))

    phase_scores: Dict[str, Optional[float]] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for phase, weight in PHASE_WEIGHTS.items():
        sims = phase_sims[phase]
        if sims:
            avg = float(np.mean(sims))
            # Cosine sim for normalized human poses clusters around [0.75, 1.0].
            # Map that window linearly to [0, 100].
            score = round(max(0.0, min(100.0, (avg - 0.75) / 0.25 * 100.0)), 1)
            phase_scores[phase] = score
            weighted_sum += score * weight
            weight_total += weight
        else:
            phase_scores[phase] = None

    overall = round(weighted_sum / weight_total, 1) if weight_total > 0 else None
    return {"overall": overall, "phases": phase_scores, "n_references": len(swings)}


# ── Database I/O ──────────────────────────────────────────────────────────────

def load_reference_db() -> Dict[str, Any]:
    if not DB_PATH.exists():
        return {"version": 1, "swings": []}
    with open(DB_PATH) as f:
        return json.load(f)


def save_reference_db(db: Dict[str, Any]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f)


def add_swing_to_db(
    db: Dict[str, Any],
    pose: Dict[str, Any],
    phases: Dict[str, Any],
    handedness: str = "right",
    label: str = "reference",
) -> bool:
    """
    Extract phase vectors from a pose dict and append to db in-place.
    Returns True if the swing was successfully added (impact vector found).
    """
    vectors = extract_phase_vectors(pose, phases, handedness)
    if vectors.get("impact") is None:
        return False

    db.setdefault("swings", []).append({
        "label": label,
        "handedness": handedness,
        "phase_vectors": vectors,
    })
    return True
