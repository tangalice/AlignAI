"""
FormAI Compare - pure Python logic (no Modal).

Used by modal_compare_app and test_compare --local.
"""

import math
from typing import Any

# MediaPipe landmark indices (33 landmarks)
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

EXERCISE_WEIGHTS: dict[str, dict[int, float]] = {
    "squat": {
        LEFT_KNEE: 1.5, RIGHT_KNEE: 1.5,
        LEFT_HIP: 1.3, RIGHT_HIP: 1.3,
        LEFT_ANKLE: 1.0, RIGHT_ANKLE: 1.0,
    },
    "lunge": {
        LEFT_KNEE: 1.5, RIGHT_KNEE: 1.5,
        LEFT_HIP: 1.2, RIGHT_HIP: 1.2,
    },
    "pushup": {
        LEFT_ELBOW: 1.5, RIGHT_ELBOW: 1.5,
        LEFT_SHOULDER: 1.3, RIGHT_SHOULDER: 1.3,
    },
    "bicep": {
        LEFT_ELBOW: 1.8, RIGHT_ELBOW: 1.8,
        LEFT_SHOULDER: 1.2, RIGHT_SHOULDER: 1.2,
    },
    "overhead": {
        LEFT_ELBOW: 1.3, RIGHT_ELBOW: 1.3,
        LEFT_SHOULDER: 1.5, RIGHT_SHOULDER: 1.5,
    },
    "jump": {
        LEFT_KNEE: 1.2, RIGHT_KNEE: 1.2,
        LEFT_HIP: 1.2, RIGHT_HIP: 1.2,
        LEFT_ANKLE: 1.0, RIGHT_ANKLE: 1.0,
    },
    "default": {},
}

# Hit tiers for UI feedback (similar to JiggleWiggle)
HIT_TIER_THRESHOLDS = [
    (90, "PERFECT"),
    (80, "GREAT"),
    (70, "OK"),
    (50, "ALMOST"),
]
EMA_ALPHA = 0.15  # Smoothing: lower = smoother, higher = more responsive


def _score_to_hit_tier(score: float) -> str:
    for threshold, tier in HIT_TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "MISS"


def _joint_score(ref_angle: float, user_angle: float) -> float:
    """0–100 score from angle difference; 0° diff = 100, ~50° diff ≈ 0."""
    diff = abs(ref_angle - user_angle)
    return max(0.0, min(100.0, 100.0 - diff * 2.0))


def _angle_between_points(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2) or 1e-9
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2) or 1e-9
    cos_a = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


def _extract_joint_angles(landmarks: list[list[float]]) -> dict[str, float]:
    if len(landmarks) < 29:
        return {}
    pts = [(lm[0], lm[1]) for lm in landmarks]
    angles = {}
    angles["left_elbow"] = _angle_between_points(pts[LEFT_SHOULDER], pts[LEFT_ELBOW], pts[LEFT_WRIST])
    angles["right_elbow"] = _angle_between_points(pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW], pts[RIGHT_WRIST])
    angles["left_knee"] = _angle_between_points(pts[LEFT_HIP], pts[LEFT_KNEE], pts[LEFT_ANKLE])
    angles["right_knee"] = _angle_between_points(pts[RIGHT_HIP], pts[RIGHT_KNEE], pts[RIGHT_ANKLE])
    angles["left_hip"] = _angle_between_points(pts[LEFT_SHOULDER], pts[LEFT_HIP], pts[LEFT_KNEE])
    angles["right_hip"] = _angle_between_points(pts[RIGHT_SHOULDER], pts[RIGHT_HIP], pts[RIGHT_KNEE])
    angles["left_shoulder"] = _angle_between_points(pts[LEFT_HIP], pts[LEFT_SHOULDER], pts[LEFT_ELBOW])
    angles["right_shoulder"] = _angle_between_points(pts[RIGHT_HIP], pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW])
    return angles


def _body_normalization(landmarks: list[list[float]]) -> dict[str, float]:
    if len(landmarks) < 29:
        return {"shoulder_hip": 1.0, "arm_length": 1.0, "torso": 1.0}
    pts = [(lm[0], lm[1]) for lm in landmarks]

    def dist(i, j):
        return math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)

    mid_shoulder = (
        (pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) / 2,
        (pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) / 2,
    )
    mid_hip = (
        (pts[LEFT_HIP][0] + pts[RIGHT_HIP][0]) / 2,
        (pts[LEFT_HIP][1] + pts[RIGHT_HIP][1]) / 2,
    )
    shoulder_hip = math.sqrt(
        (mid_shoulder[0] - mid_hip[0]) ** 2 + (mid_shoulder[1] - mid_hip[1]) ** 2
    ) or 1e-6
    arm_l = (
        dist(LEFT_SHOULDER, LEFT_ELBOW) + dist(LEFT_ELBOW, LEFT_WRIST)
        + dist(RIGHT_SHOULDER, RIGHT_ELBOW) + dist(RIGHT_ELBOW, RIGHT_WRIST)
    ) / 4 or 1e-6
    return {"shoulder_hip": shoulder_hip, "arm_length": arm_l, "torso": shoulder_hip}


def _get_exercise_weights(exercise: str) -> dict[int, float]:
    ex_lower = (exercise or "").lower()
    for key, weights in EXERCISE_WEIGHTS.items():
        if key != "default" and key in ex_lower:
            return weights
    return EXERCISE_WEIGHTS["default"]


def _extract_pose_matrix(frames: list[dict]) -> list[list[list[float]]]:
    out = []
    for f in frames:
        lm = f.get("landmarks")
        if not lm:
            continue
        arr = []
        for pt in lm:
            if len(pt) >= 3:
                arr.append([float(pt[0]), float(pt[1]), float(pt[2])])
            else:
                arr.append([float(pt[0]), float(pt[1]), 0.0])
        if len(arr) >= 33:
            out.append(arr[:33])
        elif len(arr) > 0:
            out.append(arr)
    return out


def _frame_distance(
    ref_lm: list[list[float]],
    user_lm: list[list[float]],
    ref_norm: dict[str, float],
    user_norm: dict[str, float],
    ref_angles: dict[str, float],
    user_angles: dict[str, float],
    weights: dict[int, float],
) -> float:
    n = min(len(ref_lm), len(user_lm))
    if n == 0:
        return 0.0
    scale = (ref_norm["shoulder_hip"] + user_norm["shoulder_hip"]) / 2
    scale = max(scale, 1e-6)
    lm_dist = 0.0
    total_w = 0.0
    for i in range(n):
        r = ref_lm[i]
        u = user_lm[i]
        d = math.sqrt(sum((r[j] - u[j]) ** 2 for j in range(min(3, len(r), len(u))))) / scale
        w = weights.get(i, 1.0)
        lm_dist += d * w
        total_w += w
    lm_dist /= total_w if total_w > 0 else n
    angle_dist = 0.0
    angle_keys = list(ref_angles.keys() & user_angles.keys())
    if angle_keys:
        for k in angle_keys:
            angle_dist += abs(ref_angles[k] - user_angles[k])
        angle_dist /= len(angle_keys)
        angle_dist = angle_dist / 180.0
    return 0.7 * lm_dist + 0.3 * angle_dist


def _dtw(
    ref_frames: list, user_frames: list, ref_poses: list, user_poses: list, weights: dict
) -> tuple[list[tuple[int, int]], float]:
    R, U = len(ref_poses), len(user_poses)
    if R == 0 or U == 0:
        return [], 0.0
    inf = float("inf")
    cost = [[inf] * (U + 1) for _ in range(R + 1)]
    cost[0][0] = 0.0
    for i in range(R):
        for j in range(U):
            ref_norm = _body_normalization(ref_poses[i])
            user_norm = _body_normalization(user_poses[j])
            ref_ang = _extract_joint_angles(ref_poses[i])
            user_ang = _extract_joint_angles(user_poses[j])
            d = _frame_distance(
                ref_poses[i], user_poses[j], ref_norm, user_norm, ref_ang, user_ang, weights
            )
            cost[i + 1][j + 1] = d + min(cost[i][j], cost[i][j + 1], cost[i + 1][j])
    path = []
    i, j = R, U
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev = min(
                (cost[i - 1][j - 1], i - 1, j - 1),
                (cost[i - 1][j], i - 1, j),
                (cost[i][j - 1], i, j - 1),
                key=lambda x: x[0],
            )
            i, j = prev[1], prev[2]
    path.reverse()
    return path, cost[R][U]


def compare_poses(reference: dict, user: dict, exercise: str = "") -> dict[str, Any]:
    """Full comparison pipeline. Same logic as Modal compare."""
    ref_frames = reference.get("frames") or []
    user_frames = user.get("frames") or []
    ref_poses = _extract_pose_matrix(ref_frames)
    user_poses = _extract_pose_matrix(user_frames)
    if not ref_poses or not user_poses:
        return {
            "score": 0,
            "hit_tier": "MISS",
            "per_frame": [],
            "limb_scores": {},
            "message": "Missing pose data",
            "joint_angles_ref": {},
            "joint_angles_user": {},
        }
    weights = _get_exercise_weights(exercise)
    path, total_cost = _dtw(ref_frames, user_frames, ref_poses, user_poses, weights)
    avg_cost = total_cost / len(path) if path else 0.0
    overall_score = max(0, min(100, round(100 - avg_cost * 250)))
    per_frame_raw = []
    step = max(1, len(path) // 20)
    for idx in range(0, len(path), step):
        ri, ui = path[idx]
        ref_norm = _body_normalization(ref_poses[ri])
        user_norm = _body_normalization(user_poses[ui])
        ref_ang = _extract_joint_angles(ref_poses[ri])
        user_ang = _extract_joint_angles(user_poses[ui])
        d = _frame_distance(
            ref_poses[ri], user_poses[ui], ref_norm, user_norm, ref_ang, user_ang, weights
        )
        frame_score = max(0, min(100, round(100 - d * 250)))
        joint_scores = {
            k: round(_joint_score(ref_ang[k], user_ang[k]), 1)
            for k in ref_ang.keys() & user_ang.keys()
        }
        per_frame_raw.append({
            "ref_frame": ri,
            "user_frame": ui,
            "score": frame_score,
            "distance": round(d, 4),
            "ref_angles": {k: round(v, 1) for k, v in ref_ang.items()},
            "user_angles": {k: round(v, 1) for k, v in user_ang.items()},
            "joint_scores": joint_scores,
        })
    # EMA smoothing of per-frame scores
    per_frame = []
    smoothed = None
    for i, pf in enumerate(per_frame_raw):
        raw = float(pf["score"])
        if smoothed is None:
            smoothed = raw
        else:
            smoothed = EMA_ALPHA * raw + (1 - EMA_ALPHA) * smoothed
        pf["score"] = round(smoothed, 1)
        per_frame.append(pf)
    # Aggregate limb scores (average per joint across sampled frames)
    limb_scores: dict[str, float] = {}
    if per_frame:
        joint_keys = next((pf["joint_scores"].keys() for pf in per_frame if pf.get("joint_scores")), [])
        for k in joint_keys:
            vals = [pf["joint_scores"][k] for pf in per_frame if k in pf.get("joint_scores", {})]
            if vals:
                limb_scores[k] = round(sum(vals) / len(vals), 1)
    ref_center = len(ref_poses) // 2
    user_center = len(user_poses) // 2
    hit_tier = _score_to_hit_tier(overall_score)
    return {
        "score": overall_score,
        "hit_tier": hit_tier,
        "per_frame": per_frame,
        "limb_scores": limb_scores,
        "path_length": len(path),
        "avg_cost": round(avg_cost, 4),
        "message": f"Form similarity: {overall_score}% ({hit_tier}, aligned {len(path)} pairs)",
        "joint_angles_ref": {
            k: round(v, 1) for k, v in _extract_joint_angles(ref_poses[ref_center]).items()
        },
        "joint_angles_user": {
            k: round(v, 1) for k, v in _extract_joint_angles(user_poses[user_center]).items()
        },
    }
