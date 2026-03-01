#!/usr/bin/env python3
"""
Quick test for pose compare pipeline.

Tests:
  1. Perfect match: reference == user (same poses) → expect score ~95-100%
  2. Partial match: user = first half of reference → expect high score
  3. Noisy match: user = reference + small noise → expect slightly lower score
  4. Empty user → expect 0 or error

Requires pose_server running on port 8001. Compare runs locally (no Modal).

Run (from project root):
  python compare/test_compare.py              # Use pose_server /api/pose/compare
  python compare/test_compare.py --local      # Run compare logic in-process (no HTTP)
  python compare/test_compare.py --full       # Use /api/pose/compare-full (preprocess+compare)
"""

import argparse
import json
import os
import sys

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

API_BASE = os.environ.get("API_BASE", "http://localhost:8001")
YT_URL = "https://www.youtube.com/shorts/qNBKJ9jvn_0"
SAMPLE_FPS = 8.0  # Lower = faster preprocess


def preprocess(url: str) -> dict:
    """Get pose frames from YouTube video via preprocess API."""
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f"{API_BASE}/api/preprocess",
            json={"url": url, "sample_fps": SAMPLE_FPS},
        )
        r.raise_for_status()
        return r.json()


def compare_api(reference: dict, user: dict, exercise: str = "") -> dict:
    """Call pose compare API (proxies to Modal)."""
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f"{API_BASE}/api/pose/compare",
            json={"reference": reference, "user": user, "exercise": exercise},
        )
        r.raise_for_status()
        return r.json()


def compare_full_api(youtube_url: str, user: dict, exercise: str = "", sample_fps: float = 8.0) -> dict:
    """Call full compare API (preprocess+compare on Modal)."""
    with httpx.Client(timeout=300) as client:
        r = client.post(
            f"{API_BASE}/api/pose/compare-full",
            json={"youtube_url": youtube_url, "user": user, "exercise": exercise, "sample_fps": sample_fps},
        )
        r.raise_for_status()
        return r.json()


def compare_local(reference: dict, user: dict, exercise: str = "") -> dict:
    """Run compare logic locally (no Modal)."""
    from compare_core import compare_poses
    return compare_poses(reference, user, exercise)


def add_noise(frames: list[dict], scale: float = 0.02) -> list[dict]:
    """Add small random noise to landmarks for testing."""
    import random

    out = []
    for f in frames:
        lm = f.get("landmarks", [])
        noisy = []
        for pt in lm:
            x, y = float(pt[0]), float(pt[1])
            z = float(pt[2]) if len(pt) > 2 else 0.0
            noisy.append([
                x + (random.random() - 0.5) * scale,
                y + (random.random() - 0.5) * scale,
                z + (random.random() - 0.5) * scale * 0.5,
            ])
        out.append({"index": len(out), "t": f.get("t", 0), "ms": f.get("ms", 0), "landmarks": noisy})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test FormAI compare pipeline")
    parser.add_argument("--local", action="store_true", help="Run compare locally (no Modal)")
    parser.add_argument("--full", action="store_true", help="Full pipeline on Modal (preprocess+compare)")
    args = parser.parse_args()
    use_local = args.local
    use_full = args.full

    if use_full:
        compare_fn = lambda ref, user, ex="": compare_full_api(YT_URL, user, ex, SAMPLE_FPS)
    elif use_local:
        compare_fn = compare_local
    else:
        compare_fn = compare_api

    print("=" * 60)
    print("FormAI Compare Model Test")
    print("=" * 60)
    mode = " (full Modal)" if use_full else (" (local compare)" if use_local else "")
    print(f"API: {API_BASE}{mode}")
    print(f"YouTube: {YT_URL}")
    print()

    # 1. Preprocess (get frames for user; for --full, Modal will get reference from URL)
    print("1. Preprocessing YouTube video (for user frames)...")
    try:
        data = preprocess(YT_URL)
    except httpx.HTTPStatusError as e:
        print(f"   FAIL: {e}")
        try:
            print(f"   Response: {e.response.text[:500]}")
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL: {e}")
        print("   Hint: Is pose_server running on port 8001?")
        sys.exit(1)

    frames = data.get("frames", [])
    if not frames:
        print("   FAIL: No pose frames extracted (video may have no detectable poses)")
        sys.exit(1)

    print(f"   OK: {len(frames)} frames extracted")
    ref = {"frames": frames}
    print()

    # 2. Test: Perfect match (reference == user)
    try:
        result = compare_fn(ref, ref)
        score = result.get("score", -1)
        print(f"   Score: {score}%")
        if score >= 90:
            print("   PASS: Expected ~95-100% for identical poses")
        else:
            print(f"   WARN: Expected ~95-100%, got {score}%")
        print(f"   Message: {result.get('message', '')}")
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"   FAIL: {e}")
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 503:
            print("   Hint: Is pose_server running on port 8001?")
    print()

    # 3. Test: Partial match (user = subset)
    print("3. Test: Partial match (user = first half of reference)")
    half = len(frames) // 2
    if half > 0:
        user_half = {"frames": [{"index": i, "t": f["t"], "ms": f["ms"], "landmarks": f["landmarks"]}
                     for i, f in enumerate(frames[:half])]}
        try:
            result = compare_fn(ref, user_half)
            score = result.get("score", -1)
            print(f"   Score: {score}%")
            if score >= 50:
                print("   PASS: Partial overlap should still yield reasonable score")
            else:
                print(f"   INFO: Score {score}% for half-length user")
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"   FAIL: {e}")
    else:
        print("   SKIP: Too few frames")
    print()

    # 4. Test: Noisy match
    print("4. Test: Noisy match (reference vs reference + small noise)")
    noisy_frames = add_noise(frames, scale=0.02)
    user_noisy = {"frames": noisy_frames}
    try:
        result = compare_fn(ref, user_noisy)
        score = result.get("score", -1)
        print(f"   Score: {score}%")
        if score >= 70:
            print("   PASS: Noisy poses should still score reasonably high")
        else:
            print(f"   INFO: Score {score}% with 2% landmark noise")
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"   FAIL: {e}")
    print()

    # 5. Test: Empty user
    print("5. Test: Empty user (should return 0 or low score)")
    try:
        result = compare_fn(ref, {"frames": []})
        score = result.get("score", -1)
        print(f"   Score: {score}%")
        if score == 0:
            print("   PASS: Empty user correctly returns 0")
        else:
            print(f"   INFO: Empty user returned {score}%")
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"   (Expected: {e})")
    print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
