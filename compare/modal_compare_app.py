"""
Modal app: preprocess (video → pose frames) and compare_full (YouTube + compare).

Endpoints:
  - preprocess: YouTube URL, video URL, or video bytes → pose frames (GPU for YOLOv8)
  - compare_full: YouTube URL + user → extract reference poses → compare (GPU for YOLOv8)

Compare runs locally on the backend (lightweight DTW); only heavy video/pose extraction uses Modal.
GPU (T4) used for fast YOLOv8 pose detection.

Deploy from project root:
  modal deploy compare.modal_compare_app

Set MODAL_PREPROCESS_ENDPOINT and MODAL_COMPARE_FULL_ENDPOINT in .env
"""

import re
import sys
from pathlib import Path
from typing import Optional

import modal
from fastapi.responses import JSONResponse
from starlette.requests import Request

app = modal.App("formai-compare")

# When running from compare/, parent is the compare dir
_compare_dir = Path(__file__).parent

# GPU image: yt-dlp, opencv, ultralytics, mediapipe
image_compare_full = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "opencv-python-headless", "numpy", "yt-dlp", "ultralytics>=8.0.0",
        "mediapipe", "fastapi",
    )
    .add_local_dir(str(_compare_dir), remote_path="/root/compare")
)


def _get_compare_core():
    sys.path.insert(0, "/root")
    from compare.compare_core import compare_poses
    return compare_poses


def _extract_youtube_video_id(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _get_youtube_stream_url(video_id: str) -> str:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,
        "format": "best[ext=mp4][acodec!=none][vcodec!=none]/best[acodec!=none][vcodec!=none]",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise ValueError("Video not found")
    if info.get("url"):
        return info["url"]
    formats = info.get("formats") or []
    for f in reversed(formats):
        if f.get("url") and (f.get("vcodec") or "none") != "none":
            return f["url"]
    raise ValueError("No stream URL found")


def _run_pose_preprocess(cap, sample_fps: float) -> list:
    """Run pose detection on video. Uses YOLOv8-pose (GPU) if available, else MediaPipe."""
    import cv2

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    frames_out = []
    frame_idx = 0

    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n-pose.pt")
        use_half = False
        try:
            import torch
            use_half = torch.cuda.is_available()
        except ImportError:
            pass

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            t_ms = frame_idx / video_fps * 1000.0
            frame_small = cv2.resize(frame_bgr, (256, 256))
            results = model(frame_small, verbose=False, imgsz=256, half=use_half)
            kps = results[0].keypoints
            xyn = kps.xyn
            conf = kps.conf
            if xyn is not None and len(xyn) > 0:
                xy = xyn[0].cpu().numpy()
                cf = conf[0].cpu().numpy() if conf is not None else None
                landmarks = []
                for i in range(xy.shape[0]):
                    x, y = float(xy[i][0]), float(xy[i][1])
                    c = float(cf[i]) if cf is not None and i < len(cf) else 1.0
                    landmarks.append([x, y, c])
                if landmarks:
                    frames_out.append({
                        "index": len(frames_out),
                        "t": round(t_ms / 1000.0, 3),
                        "ms": round(t_ms, 1),
                        "landmarks": landmarks,
                    })
            frame_idx += 1
        return frames_out
    except ImportError:
        pass

    # Fallback: MediaPipe (CPU)
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            t_ms = frame_idx / video_fps * 1000.0
            frame_bgr = cv2.resize(frame_bgr, (256, 256))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                frames_out.append({
                    "index": len(frames_out),
                    "t": round(t_ms / 1000.0, 3),
                    "ms": round(t_ms, 1),
                    "landmarks": landmarks,
                })
            frame_idx += 1
    finally:
        pose.close()
    return frames_out


@app.function(
    image=image_compare_full,
    timeout=10 * 60,  # 10 min for YouTube download + pose extraction
    gpu="T4",  # Optional: comment out for CPU-only (MediaPipe fallback)
)
@modal.web_endpoint(method="POST")
async def compare_full(request: Request):
    """
    POST with JSON: { "youtube_url": "...", "user": {...}, "exercise": "", "sample_fps": 8.0 }
    or { "reference": {...}, "user": {...}, "exercise": "" }.
    Extracts reference poses from YouTube if url given, then compares.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    youtube_url = (body.get("youtube_url") or "").strip() or None
    reference = body.get("reference")
    user = body.get("user")
    exercise = (body.get("exercise") or "").strip()
    muscle = (body.get("exercise_muscle") or "").strip()
    sample_fps = max(0.5, min(30.0, float(body.get("sample_fps", 8.0))))

    if not youtube_url and not reference:
        return JSONResponse({"error": "Provide youtube_url or reference"}, status_code=400)
    if not user:
        return JSONResponse({"error": "user is required"}, status_code=400)

    try:
        if reference is None and youtube_url:
            video_id = _extract_youtube_video_id(youtube_url)
            if not video_id:
                return JSONResponse({"error": "Invalid YouTube URL"}, status_code=400)
            stream_url = _get_youtube_stream_url(video_id)
            import cv2
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                return JSONResponse({"error": "Could not open video stream"}, status_code=502)
            try:
                frames_out = _run_pose_preprocess(cap, sample_fps)
                if not frames_out:
                    return JSONResponse({"error": "No pose frames extracted from video"}, status_code=400)
                reference = {"frames": frames_out}
            finally:
                cap.release()

        compare_poses = _get_compare_core()
        result = compare_poses(reference, user, exercise, muscle)
        return JSONResponse(result)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _open_video_from_payload(body: dict):
    """Open cv2.VideoCapture from youtube_url, video_url, or video_base64."""
    import base64
    import tempfile
    import cv2

    youtube_url = (body.get("youtube_url") or "").strip() or None
    video_url = (body.get("video_url") or "").strip() or None
    video_base64 = body.get("video_base64")

    if youtube_url:
        video_id = _extract_youtube_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        stream_url = _get_youtube_stream_url(video_id)
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError("Could not open YouTube stream")
        return cap

    if video_url:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise RuntimeError("Could not open video URL")
        return cap

    if video_base64:
        data = base64.b64decode(video_base64)
        if not data:
            raise ValueError("Empty video_base64")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                raise RuntimeError("Could not open uploaded video")
            return cap
        except Exception:
            import os
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise

    raise ValueError("Provide youtube_url, video_url, or video_base64")


@app.function(
    image=image_compare_full,
    timeout=10 * 60,
    gpu="T4",
)
@modal.web_endpoint(method="POST")
async def preprocess(request: Request):
    """
    POST with JSON: { "youtube_url": "..." } or { "video_url": "..." } or { "video_base64": "..." }.
    Returns { "frames": [...] } with pose landmarks. Use for form extraction from any video.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    sample_fps = max(0.5, min(30.0, float(body.get("sample_fps", 8.0))))

    try:
        cap = _open_video_from_payload(body)
        try:
            frames_out = _run_pose_preprocess(cap, sample_fps)
            if not frames_out:
                return JSONResponse({"error": "No pose frames extracted from video"}, status_code=400)
            return JSONResponse({"frames": frames_out})
        finally:
            cap.release()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
