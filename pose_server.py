import io
import os
import re
import tempfile

# Fix SSL certs on macOS (python.org installs)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass
from pathlib import Path
from typing import Optional

import cv2
import httpx
import mediapipe as mp
import numpy as np
import yt_dlp
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel


class AIVideoRequest(BaseModel):
    prompt: str


app = FastAPI(title="FormAI Pose Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,   # FAST
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
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
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,
        # Prefer a single, muxed stream that an HTML5 <video> can play.
        # (Avoid HLS/DASH manifests and video-only / audio-only streams.)
        "format": "best[ext=mp4][acodec!=none][vcodec!=none]/best[acodec!=none][vcodec!=none]",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise HTTPException(status_code=404, detail="Video not found")
    # Prefer direct url (set when format selector picks one)
    if info.get("url"):
        return info["url"]
    # Otherwise pick best format with a URL and video
    formats = info.get("formats") or []
    for f in reversed(formats):
        if f.get("url") and (f.get("vcodec") or "none") != "none":
            return f["url"]
    raise HTTPException(status_code=404, detail="No stream URL found")


class PreprocessRequest(BaseModel):
    url: str
    sample_fps: float = 8.0


class PrimaryIssueModel(BaseModel):
    joint: str
    message: str


class ComparePoseRequest(BaseModel):
    video_t: Optional[float] = None
    score: float
    limbScores: Optional[dict[str, float]] = None
    feedback: Optional[PrimaryIssueModel] = None


def _download_youtube_to_temp(url: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / "video.%(ext)s"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_path),
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best[height<=720]",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    # yt-dlp may have produced video.mp4 or video.mkv etc.
    for p in tmp_dir.iterdir():
        if p.suffix in (".mp4", ".mkv", ".webm", ".avi"):
            return p
    raise FileNotFoundError("No video file downloaded")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _run_pose_preprocess(cap: cv2.VideoCapture, sample_fps: float) -> list:
    """Run pose detection on sampled frames. Uses YOLOv8-pose (GPU) if available, else MediaPipe."""
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    frames_out = []
    frame_idx = 0

    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n-pose.pt")  # Uses GPU if available
        use_half = _cuda_available()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            t_ms = frame_idx / video_fps * 1000.0
            frame_small = cv2.resize(frame_bgr, (256, 256))
            results = model(
                frame_small,
                verbose=False,
                imgsz=256,
                half=use_half,
            )
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
    frame_idx = 0
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
        results = pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            frames_out.append({
                "index": len(frames_out),
                "t": round(t_ms / 1000.0, 3),
                "ms": round(t_ms, 1),
                "landmarks": landmarks,
            })
        frame_idx += 1
    return frames_out


@app.post("/api/compare-pose")
async def compare_pose(body: ComparePoseRequest) -> JSONResponse:
    """Log pose comparison score and coaching feedback from frontend."""
    parts = [f"video_t={body.video_t}", f"score={body.score:.3f}"]
    if body.feedback:
        f = body.feedback
        parts.append(f'feedback: "{f.message}"')
    elif body.limbScores:
        parts.append("limbs=" + ",".join(f"{k}={v:.2f}" for k, v in sorted(body.limbScores.items())[:4]))
    print(f"[compare-pose] {' '.join(parts)}")
    return JSONResponse(content={"ok": True, "score": body.score})


@app.post("/api/preprocess")
async def preprocess_youtube_for_supermemory(body: PreprocessRequest) -> JSONResponse:
    """Stream YouTube video, run pose detection (GPU via YOLOv8 when available) on sampled frames."""
    url = (body.url or "").strip()
    video_id = _extract_youtube_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="A YouTube video URL is required")
    sample_fps = max(0.5, min(30.0, body.sample_fps))

    try:
        stream_url = _get_youtube_stream_url(video_id)
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise HTTPException(status_code=502, detail="Could not open video stream")

        frames_out = _run_pose_preprocess(cap, sample_fps)
        cap.release()
        return JSONResponse(content={"frames": frames_out})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocess failed: {e}") from e


@app.get("/api/youtube/{video_id}")
async def youtube_stream(video_id: str):
    """Serve a YouTube video as a same-origin MP4 file so WebGL/MediaPipe can read pixels safely."""
    if not video_id or len(video_id) > 20:
        raise HTTPException(status_code=400, detail="Invalid video ID")
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp_path: Path | None = None
    try:
        tmp_path = _download_youtube_to_temp(url)
        data = tmp_path.read_bytes()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not download video: {e}") from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
            try:
                tmp_path.parent.rmdir()
            except OSError:
                pass
    return StreamingResponse(io.BytesIO(data), media_type="video/mp4")


def _read_image_from_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image data")
    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


def _annotate_pose_on_frame(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)
    if not results.pose_landmarks:
        return None
    annotated = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
    )
    return annotated


MODAL_VIDEO_ENDPOINT = os.environ.get("MODAL_VIDEO_ENDPOINT", "").rstrip("/")


@app.post("/api/ai-video/generate")
async def ai_video_generate(body: AIVideoRequest) -> StreamingResponse:
    """Generate a workout/exercise video from a text description via Modal.com."""
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing or empty prompt")

    if not MODAL_VIDEO_ENDPOINT:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI video generation is not configured. Set MODAL_VIDEO_ENDPOINT to your Modal "
                "deployed endpoint (e.g. https://your-app--generate.modal.run)."
            ),
        )

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(MODAL_VIDEO_ENDPOINT, json={"prompt": prompt})
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Video generation timed out. Try again or use a shorter description.",
        ) from None
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Modal request failed: {e}") from e

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=r.text or f"Modal returned {r.status_code}",
        )

    content = r.content
    if not content:
        raise HTTPException(status_code=502, detail="Empty response from Modal")

    return StreamingResponse(
        io.BytesIO(content),
        media_type="video/mp4",
        headers={"Content-Disposition": "inline; filename=ai-workout.mp4"},
    )


@app.post("/pose/frame")
async def pose_on_frame(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        frame = _read_image_from_upload(file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {exc}") from exc

    annotated = _annotate_pose_on_frame(frame)
    if annotated is None:
        return JSONResponse({"message": "No pose detected"}, status_code=200)

    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pose_server:app", host="0.0.0.0", port=8001, reload=True)

