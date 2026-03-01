import io
import os
from typing import Optional

import cv2
import httpx
import mediapipe as mp
import numpy as np
import yt_dlp
import json
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, Response


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
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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


@app.get("/api/youtube/{video_id}")
async def youtube_stream(video_id: str) -> RedirectResponse:
    if not video_id or len(video_id) > 20:
        raise HTTPException(status_code=400, detail="Invalid video ID")
    try:
        stream_url = _get_youtube_stream_url(video_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not get stream: {e}") from e
    # Let the browser fetch the media directly (supports Range/seek).
    return RedirectResponse(url=stream_url, status_code=307)


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


_gen = (os.environ.get("MODAL_VIDEO_GENERATE_ENDPOINT") or "").strip().strip('"\'')
_dl  = (os.environ.get("MODAL_VIDEO_DOWNLOAD_BASE") or "").strip().strip('"\'')

def _clean_url(u: str) -> str:
    u = (u or "").strip().strip('"\'')
    u = u.replace("\u201c", '"').replace("\u201d", '"')  # protect from smart quotes
    u = u.strip('"\'')
    if u and not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u.rstrip("/")

MODAL_VIDEO_GENERATE_ENDPOINT = _clean_url(_gen)
MODAL_VIDEO_DOWNLOAD_BASE = _clean_url(_dl)

# Avoid Modal/API ASCII codec errors from smart quotes etc.
_unicode_to_ascii = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
})


def _ascii_safe_prompt(s: str) -> str:
    return s.translate(_unicode_to_ascii).encode("ascii", "replace").decode("ascii")


@app.post("/api/ai-video/generate-sse")
async def ai_video_generate_sse(body: AIVideoRequest):
    if not MODAL_VIDEO_GENERATE_ENDPOINT:
        raise HTTPException(status_code=503, detail="Set MODAL_VIDEO_GENERATE_ENDPOINT")

    generate_url = f"{MODAL_VIDEO_GENERATE_ENDPOINT}/generate"

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", generate_url, json={"prompt": body.prompt}) as r:
                if r.status_code != 200:
                    text = await r.aread()
                    # emit a single SSE error so the client can show it
                    err = text.decode("utf-8", "replace")
                    yield f"data: {json.dumps({'type':'error','message':err})}\n\n"
                    return

                async for chunk in r.aiter_bytes():
                    # passthrough SSE bytes exactly
                    yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/ai-video/download/{gen_id}")
async def ai_video_download(gen_id: str):
    if not MODAL_VIDEO_DOWNLOAD_BASE:
        raise HTTPException(status_code=503, detail="Set MODAL_VIDEO_ENDPOINT")
    download_url = f"{MODAL_VIDEO_DOWNLOAD_BASE}/download/{gen_id}"

    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.get(download_url)

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=r.text or f"Modal returned {r.status_code}",
        )

    content = r.content
    if not content:
        raise HTTPException(status_code=502, detail="Empty response from Modal")

    return StreamingResponse(
        io.BytesIO(r.content),
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{gen_id}.mp4"'},
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

    uvicorn.run("pose_server:app", host="0.0.0.0", port=8000, reload=True)

