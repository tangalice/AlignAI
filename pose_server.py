import asyncio
import io
import os
import re
import tempfile
import time
from difflib import get_close_matches

from dotenv import load_dotenv

load_dotenv()

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


# --- YMove Exercise API (https://ymove.app/exercise-api/docs) ---
YMOVE_API_KEY = os.environ.get("YMOVE_API_KEY", "").strip()
YMOVE_BASE = "https://exercise-api.ymove.app/api/v2"

# Map normalized query fragments to (search_keywords, muscle_group).
# YMove titles may use "Overhead Press", "Military Press" — we search by muscle + keywords.
EXERCISE_SEARCH_ALIASES: dict[str, tuple[list[str], str | None]] = {
    "shoulder press": (["press", "overhead", "military", "arnold", "dumbbell press"], "shoulders"),
    "overhead press": (["press", "overhead", "military", "shoulder"], "shoulders"),
    "military press": (["press", "military", "overhead", "shoulder"], "shoulders"),
    "arnold press": (["arnold", "press", "shoulder"], "shoulders"),
    "bench press": (["press", "bench", "chest", "incline"], "chest"),
    "chest press": (["press", "chest"], "chest"),
    "squat": (["squat", "goblet", "back", "front"], "quads"),
    "deadlift": (["deadlift", "romanian", "conventional"], "full_body"),
    "bicep curl": (["curl", "bicep", "hammer"], "biceps"),
    "tricep extension": (["extension", "tricep", "skull crusher", "pushdown"], "triceps"),
    "lat pull": (["pull", "pulldown", "lat"], "back"),
    "lat pulldown": (["pulldown", "pull", "lat"], "back"),
    "row": (["row", "barbell", "dumbbell", "cable"], "back"),
    "lunge": (["lunge", "forward", "reverse", "walking"], "quads"),
    "leg press": (["press", "leg"], "quads"),
    "leg curl": (["curl", "leg", "hamstring"], "hamstrings"),
    "hip thrust": (["thrust", "hip", "bridge", "glute"], "glutes"),
    "lateral raise": (["raise", "lateral", "side"], "shoulders"),
    "front raise": (["raise", "front"], "shoulders"),
    "press": (["press"], None),  # generic
}
YMOVE_MUSCLE_GROUPS = frozenset({
    "chest", "back", "shoulders", "biceps", "triceps", "forearms",
    "quads", "hamstrings", "glutes", "calves", "core", "full_body",
})

# Abbreviations users might type
SEARCH_ABBREVIATIONS: dict[str, str] = {
    "ohp": "overhead press",
    "bb": "barbell",
    "db": "dumbbell",
    "rdl": "romanian deadlift",
    "rdls": "romanian deadlift",
    "bw": "bodyweight",
    "goblet": "goblet squat",
}

# Cache: (query, limit) -> (result_dict, expiry_time)
_search_cache: dict[tuple[str, int], tuple[dict, float]] = {}
SEARCH_CACHE_TTL_SEC = 300  # 5 minutes

# Video URL cache: exercise_id -> (video_data_dict, expiry_time). URLs expire in 48h from YMove.
_video_cache: dict[str, tuple[dict, float]] = {}
VIDEO_CACHE_TTL_SEC = 3600  # 1 hour (YMove URLs last 48h)


def _normalize_query(q: str) -> str:
    """Expand abbreviations and optionally fix typos for alias matching."""
    q = (q or "").strip().lower()
    if not q:
        return q
    words = q.split()
    expanded = []
    for w in words:
        expanded.append(SEARCH_ABBREVIATIONS.get(w, w))
    return " ".join(expanded)


def _expand_search_terms(q: str) -> tuple[list[str], str | None]:
    """Return (search_terms, muscle_group). Uses aliases, typo tolerance, and infers muscle group."""
    raw = (q or "").strip().lower()
    q = _normalize_query(raw)
    if not q:
        return [], None
    terms = [q]
    muscle: str | None = None
    alias_keys = list(EXERCISE_SEARCH_ALIASES.keys())
    for key, (keywords, mg) in EXERCISE_SEARCH_ALIASES.items():
        if key in q or q in key:
            terms = [q] + [k for k in keywords if k not in q][:4]
            muscle = mg
            break
    # Typo tolerance: fuzzy match to alias keys if no exact match
    if not muscle and len(q) >= 4:
        matches = get_close_matches(q, alias_keys, n=1, cutoff=0.75)
        if matches:
            key = matches[0]
            keywords, muscle = EXERCISE_SEARCH_ALIASES[key]
            terms = [q, key] + [k for k in keywords if k not in (q, key)][:3]
    if not muscle:
        for word in q.split():
            if word in YMOVE_MUSCLE_GROUPS:
                muscle = word
                break
            w = word.rstrip("s")
            if w + "s" in YMOVE_MUSCLE_GROUPS:
                muscle = w + "s"
                break
    return terms[:5], muscle


def _get_suggestions(query: str) -> list[str]:
    """Return suggested search terms when no results, based on query similarity to aliases."""
    q = (query or "").strip().lower()
    if not q or len(q) < 3:
        return []
    alias_keys = list(EXERCISE_SEARCH_ALIASES.keys())
    matches = get_close_matches(q, alias_keys, n=3, cutoff=0.5)
    return [k for k in matches if k != q]


async def _ymove_request(path: str, params: dict | None = None) -> dict:
    """Call YMove API. Requires YMOVE_API_KEY env var."""
    if not YMOVE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="YMOVE_API_KEY not set. Get a key at https://ymove.app/exercise-api/signup",
        )
    url = f"{YMOVE_BASE}{path}"
    headers = {"X-API-Key": YMOVE_API_KEY}
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(url, params=params or {}, headers=headers)
    if r.status_code == 401:
        raise HTTPException(status_code=502, detail="Invalid YMove API key")
    if r.status_code == 429:
        data = r.json() if r.content else {}
        raise HTTPException(status_code=429, detail=data.get("error", "YMove rate limit exceeded"))
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"YMove API error: {r.status_code}")
    return r.json()


def _normalize_exercise(ex: dict) -> dict:
    """Normalize YMove exercise to frontend format."""
    return {
        "id": ex.get("id"),
        "name": ex.get("title", ""),
        "slug": ex.get("slug"),
        "muscle": ex.get("muscleGroup", ""),
        "level": ex.get("difficulty") or "intermediate",
        "equipment": ex.get("equipment", ""),
    }


def _rank_exercise(item: dict, query: str, muscle_hint: str | None, search_terms: list[str]) -> int:
    """Lower = better. Prioritize exact/partial title match, keyword match, and muscle match."""
    name = (item.get("name") or "").lower()
    q = (query or "").strip().lower()
    muscle = (item.get("muscle") or "").lower()
    score = 0
    if q and q in name:
        score -= 200
    if q:
        for w in q.split():
            if len(w) >= 2 and w in name:
                score -= 30
    for t in search_terms:
        if len(t) >= 3 and t in name:
            score -= 25
    if muscle_hint and muscle == muscle_hint:
        score -= 15
    return score


@app.get("/api/exercises/search")
async def exercises_search(q: str = "", limit: int = 20) -> JSONResponse:
    """Search exercises via YMove API. Cached, with muscle-group + parallel keyword searches."""
    base_params = {"hasVideo": "true", "pageSize": min(limit * 2, 50)}
    query = (q or "").strip()
    limit = min(limit, 50)
    cache_key = (query, limit)
    now = time.monotonic()

    # Check cache (skip for empty query to keep defaults fresh)
    if query:
        if cache_key in _search_cache:
            cached, expiry = _search_cache[cache_key]
            if now < expiry:
                return JSONResponse(content=cached)
        # Evict expired entries occasionally
        if len(_search_cache) > 200:
            _search_cache.clear()

    if not query:
        data = await _ymove_request("/exercises", base_params)
        exercises = data.get("data", [])
        out = [_normalize_exercise(ex) for ex in exercises[:limit]]
        return JSONResponse(content={"exercises": out})

    search_terms, muscle_hint = _expand_search_terms(query)

    async def fetch_batch(search_term: str | None, muscle_group: str | None) -> list[dict]:
        p = dict(base_params)
        if search_term:
            p["search"] = search_term
        if muscle_group:
            p["muscleGroup"] = muscle_group
        data = await _ymove_request("/exercises", p)
        return data.get("data", [])

    # Single combined request when we have muscle hint (faster than 2–3 parallel calls)
    if muscle_hint:
        data = await fetch_batch(search_terms[0] if search_terms else None, muscle_hint)
    else:
        data = await fetch_batch(search_terms[0] if search_terms else None, None)

    seen: set[str] = set()
    merged: list[dict] = []
    for ex in data or []:
        eid = ex.get("id")
        if eid and eid not in seen:
            seen.add(eid)
            merged.append(_normalize_exercise(ex))

    merged.sort(key=lambda x: _rank_exercise(x, query, muscle_hint, search_terms))
    out = merged[:limit]
    suggestions = _get_suggestions(query) if len(out) == 0 else []
    payload = {"exercises": out, "suggestions": suggestions}

    _search_cache[cache_key] = (payload, now + SEARCH_CACHE_TTL_SEC)
    return JSONResponse(content=payload)


@app.get("/api/exercises/video")
async def exercises_video(id: str = "") -> JSONResponse:
    """Get fresh video URL for exercise from YMove. Cached for 1h (URLs expire in 48h)."""
    ex_id = (id or "").strip()
    if not ex_id:
        raise HTTPException(status_code=400, detail="Missing exercise id")
    now = time.monotonic()
    if ex_id in _video_cache:
        cached, expiry = _video_cache[ex_id]
        if now < expiry:
            return JSONResponse(content=cached)
    data = await _ymove_request(f"/exercises/{ex_id}")
    ex = data.get("data") or {}
    video_url = ex.get("videoUrl")
    if not video_url:
        raise HTTPException(
            status_code=404,
            detail="No video available for this exercise. Try a different one or check your YMove plan.",
        )
    payload = {
        "video_url": video_url,
        "title": ex.get("title"),
        "id": ex.get("id"),
        "thumbnail_url": ex.get("thumbnailUrl"),
    }
    _video_cache[ex_id] = (payload, now + VIDEO_CACHE_TTL_SEC)
    return JSONResponse(content=payload)


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


class LLMCoachingRequest(BaseModel):
    score: float
    limbScores: Optional[dict[str, float]] = None
    feedback_message: Optional[str] = None
    worst_limb: Optional[str] = None
    exercise_name: Optional[str] = None
    exercise_muscle: Optional[str] = None


class CompareRequest(BaseModel):
    """Request for Modal compare: reference and user pose sequences."""
    reference: dict  # { "frames": [{ "landmarks": [[x,y,z],...] }], ... }
    user: dict  # Same format
    exercise: str = ""  # e.g. "squat", "pushup"


class CompareFullRequest(BaseModel):
    """Full pipeline: youtube_url OR reference + user. Runs preprocess+compare on Modal."""
    youtube_url: Optional[str] = None
    reference: Optional[dict] = None
    user: dict
    exercise: str = ""
    sample_fps: float = 8.0


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


class TTSRequest(BaseModel):
    text: str


# ElevenLabs TTS: default API key (override with ELEVENLABS_API_KEY env).


def _get_elevenlabs_api_key() -> str:
    return (os.environ.get("ELEVENLABS_API_KEY") or _ELEVENLABS_API_KEY_DEFAULT or "").strip()


@app.get("/api/tts/config")
async def tts_config() -> JSONResponse:
    """Return whether ElevenLabs TTS is available (so frontend can prefer it)."""
    return JSONResponse(content={"enabled": bool(_get_elevenlabs_api_key())})


@app.post("/api/tts")
async def tts_convert(body: TTSRequest) -> StreamingResponse:
    """Convert text to speech via ElevenLabs; returns MP3 audio."""
    api_key = _get_elevenlabs_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="ElevenLabs TTS not configured (set ELEVENLABS_API_KEY)")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"output_format": "mp3_22050_32"}
    payload = {"text": text, "model_id": "eleven_multilingual_v2"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                url,
                params=params,
                json=payload,
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            )
    except Exception as e:
        print(f"[tts] ElevenLabs request failed: {e}")
        raise HTTPException(status_code=502, detail=f"ElevenLabs request failed: {e}")
    if r.status_code != 200:
        print(f"[tts] ElevenLabs error status={r.status_code} body={r.text[:300]}")
        raise HTTPException(status_code=502, detail=f"ElevenLabs error: {r.status_code} {r.text[:200]}")
    print(f"[tts] ok text={text[:60]!r} len={len(r.content)}")
    return StreamingResponse(
        io.BytesIO(r.content),
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(r.content))},
    )


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()


@app.get("/api/coaching/llm/config")
async def llm_coaching_config() -> JSONResponse:
    """Check if LLM coaching is available (OPENAI_API_KEY set)."""
    return JSONResponse(content={"enabled": bool(OPENAI_API_KEY)})


@app.post("/api/coaching/llm")
async def llm_coaching(body: LLMCoachingRequest) -> JSONResponse:
    """Generate smarter coaching feedback via LLM. Requires OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"error": "OPENAI_API_KEY not set", "message": body.feedback_message},
        )

    score_pct = round(body.score * 100)
    worst_limbs = (
        sorted(
            (k for k, v in (body.limbScores or {}).items() if isinstance(v, (int, float)) and v < 0.85),
            key=lambda k: body.limbScores.get(k, 1),
        )[:3]
    )
    context_parts = [
        f"Exercise: {body.exercise_name or 'unknown'}",
        f" (target muscle: {body.exercise_muscle})" if body.exercise_muscle else "",
        f". Form score: {score_pct}%.",
    ]
    if body.worst_limb:
        limb_label = body.worst_limb.replace("_", " ")
        context_parts.append(f" Primary issue: {limb_label}.")
    if body.feedback_message:
        context_parts.append(f" Generic feedback: {body.feedback_message}")
    if worst_limbs:
        context_parts.append(f" Lowest-scoring limbs: {', '.join(w.replace('_', ' ') for w in worst_limbs)}.")

    context = "".join(context_parts).strip()

    system_prompt = """You are a concise fitness coach. The user is doing form comparison against an exercise demo. 
Give ONE short, actionable tip (under 12 words) to fix the form issue. Be specific to the limb/area mentioned. 
Use plain language. Examples: "Bring your elbow closer to your body", "Straighten your back", "Keep your knee over your ankle".
Reply with ONLY the tip, no preamble or quotes."""

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        r = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_tokens=50,
            temperature=0.6,
        )
        msg = (r.choices[0].message.content or "").strip()
        if not msg:
            msg = body.feedback_message or "Adjust your form to match the demo."
        return JSONResponse(content={"message": msg})
    except Exception as e:
        err_msg = str(e)
        print(f"[llm-coaching] error: {err_msg}")
        fallback = body.feedback_message or "Adjust your form to match the demo."
        return JSONResponse(
            status_code=502,
            content={"error": err_msg, "message": fallback},
        )


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


@app.post("/api/pose/compare")
async def pose_compare(body: CompareRequest) -> JSONResponse:
    """Compare reference pose sequence with user pose sequence (runs locally)."""
    from compare.compare_core import compare_poses
    result = compare_poses(
        body.reference,
        body.user,
        (body.exercise or "").strip(),
    )
    return JSONResponse(content=result)


@app.post("/api/pose/compare-full")
async def pose_compare_full(body: CompareFullRequest) -> JSONResponse:
    """Full pipeline: preprocess YouTube (if url) + compare. Runs locally."""
    from compare.compare_core import compare_poses

    if not body.youtube_url and not body.reference:
        raise HTTPException(status_code=400, detail="Provide youtube_url or reference")

    reference = body.reference
    if reference is None and body.youtube_url:
        video_id = _extract_youtube_video_id(body.youtube_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        try:
            stream_url = _get_youtube_stream_url(video_id)
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise HTTPException(status_code=502, detail="Could not open video stream")
            sample_fps = max(0.5, min(30.0, body.sample_fps))
            frames_out = _run_pose_preprocess(cap, sample_fps)
            cap.release()
            if not frames_out:
                raise HTTPException(status_code=400, detail="No pose frames extracted from video")
            reference = {"frames": frames_out}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preprocess failed: {e}") from e

    result = compare_poses(
        reference,
        body.user,
        (body.exercise or "").strip(),
    )
    return JSONResponse(content=result)


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

