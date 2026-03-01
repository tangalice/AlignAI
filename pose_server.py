import asyncio
import io
import json
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
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import httpx
import mediapipe as mp
import numpy as np
import yt_dlp
import json
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, Response

class AIVideoRequest(BaseModel):
    prompt: str


app = FastAPI(title="FormAI Pose Server", version="0.1.0")

# CORS: use CORS_ORIGINS env for production (comma-separated), "*" for dev
_cors_origins = os.environ.get("CORS_ORIGINS", "*").strip()
CORS_ORIGINS = [o.strip() for o in _cors_origins.split(",") if o.strip()] if _cors_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limit for expensive endpoints (per IP)
_rate_limit: dict[str, tuple[int, float]] = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30  # requests per window


def _check_rate_limit(client_host: str) -> bool:
    now = time.monotonic()
    if client_host in _rate_limit:
        count, window_start = _rate_limit[client_host]
        if now - window_start > RATE_LIMIT_WINDOW:
            _rate_limit[client_host] = (1, now)
            return True
        if count >= RATE_LIMIT_MAX:
            return False
        _rate_limit[client_host] = (count + 1, window_start)
        return True
    _rate_limit[client_host] = (1, now)
    return True

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
    "curl": "bicep curl",
}

# Cache: (query, limit) -> (result_dict, expiry_time)
_search_cache: dict[tuple[str, int], tuple[dict, float]] = {}
SEARCH_CACHE_TTL_SEC = 300  # 5 minutes
_MAX_SEARCH_CACHE = 300

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
    if not muscle and len(q) >= 3:
        matches = get_close_matches(q, alias_keys, n=1, cutoff=0.6)
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


_ymove_client: httpx.AsyncClient | None = None


async def _get_ymove_client() -> httpx.AsyncClient:
    """Reuse HTTP client for connection pooling to YMove."""
    global _ymove_client
    if _ymove_client is None:
        _ymove_client = httpx.AsyncClient(timeout=6.0)
    return _ymove_client


async def _ymove_request(path: str, params: dict | None = None) -> dict:
    """Call YMove API. Requires YMOVE_API_KEY env var."""
    if not YMOVE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="YMOVE_API_KEY not set. Get a key at https://ymove.app/exercise-api/signup",
        )
    url = f"{YMOVE_BASE}{path}"
    headers = {"X-API-Key": YMOVE_API_KEY}
    client = await _get_ymove_client()
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


def _prefix_cache_lookup(query: str, now: float) -> dict | None:
    """If we have a cached search for a longer query that starts with this one, return it for instant results."""
    q = query.lower()
    if len(q) < 2:
        return None
    best: tuple[str, dict] | None = None
    for (cached_q, _), (payload, expiry) in _search_cache.items():
        if now >= expiry:
            continue
        c = cached_q.lower()
        if c.startswith(q) and len(c) > len(q):
            if best is None or len(c) < len(best[0]):
                best = (c, payload)
    return best[1] if best else None


@app.get("/api/exercises/search")
async def exercises_search(
    q: str = "", limit: int = 20, quick: int = 0
) -> JSONResponse:
    """Search exercises via YMove API. Cached. Use ?quick=1 for smaller initial response."""
    base_params = {"hasVideo": "true", "pageSize": min(limit * 2, 50)}
    query = (q or "").strip()
    limit = min(limit, 50)
    if quick:
        limit = min(limit, 8)
        base_params["pageSize"] = min(16, base_params["pageSize"])
    cache_key = (query, limit)
    now = time.monotonic()

    # Exact cache hit
    if query and cache_key in _search_cache:
        cached, expiry = _search_cache[cache_key]
        if now < expiry:
            return JSONResponse(content=cached)

    # Prefix cache: instant results when user is typing (e.g. "cur" → use "curl" cache)
    if query and len(query) >= 2:
        prefix_hit = _prefix_cache_lookup(query, now)
        if prefix_hit:
            return JSONResponse(content=prefix_hit)

    if len(_search_cache) > _MAX_SEARCH_CACHE:
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
    temporal_trend: Optional[str] = None  # "improving" | "worsening" | null


class SupermemoryAddExerciseRequest(BaseModel):
    exercise_name: str
    exercise_muscle: Optional[str] = None
    video_url: Optional[str] = None
    exercise_id: Optional[str] = None


class WorkoutSample(BaseModel):
    video_t: Optional[float] = None
    score: float
    limbScores: Optional[dict[str, float]] = None
    feedback: Optional[PrimaryIssueModel] = None


class WorkoutSummaryRequest(BaseModel):
    exercise_name: str = ""
    exercise_muscle: str = ""
    duration_sec: Optional[float] = None
    reps: Optional[int] = None
    samples: list[WorkoutSample] = []


class CoachChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class CoachPTReportRequest(BaseModel):
    reason: str  # "progress_declining" | "low_form_score" | "user_requested"
    user_id: str = "default"


class CompareRequest(BaseModel):
    """Request for Modal compare: reference and user pose sequences."""
    reference: dict  # { "frames": [{ "landmarks": [[x,y,z],...] }], ... }
    user: dict  # Same format
    exercise: str = ""  # e.g. "squat", "pushup", "Bicep Curl"
    exercise_muscle: str = ""  # Optional YMove muscle group for better weights


class CompareFullRequest(BaseModel):
    """Full pipeline: youtube_url OR reference + user. Runs preprocess+compare on Modal."""
    youtube_url: Optional[str] = None
    reference: Optional[dict] = None
    user: dict
    exercise: str = ""
    exercise_muscle: str = ""
    sample_fps: float = 8.0


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


# ElevenLabs TTS: set ELEVENLABS_API_KEY in env to enable.


def _get_elevenlabs_api_key() -> str:
    return (os.environ.get("ELEVENLABS_API_KEY") or "").strip()


@app.get("/api/tts/config")
async def tts_config() -> JSONResponse:
    """Return whether ElevenLabs TTS is available (so frontend can prefer it)."""
    return JSONResponse(content={"enabled": bool(_get_elevenlabs_api_key())})


@app.post("/api/tts")
async def tts_convert(request: Request, body: TTSRequest) -> StreamingResponse:
    """Convert text to speech via ElevenLabs; returns MP3 audio."""
    client = getattr(request, "client", None)
    if client and not _check_rate_limit(client.host or "unknown"):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
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
PPLX_API_KEY = os.environ.get("PPLX_API_KEY", "").strip()
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "").strip()
SUPERMEMORY_CONTAINER = "formai_exercises"
SUPERMEMORY_USER_CONTAINER = "formai_user"


async def _supermemory_add_user(content: str, custom_id: str) -> bool:
    """Add user-specific content to Supermemory for personalized coaching."""
    if not SUPERMEMORY_API_KEY:
        return False
    try:
        from supermemory import AsyncSupermemory

        client = AsyncSupermemory(api_key=SUPERMEMORY_API_KEY)
        await client.add(
            content=content,
            container_tags=[SUPERMEMORY_USER_CONTAINER],
            custom_id=custom_id,
        )
        return True
    except Exception as e:
        print(f"[supermemory] add-user error: {e}")
        return False


async def _supermemory_search(
    q: str, limit: int = 5, include_user: bool = True
) -> list[str]:
    """Search Supermemory for form cues and optionally user context. Returns relevant text chunks."""
    if not SUPERMEMORY_API_KEY:
        return []
    try:
        from supermemory import AsyncSupermemory

        client = AsyncSupermemory(api_key=SUPERMEMORY_API_KEY)
        chunks = []
        per_container = max(2, limit // 2)
        results_ex = await client.search.documents(
            q=q,
            container_tags=[SUPERMEMORY_CONTAINER],
            limit=per_container,
        )
        for r in results_ex.results:
            text = getattr(r, "chunk", None) or getattr(r, "memory", None) or getattr(r, "content", None)
            if text and isinstance(text, str):
                chunks.append(text.strip())
        if include_user:
            results_user = await client.search.documents(
                q=q,
                container_tags=[SUPERMEMORY_USER_CONTAINER],
                limit=per_container,
            )
            for r in results_user.results:
                text = getattr(r, "chunk", None) or getattr(r, "memory", None) or getattr(r, "content", None)
                if text and isinstance(text, str):
                    chunks.append(text.strip())
        return chunks[:limit]
    except Exception as e:
        print(f"[supermemory] search error: {e}")
        return []


async def _supermemory_add(content: str, custom_id: str) -> bool:
    """Add content to Supermemory. Returns True on success."""
    if not SUPERMEMORY_API_KEY:
        return False
    try:
        from supermemory import AsyncSupermemory

        client = AsyncSupermemory(api_key=SUPERMEMORY_API_KEY)
        await client.add(
            content=content,
            container_tags=[SUPERMEMORY_CONTAINER],
            custom_id=custom_id,
        )
        return True
    except Exception as e:
        print(f"[supermemory] add error: {e}")
        return False


@app.post("/api/supermemory/add-exercise")
async def supermemory_add_exercise(body: SupermemoryAddExerciseRequest) -> JSONResponse:
    """Add exercise context to Supermemory for RAG. Called when user selects an exercise."""
    if not SUPERMEMORY_API_KEY:
        return JSONResponse(content={"ok": False, "error": "SUPERMEMORY_API_KEY not set"})
    name = (body.exercise_name or "").strip()
    if not name:
        return JSONResponse(content={"ok": False, "error": "exercise_name required"})
    muscle = (body.exercise_muscle or "").strip()
    custom_id = f"exercise_{body.exercise_id or name.lower().replace(' ', '_')}"
    added = []
    if body.video_url:
        ok = await _supermemory_add(body.video_url, f"{custom_id}_video")
        if ok:
            added.append("video")
    desc = f"Exercise: {name}"
    if muscle:
        desc += f" (muscle: {muscle})"
    desc += ". Form cues: match the demo's arm and leg positions, keep core braced, control the movement."
    ok = await _supermemory_add(desc, f"{custom_id}_desc")
    if ok:
        added.append("desc")
    return JSONResponse(content={"ok": len(added) > 0, "added": added})


@app.get("/api/coaching/llm/config")
async def llm_coaching_config() -> JSONResponse:
    """Check if LLM coaching, Supermemory, and ElevenLabs TTS are available."""
    return JSONResponse(content={
        "enabled": bool(OPENAI_API_KEY),
        "supermemory": bool(SUPERMEMORY_API_KEY),
        "tts": bool(_get_elevenlabs_api_key()),
    })


@app.post("/api/coaching/llm")
async def llm_coaching(request: Request, body: LLMCoachingRequest) -> JSONResponse:
    """Generate smarter coaching feedback via LLM. Requires OPENAI_API_KEY."""
    client = getattr(request, "client", None)
    if client and not _check_rate_limit(client.host or "unknown"):
        raise HTTPException(status_code=429, detail={"error": "Too many requests", "message": body.feedback_message})
    if not OPENAI_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"error": "OPENAI_API_KEY not set", "message": body.feedback_message},
        )

    score_pct = round(body.score * 100)
    worst_limbs = (
        sorted(
            (k for k, v in (body.limbScores or {}).items() if isinstance(v, (int, float)) and v < 0.75),
            key=lambda k: body.limbScores.get(k, 1),
        )[:3]
    )
    worst_limb_score = 1.0
    if body.worst_limb and body.limbScores:
        worst_limb_score = body.limbScores.get(body.worst_limb, 1.0)

    context_parts = [
        f"Exercise: {body.exercise_name or 'unknown'}",
        f" (target muscle: {body.exercise_muscle})" if body.exercise_muscle else "",
        f". Form score: {score_pct}%. Worst limb score: {round(worst_limb_score * 100)}%.",
    ]
    if body.worst_limb:
        limb_label = body.worst_limb.replace("_", " ")
        context_parts.append(f" PRIMARY ISSUE TO ADDRESS: {limb_label}.")
    if body.feedback_message:
        context_parts.append(f" Generic cue from pose comparison: {body.feedback_message}")
    if worst_limbs:
        context_parts.append(f" Other weak areas: {', '.join(w.replace('_', ' ') for w in worst_limbs)}.")
    if body.temporal_trend:
        context_parts.append(f" Trend: form is {body.temporal_trend} over the last few seconds.")

    # Supermemory RAG: fetch exercise form cues + user's past workout context
    rag_context = ""
    if body.exercise_name or body.worst_limb:
        search_q = f"{body.exercise_name or 'exercise'} form cues {body.worst_limb or ''}".strip()
        chunks = await _supermemory_search(search_q, limit=5, include_user=True)
        if chunks:
            rag_context = "\n\nContext (form guides + user's past sessions):\n" + "\n".join(f"- {c}" for c in chunks)

    context = "".join(context_parts).strip() + rag_context

    system_prompt = """You are a concise fitness coach. The user is mirroring an exercise demo. Give feedback ONLY for the PRIMARY issue—the worst limb or body part mentioned in the context.

RULES:
- Base your tip on the specific limb/body part flagged (worst_limb, lowest-scoring limbs). Do not invent new issues.
- Use the generic feedback as a starting point but make it more actionable. If it says "raise left arm", say "Raise your left arm to shoulder height" or similar.
- Be concrete: name the body part and what to do. Examples: "Tuck your elbows in", "Keep your knee over your ankle", "Straighten your back—avoid rounding".
- NEVER say vague things like "adjust slightly" or "move a little". Be specific.
- Keep the tip under 12 words. Reply with ONLY the tip, no preamble or quotes."""

    fallback = body.feedback_message or "Adjust your form to match the demo."
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
            temperature=0.3,
        )
        if r.choices and len(r.choices) > 0:
            msg = (r.choices[0].message.content or "").strip()
        else:
            msg = ""
        if not msg:
            msg = fallback
        return JSONResponse(content={"message": msg})
    except Exception as e:
        err_msg = str(e)
        print(f"[llm-coaching] error: {err_msg}")
        return JSONResponse(content={"message": fallback})


@app.post("/api/workout/summary")
async def workout_summary(request: Request, body: WorkoutSummaryRequest) -> JSONResponse:
    """Generate an AI summary of a workout session. Uses Perplexity API if set, else falls back to OpenAI."""
    client = getattr(request, "client", None)
    if client and not _check_rate_limit(client.host or "unknown"):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    if not PPLX_API_KEY and not OPENAI_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"error": "PPLX_API_KEY or OPENAI_API_KEY required for summaries. Add one to .env and restart.", "summary": None},
        )

    samples = body.samples or []
    if not samples:
        return JSONResponse(content={"summary": "No form data was recorded during this workout. Start the exercise and mirror the demo to get a summary next time."})

    # Aggregate stats
    scores = [s.score for s in samples if isinstance(s.score, (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0

    limb_counts: dict[str, int] = {}
    feedback_messages: list[str] = []
    for s in samples:
        if s.feedback and s.feedback.message:
            feedback_messages.append(s.feedback.message)
        if s.limbScores:
            for limb, sc in s.limbScores.items():
                if isinstance(sc, (int, float)) and sc < 0.85:
                    limb_counts[limb] = limb_counts.get(limb, 0) + 1

    worst_limbs = sorted(limb_counts.keys(), key=lambda k: limb_counts[k], reverse=True)[:5]
    unique_feedback = list(dict.fromkeys(feedback_messages))[:10]

    exercise_info = (body.exercise_name or "this exercise").strip()
    if body.exercise_muscle:
        exercise_info += f" (target: {body.exercise_muscle})"

    context = f"""Workout summary data:
- Exercise: {exercise_info}
- Duration: {body.duration_sec or 0:.0f} seconds
- Form comparison samples: {len(samples)}
- Average form score: {avg_score * 100:.0f}%
- Best moment: {max_score * 100:.0f}%
- Toughest moment: {min_score * 100:.0f}%
- Most frequently flagged areas: {', '.join(w.replace('_', ' ') for w in worst_limbs) if worst_limbs else 'none'}
- Feedback moments: {'; '.join(unique_feedback[:5]) if unique_feedback else 'none'}"""

    system_prompt = """You are a friendly, encouraging fitness coach. The user just finished a form-check workout session.
Generate a concise summary (2-3 short paragraphs) that:
1. Opens with a brief overall assessment — what went well, energy level.
2. Highlights 1-2 things they did well (be specific, reference the data).
3. Suggests 1-2 actionable improvements for next time (body part or form cue).
4. Ends on an encouraging note.

Be concise, warm, and specific. Use plain language. No bullet points or headers.
Do not include any citations, reference numbers, or [1][2][3] style references.
Use plain text only — no markdown (no ** for bold)."""

    try:
        from openai import AsyncOpenAI

        if PPLX_API_KEY:
            client = AsyncOpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")
            model = "sonar"
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4o-mini"
        r = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_tokens=400,
            temperature=0.7,
        )
        summary = ""
        if r.choices and len(r.choices) > 0:
            summary = (r.choices[0].message.content or "").strip()
        # Remove citation refs [1][2][3] and markdown bold asterisks so UI stays clean
        summary = re.sub(r"\s*\[\d+\]\s*", " ", summary).strip()
        summary = re.sub(r"\*\*([^*]+)\*\*", r"\1", summary)
        if not summary:
            summary = "Your workout session was recorded. Keep practicing to improve your form!"

        if SUPERMEMORY_API_KEY:
            memory_content = f"""Workout session: {exercise_info}
Date: {datetime.utcnow().strftime('%Y-%m-%d')}
Summary: {summary}
Areas to work on: {', '.join(w.replace('_', ' ') for w in worst_limbs) if worst_limbs else 'none'}
Duration: {body.duration_sec or 0:.0f}s"""
            if body.reps:
                memory_content += f"\nReps completed: {body.reps}"
            if unique_feedback:
                memory_content += f"\nFeedback received during workout: {'; '.join(unique_feedback[:5])}"
            custom_id = f"workout_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{(body.exercise_name or 'unknown').lower().replace(' ', '_')[:30]}"
            ok = await _supermemory_add_user(memory_content, custom_id)
            if ok:
                print(f"[supermemory] saved workout to user context: {custom_id}")

        try:
            from progress_store import add_progress

            add_progress(
                "default",
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                exercise=body.exercise_name or "unknown",
                muscle=body.exercise_muscle or "",
                avg_score=avg_score,
                min_score=min_score,
                max_score=max_score,
                reps=body.reps or 0,
                duration_sec=body.duration_sec or 0,
                worst_limbs=worst_limbs,
            )
        except Exception as e:
            print(f"[progress] save error: {e}")

        return JSONResponse(content={"summary": summary})
    except Exception as e:
        print(f"[workout-summary] error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": str(e), "summary": "Could not generate summary. Your workout was still recorded — try again later."},
        )


@app.get("/api/progress")
async def get_progress_api(user_id: str = "default") -> JSONResponse:
    """Get progress data for charts and trend analysis."""
    try:
        from progress_store import get_progress_summary
        data = get_progress_summary(user_id)
        return JSONResponse(content=data)
    except Exception as e:
        print(f"[progress] get error: {e}")
        return JSONResponse(content={"entries": [], "trend": "unknown", "alert": None})


@app.get("/api/coach/status")
async def coach_status() -> JSONResponse:
    """Check if the AI coach is available (PPLX or OPENAI API key set)."""
    available = bool(PPLX_API_KEY or OPENAI_API_KEY)
    return JSONResponse(content={
        "available": available,
        "reason": None if available else "Set PPLX_API_KEY or OPENAI_API_KEY in .env to enable the coach.",
    })


@app.post("/api/coach/chat")
async def coach_chat(request: Request, body: CoachChatRequest) -> JSONResponse:
    """AI coach chatbot with user context from Supermemory. Uses Perplexity if PPLX_API_KEY set, else OpenAI."""
    if not PPLX_API_KEY and not OPENAI_API_KEY:
        return JSONResponse(status_code=503, content={"error": "No API key", "message": "Coach is unavailable. Set PPLX_API_KEY or OPENAI_API_KEY in the server .env to enable the AI coach."})
    client = getattr(request, "client", None)
    if client and not _check_rate_limit(client.host or "unknown"):
        raise HTTPException(status_code=429, detail="Too many requests.")
    msg = (body.message or "").strip()
    if not msg:
        return JSONResponse(content={"message": "What would you like to know about your workouts?"})

    async def _do_chat():
        search_q = f"{msg} user workout history progress form"
        try:
            chunks = await asyncio.wait_for(_supermemory_search(search_q, limit=6, include_user=True), timeout=8.0)
        except asyncio.TimeoutError:
            print("[coach-chat] supermemory search timed out, using no context")
            chunks = []
        rag = "\n".join(f"- {c}" for c in chunks) if chunks else "No prior context yet."

        system = """You are a friendly, knowledgeable fitness coach who knows this user from their FormAI workout history. 
You have access to their past sessions, form feedback, and progress. Be conversational, supportive, and specific.
Reference their data when relevant. Keep responses concise (2-4 sentences unless they ask for more).
If they mention pain, injury, or concerning symptoms, gently suggest they consult a physical therapist.
Do not include any citations or [1][2][3] style references. Use plain text only."""

        user_content = f"Context about this user:\n{rag}\n\nUser says: {msg}"

        from openai import AsyncOpenAI
        if PPLX_API_KEY:
            oai = AsyncOpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")
            model = "sonar"
        else:
            oai = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4o-mini"
        r = await oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        reply = ""
        if r.choices and len(r.choices) > 0:
            reply = (r.choices[0].message.content or "").strip()
        # Strip Perplexity citation refs if present
        reply = re.sub(r"\s*\[\d+\]\s*", " ", reply).strip()
        reply = re.sub(r"\*\*([^*]+)\*\*", r"\1", reply)
        if not reply:
            reply = "I'm here to help with your workouts. Could you try asking again?"
        return reply

    try:
        reply = await asyncio.wait_for(_do_chat(), timeout=25.0)
        return JSONResponse(content={"message": reply})
    except asyncio.TimeoutError:
        print("[coach-chat] request timed out")
        return JSONResponse(status_code=504, content={"message": "I'm taking too long. Please try a shorter question."})
    except Exception as e:
        print(f"[coach-chat] error: {e}")
        msg = "Sorry, I couldn't process that. Try again."
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "insufficient_quota" in err_str:
            msg = "OpenAI API quota exceeded. Check your plan and billing at platform.openai.com, or try again later."
        elif "rate" in err_str.lower() and "limit" in err_str.lower():
            msg = "Rate limit reached. Please wait a moment and try again."
        return JSONResponse(status_code=502, content={"message": msg})


@app.post("/api/coach/pt-report")
async def coach_pt_report(request: Request, body: CoachPTReportRequest) -> JSONResponse:
    """Generate formal report for physical therapist when injury/decline detected."""
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=503, content={"error": "OPENAI_API_KEY not set", "report": ""})
    client = getattr(request, "client", None)
    if client and not _check_rate_limit(client.host or "unknown"):
        raise HTTPException(status_code=429, detail="Too many requests.")

    try:
        from progress_store import get_progress_summary, get_progress

        summary = get_progress_summary(body.user_id)
        entries = get_progress(body.user_id, limit=30)

        search_q = "workout summary form feedback areas to work on"
        chunks = await _supermemory_search(search_q, limit=8, include_user=True)
        rag = "\n---\n".join(chunks) if chunks else "No additional context."

        report_context = f"""Generate a formal Physical Therapist Referral Report.

Reason for referral: {body.reason}
User progress data (last {len(entries)} sessions):
{json.dumps(summary, indent=2)}

Recent workout context from user's history:
{rag}

Create a professional 1-2 page report with:
1. Patient Summary (anonymous - no PII)
2. Exercise/Form Data Overview
3. Trend Analysis
4. Areas of Concern
5. Recommended Next Steps for PT Evaluation

Use formal medical/fitness terminology. Be concise but thorough. Format for a healthcare professional."""

        from openai import AsyncOpenAI
        oai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        r = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": report_context}],
            max_tokens=800,
            temperature=0.3,
        )
        report = ""
        if r.choices and len(r.choices) > 0:
            report = (r.choices[0].message.content or "").strip()
        if not report:
            report = "Unable to generate report. Your progress data is available in the app."
        return JSONResponse(content={"report": report, "alert": body.reason})
    except Exception as e:
        print(f"[coach-pt-report] error: {e}")
        return JSONResponse(
            content={
                "report": "Sorry, the report couldn't be generated. Try again or check your connection.",
                "alert": body.reason,
            },
        )
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
    """Extract pose frames from video URL. Proxies to Modal (GPU) when MODAL_PREPROCESS_ENDPOINT is set."""
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    sample_fps = max(0.5, min(30.0, body.sample_fps))

    if MODAL_PREPROCESS_ENDPOINT:
        try:
            payload = {"sample_fps": sample_fps}
            if _extract_youtube_video_id(url):
                payload["youtube_url"] = url
            else:
                payload["video_url"] = url
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(MODAL_PREPROCESS_ENDPOINT, json=payload)
            if r.status_code != 200:
                err = r.json() if r.content else {}
                raise HTTPException(
                    status_code=r.status_code,
                    detail=err.get("error", r.text) or "Modal preprocess failed",
                )
            return JSONResponse(content=r.json())
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Preprocess timed out") from None
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Modal preprocess failed: {e}") from e

    video_id = _extract_youtube_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="A YouTube video URL is required (or set MODAL_PREPROCESS_ENDPOINT for other URLs)")
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


@app.post("/api/preprocess/video")
async def preprocess_video_upload(file: UploadFile = File(...), sample_fps: float = 8.0) -> JSONResponse:
    """Extract pose frames from uploaded video (e.g. AI-generated). Proxies to Modal when MODAL_PREPROCESS_ENDPOINT is set."""
    sample_fps = max(0.5, min(30.0, sample_fps))
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty video file")

    if MODAL_PREPROCESS_ENDPOINT:
        try:
            import base64
            payload = {
                "video_base64": base64.b64encode(data).decode("ascii"),
                "sample_fps": sample_fps,
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(MODAL_PREPROCESS_ENDPOINT, json=payload)
            if r.status_code != 200:
                err = r.json() if r.content else {}
                raise HTTPException(
                    status_code=r.status_code,
                    detail=err.get("error", r.text) or "Modal preprocess failed",
                )
            return JSONResponse(content=r.json())
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Preprocess timed out") from None
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Modal preprocess failed: {e}") from e

    raise HTTPException(
        status_code=503,
        detail="Video upload preprocess requires MODAL_PREPROCESS_ENDPOINT. Deploy: modal deploy compare.modal_compare_app",
    )


@app.get("/api/youtube/{video_id}")
async def youtube_stream(video_id: str):
    """Stream YouTube video through the backend so the browser loads from our origin (no redirect to YouTube)."""
    if not video_id or len(video_id) > 20:
        raise HTTPException(status_code=400, detail="Invalid video ID")
    try:
        stream_url = _get_youtube_stream_url(video_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not get stream: {e}") from e

    async def stream_body():
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            async with client.stream("GET", stream_url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream_body(),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


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
MODAL_COMPARE_ENDPOINT = os.environ.get("MODAL_COMPARE_ENDPOINT", "").rstrip("/")
MODAL_COMPARE_FULL_ENDPOINT = os.environ.get("MODAL_COMPARE_FULL_ENDPOINT", "").rstrip("/")
MODAL_PREPROCESS_ENDPOINT = os.environ.get("MODAL_PREPROCESS_ENDPOINT", "").rstrip("/")

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
        raise HTTPException(status_code=503, detail="Set MODAL_VIDEO_DOWNLOAD_BASE")
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


@app.post("/api/pose/compare")
async def pose_compare(body: CompareRequest) -> JSONResponse:
    """Compare reference pose sequence with user pose sequence. Runs locally (lightweight DTW)."""
    from compare.compare_core import compare_poses
    result = compare_poses(
        body.reference,
        body.user,
        (body.exercise or "").strip(),
        (body.exercise_muscle or "").strip(),
    )
    return JSONResponse(content=result)


@app.post("/api/pose/compare-full")
async def pose_compare_full(body: CompareFullRequest) -> JSONResponse:
    """Full pipeline: preprocess YouTube (if url) + compare.
    Proxies to Modal if MODAL_COMPARE_FULL_ENDPOINT is set."""
    if MODAL_COMPARE_FULL_ENDPOINT:
        try:
            payload = {
                "user": body.user,
                "exercise": (body.exercise or "").strip(),
                "exercise_muscle": (body.exercise_muscle or "").strip(),
                "sample_fps": body.sample_fps,
            }
            if body.youtube_url:
                payload["youtube_url"] = body.youtube_url
            if body.reference:
                payload["reference"] = body.reference
            async with httpx.AsyncClient(timeout=600.0) as client:
                r = await client.post(MODAL_COMPARE_FULL_ENDPOINT, json=payload)
            if r.status_code != 200:
                err = r.json() if r.content else {}
                raise HTTPException(
                    status_code=r.status_code,
                    detail=err.get("error", r.text) or f"Modal compare-full returned {r.status_code}",
                )
            return JSONResponse(content=r.json())
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Compare-full timed out") from None
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Modal compare-full failed: {e}") from e

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
        (body.exercise_muscle or "").strip(),
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

