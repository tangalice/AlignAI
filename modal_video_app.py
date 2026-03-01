"""
Modal app: Perplexity (research + final video prompt) -> xAI video -> store -> SSE progress + download.
Also: Eleven Labs TTS for voice coach (coach-like voice).

Endpoints:
  POST /generate            (text/event-stream)
  GET  /download/{gen_id}   (video/mp4)
  POST /tts                 (JSON { "text": "..." } -> audio/mpeg)

Modal Secret (name: formai-video-keys):
  PERPLEXITY_API_KEY
  XAI_API_KEY
  ELEVENLABS_API_KEY        (for TTS; optional ELEVENLABS_VOICE_ID for coach voice)
"""

import json
import os
import time
import uuid
import httpx
import unicodedata
import anyio

import modal
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response

APP_NAME = "formai-video"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    .pip_install("fastapi", "httpx", "openai")
)

vol = modal.Volume.from_name("formai-videos", create_if_missing=True)
VOL_PATH = "/vol"

MINUTES = 60


def sse(data: dict) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def sanitize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(ch for ch in s if ch >= " " or ch in "\n\t").strip()


def perplexity_make_video_prompt(user_prompt: str) -> str:
    """
    Perplexity Sonar does web-grounded research and outputs the FINAL Grok video prompt.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    # Keep this *very strict* so Grok gets a simple/fast scene.
    instruction = f"""
Research the exercise/workout requested: {user_prompt}

Then output ONE single-paragraph text-to-video prompt for a 8–10 second clip.
STRICT REQUIREMENTS:
- STATIC CAMERA: locked tripod, no camera movement, no zoom
- PLAIN BACKGROUND: empty white studio or light gray wall, no props, no gym equipment
- FULL BODY: head-to-toe visible, centered
- ONE PERSON facing forward
- SIMPLE LIGHTING: even soft light
- NO TEXT, NO CAPTIONS, NO LOGOS
- NO NARRATION, NO AUDIO
- MOVEMENT: only the exercise, slow and controlled, no extra gestures

The output must be ONLY the final video prompt paragraph. No bullets, no headings, no citations.
Max 120 words.
""".strip()

    resp = client.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": instruction}],
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()


def xai_generate_video(video_prompt: str, on_poll=None,*, duration=10, aspect_ratio="4:3", resolution="480p") -> bytes:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.x.ai/v1/videos/generations",
            headers=headers,
            json={
                "model": "grok-imagine-video",
                "prompt": video_prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
            },
        )
        r.raise_for_status()
        request_id = r.json().get("request_id")
        if not request_id:
            raise RuntimeError(f"xAI did not return request_id")

        poll_url = f"https://api.x.ai/v1/videos/{request_id}"

        last_data = None
        last_status = None
        deadline = time.time() + 15 * 60  # 15 min

        while time.time() < deadline:
            rr = client.get(poll_url, headers={"Authorization": f"Bearer {api_key}"})
            rr.raise_for_status()
            data = rr.json()

            if data != last_data:
                print("xAI poll:", data)
            last_data = data
            url = (data.get("video") or {}).get("url")
            status = (data.get("status") or "").lower()
            last_status = status

            if on_poll:
                on_poll(data)

            if url or status == "done":
                if not url:
                    url = (data.get("video") or {}).get("url")
                if not url:
                    raise RuntimeError(f"xAI done but missing video.url: {data}")
                vid = client.get(url, timeout=httpx.Timeout(300.0))
                vid.raise_for_status()
                return vid.content

            if status in ("failed", "expired"):
                raise RuntimeError(f"xAI status={status}: {data}")

            time.sleep(2)

        raise RuntimeError(f"xAI video generation timed out (15min). last_status={last_status}, last_data={last_data}")


api = FastAPI(title="FormAI Video (Modal)")


@api.post("/generate")
async def generate(request: Request):
    body = await request.json()
    user_prompt = sanitize_text(body.get("prompt", ""))

    if len(user_prompt) < 3:
        return Response(
            content=json.dumps({"error": "Prompt too short"}),
            media_type="application/json",
            status_code=400,
        )

    gen_id = f"gen_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    out_path = f"{VOL_PATH}/{gen_id}.mp4"

    async def stream():
        try:
            yield sse({"type": "genId", "id": gen_id})
            yield sse({"type": "progress", "percent": 10, "phase": "researching+prompting"})

            video_prompt = await anyio.to_thread.run_sync(lambda: perplexity_make_video_prompt(user_prompt))
            video_prompt = sanitize_text(video_prompt)

            yield sse({"type": "progress", "percent": 40, "phase": "prompt_ready"})
            yield sse({"type": "synthesis", "description": video_prompt})

            yield sse({"type": "progress", "percent": 55, "phase": "generating"})
            video_bytes = await anyio.to_thread.run_sync(lambda: xai_generate_video(video_prompt, duration=10, aspect_ratio="4:3", resolution="480p"))

            yield sse({"type": "progress", "percent": 90, "phase": "saving"})

            with open(out_path, "wb") as f:
                f.write(video_bytes)
            await vol.commit.aio()

            yield sse({"type": "progress", "percent": 100, "phase": "done"})
            yield sse({"type": "done", "id": gen_id})

        except Exception as e:
            yield sse({"type": "error", "message": sanitize_text(repr(e))})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# Eleven Labs TTS default: natural/conversational. Override with ELEVENLABS_VOICE_ID in Modal secret.
# Natural: Rachel 21m00Tcm4TlvDq8ikWAM, Bella EXAVITQu4vr4xnSDxMaL. Coach: Adam pNInz6obpgDQGcFmaJgB, Josh TxGEqnHWrfWFTfGW9XjX.
ELEVENLABS_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel – calm, conversational


@api.post("/tts")
async def tts(request: Request):
    """Convert text to speech via Eleven Labs; returns MP3. Default voice: natural/conversational (override with ELEVENLABS_VOICE_ID in secret)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = (body.get("text") or "").strip()
    if not text:
        return Response(
            content=json.dumps({"error": "text is required"}),
            media_type="application/json",
            status_code=400,
        )
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return Response(
            content=json.dumps({"error": "ELEVENLABS_API_KEY not set"}),
            media_type="application/json",
            status_code=503,
        )
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", ELEVENLABS_DEFAULT_VOICE_ID).strip()
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
        return Response(
            content=json.dumps({"error": f"ElevenLabs request failed: {e}"}),
            media_type="application/json",
            status_code=502,
        )
    if r.status_code != 200:
        return Response(
            content=json.dumps({"error": f"ElevenLabs error: {r.status_code} {r.text[:200]}"}),
            media_type="application/json",
            status_code=502,
        )
    return Response(
        content=r.content,
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(r.content))},
    )


@api.get("/download/{gen_id}")
def download(gen_id: str):
    gen_id = sanitize_text(gen_id)

    # IMPORTANT: make latest committed files visible in this container
    vol.reload()

    path = f"{VOL_PATH}/{gen_id}.mp4"
    if not os.path.exists(path):
        return Response(content="Not found", status_code=404, media_type="text/plain")

    with open(path, "rb") as f:
        data = f.read()

    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{gen_id}.mp4"'},
    )


@app.function(
    image=image,
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("formai-video-keys")],
    volumes={VOL_PATH: vol},
)
@modal.asgi_app()
def fastapi_app():
    return api