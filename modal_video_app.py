"""
Modal app: Perplexity research → GPT-4o synthesis → Grok video generation.

Pipeline:
  1. Perplexity: research workout/exercise form and best practices for the user's description
  2. GPT-4o: synthesize research + description into a single, detailed video-generation prompt
  3. Grok (xAI): text-to-video with grok-imagine-video; download result and return bytes

Secret: create a Modal secret named `formai-video-keys` with keys
  PERPLEXITY_API_KEY, OPENAI_API_KEY, XAI_API_KEY
(Or attach multiple secrets that provide these env vars.)

Deploy:
  modal deploy modal_video_app

Set MODAL_VIDEO_ENDPOINT to the deployed web endpoint URL.
"""

import os
import time

import modal
from fastapi.responses import Response

app = modal.App("formai-video")

# Lightweight image: no GPU, just HTTP and OpenAI/Perplexity clients
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("openai", "httpx", "fastapi")
)

MINUTES = 60


def _perplexity_research(user_prompt: str) -> str:
    """Research workout/exercise topic via Perplexity (Sonar)."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    research_prompt = (
        f"Research this workout or exercise for a short instructional video: \"{user_prompt}\". "
        "Summarize: (1) correct form and key movements, (2) what the viewer should see in a 5–10 second "
        "demo clip, (3) any safety or form cues. Keep the summary under 300 words and focused on "
        "what to show visually."
    )
    resp = client.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": research_prompt}],
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()


def _gpt4o_synthesize(user_prompt: str, research: str) -> str:
    """Synthesize research + user prompt into one video-generation prompt for Grok."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    system = (
        "You output a single, detailed text prompt for an AI video model (text-to-video). "
        "The prompt should describe exactly what to show in a 5–10 second workout/exercise clip: "
        "the person, setting, movements, camera angle, and lighting. No narration or bullet points—"
        "one flowing paragraph suitable for video generation. Be specific and visual."
    )
    user = (
        f"User request: {user_prompt}\n\n"
        f"Research context:\n{research}\n\n"
        "Output only the video prompt, nothing else."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=500,
    )
    return (resp.choices[0].message.content or "").strip()


def _grok_generate_video(prompt: str) -> bytes:
    """Create Grok video job, poll until done, download video and return bytes."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Start generation
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.x.ai/v1/videos/generations",
            headers=headers,
            json={
                "model": "grok-imagine-video",
                "prompt": prompt,
                "duration": 10,
                "aspect_ratio": "16:9",
                "resolution": "720p",
            },
        )
    r.raise_for_status()
    data = r.json()
    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError("xAI did not return request_id")
    # Poll until done
    poll_url = f"https://api.x.ai/v1/videos/{request_id}"
    for _ in range(120):  # up to ~10 min at 5s intervals
        with httpx.Client(timeout=30) as client:
            r = client.get(poll_url, headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "done":
            video_info = data.get("video") or {}
            url = video_info.get("url")
            if not url:
                raise RuntimeError("xAI response missing video URL")
            # Download video bytes
            with httpx.Client(timeout=60) as client:
                r = client.get(url)
            r.raise_for_status()
            return r.content
        if status == "expired":
            raise RuntimeError("xAI video request expired")
        time.sleep(5)
    raise RuntimeError("xAI video generation timed out")


@app.function(
    image=image,
    timeout=15 * MINUTES,
    secrets=[
        modal.Secret.from_name("formai-video-keys"),
    ],
)
@modal.web_endpoint(method="POST")
async def generate(request):
    """
    POST with JSON: {"prompt": "description of workout/exercise"}.
    Pipeline: Perplexity research → GPT-4o synthesis → Grok video gen → return MP4.
    """
    try:
        body = await request.json()
    except Exception:
        return Response("Invalid JSON", status_code=400)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return Response("Missing or empty prompt", status_code=400)

    try:
        research = _perplexity_research(prompt)
        video_prompt = _gpt4o_synthesize(prompt, research)
        video_bytes = _grok_generate_video(video_prompt)
    except RuntimeError as e:
        return Response(str(e), status_code=502)
    except Exception as e:
        return Response(f"Pipeline error: {e}", status_code=500)

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": "inline; filename=ai-workout.mp4"},
    )
