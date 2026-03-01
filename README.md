# FormAI

Simple demo with a React frontend that runs a MediaPipe Pose model in WebAssembly directly in the browser and shows it on live webcam video.

## Quick start

From the project root:

```bash
# Backend (Terminal 1)
./run-backend.sh

# Frontend (Terminal 2)
./run-frontend.sh
```

Then open **http://localhost:8000** in your browser. Run `npm install` in `frontend/` the first time.

---

## Manual setup

### Backend

```bash
cd FormAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pose_server.py
```

Runs at `http://localhost:8001`. For the Exercise tab, set `YMOVE_API_KEY` (get a key at [ymove.app/exercise-api/signup](https://ymove.app/exercise-api/signup)).

### Frontend

```bash
cd FormAI/frontend
npm install
npm run dev
```

Runs at `http://localhost:8000`.

When you open the app, allow webcam access. You should see your live camera feed and an annotated video where the pose skeleton is overlaid, powered entirely in-browser by a WebAssembly MediaPipe Pose model (no Python backend needed).

## Tabs

- **Exercise Workout**: Search exercises from the [YMove Exercise API](https://ymove.app/exercise-api/docs). Select an exercise to load its demo video instantly (no YouTube download), then compare your form with your webcam. Requires `YMOVE_API_KEY` (get one at [ymove.app/exercise-api/signup](https://ymove.app/exercise-api/signup)).
- **AI Generated Video**: Enter a text description of a workout/exercise; the app runs a pipeline on [Modal.com](https://modal.com) and displays the result in the same left-panel layout with play/pause, seek, volume, and speed controls.

### AI video pipeline (Modal.com)

The pipeline runs on Modal: **Perplexity** (research) → **GPT-4o** (synthesis) → **Grok** (xAI video gen).

1. Install Modal and authenticate: `pip install modal` then `modal setup`.
2. Create a Modal secret with your API keys (Modal dashboard → Secrets, or CLI):
   - Secret name: `formai-video-keys`
   - Keys: `PERPLEXITY_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`
3. Deploy: `modal deploy modal_video_app`
4. Set `MODAL_VIDEO_ENDPOINT` to your deployed web endpoint URL (e.g. `https://your-workspace--formai-video-generate.modal.run`) and run the backend.

Without `MODAL_VIDEO_ENDPOINT` or the API keys, the AI tab will show an error when you click Generate; the Exercise tab and pose demo still work if `YMOVE_API_KEY` is set.

### Voice coaching (ElevenLabs, optional)

For spoken coaching that works automatically in Chrome (without requiring a user gesture), set:

- `ELEVENLABS_API_KEY` – your [ElevenLabs](https://elevenlabs.io) API key (backend only; never sent to the frontend).
- `ELEVENLABS_VOICE_ID` – optional; default is `21m00Tcm4TlvDq8ikWAM` (Rachel).

The backend proxies TTS at `POST /api/tts`. If the key is not set, coaching falls back to browser speech plus per-limb tones.

### AI coaching (optional)

For smarter, contextual coaching feedback, set `OPENAI_API_KEY`. When enabled, the app will call GPT-4o-mini to generate short, actionable form tips (e.g. "Bring your elbow closer to your body") instead of generic "Adjust your arm" messages. The AI coaching toggle appears next to the Voice coach toggle when the key is configured.

### Pose compare (local)

The compare model runs locally on the backend (DTW alignment, joint angles, exercise-specific weights).

- **POST `/api/pose/compare`** with `reference` and `user` pose sequences → returns `{ score, per_frame, ... }`
- **POST `/api/pose/compare-full`** with `youtube_url` + `user`, or `reference` + `user` → same result

Uses MediaPipe-style 33-landmark format; preprocess outputs `{ "frames": [{ "landmarks": [[x,y,z],...] }] }`.
