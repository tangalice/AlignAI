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

Runs at `http://localhost:8001`.

### Frontend

```bash
cd FormAI/frontend
npm install
npm run dev
```

Runs at `http://localhost:8000`.

When you open the app, allow webcam access. You should see your live camera feed and an annotated video where the pose skeleton is overlaid, powered entirely in-browser by a WebAssembly MediaPipe Pose model (no Python backend needed).

## Tabs

- **YouTube Video**: Paste a YouTube link to load a workout video in the left panel and compare with your webcam pose.
- **AI Generated Video**: Enter a text description of a workout/exercise; the app runs a pipeline on [Modal.com](https://modal.com) and displays the result in the same left-panel layout with play/pause, seek, volume, and speed controls.

### AI video pipeline (Modal.com)

The pipeline runs on Modal: **Perplexity** (research) → **GPT-4o** (synthesis) → **Grok** (xAI video gen).

1. Install Modal and authenticate: `pip install modal` then `modal setup`.
2. Create a Modal secret with your API keys (Modal dashboard → Secrets, or CLI):
   - Secret name: `formai-video-keys`
   - Keys: `PERPLEXITY_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`
3. Deploy: `modal deploy modal_video_app`
4. Set `MODAL_VIDEO_ENDPOINT` to your deployed web endpoint URL (e.g. `https://your-workspace--formai-video-generate.modal.run`) and run the backend.

Without `MODAL_VIDEO_ENDPOINT` or the API keys, the AI tab will show an error when you click Generate; the YouTube tab and pose demo still work.
