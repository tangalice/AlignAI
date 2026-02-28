# FormAI

Simple demo with a React frontend that runs a MediaPipe Pose model in WebAssembly directly in the browser and shows it on live webcam video.

## Frontend (React + Vite)

The frontend lives in the `frontend` folder.

- **Install dependencies**:

```bash
cd frontend
npm install
```

- **Run the dev server**:

```bash
npm run dev
```

By default this runs on `http://localhost:5173`.

When you open the app in the browser, allow webcam access. You should see your live camera feed and an annotated video where the pose skeleton is overlaid, powered entirely in-browser by a WebAssembly MediaPipe Pose model (no Python backend needed).
