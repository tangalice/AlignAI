import io
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse


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

