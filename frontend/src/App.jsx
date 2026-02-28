import React, { useEffect, useRef, useState } from "react";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

function App() {
  const videoRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream;

    async function setupCamera() {
      try {
        setError(null);
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setIsStreaming(true);
        }
      } catch (err) {
        setError("Could not access webcam. Please allow camera permissions.");
        console.error(err);
      }
    }

    setupCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });

        if (!cancelled) {
          poseLandmarkerRef.current = poseLandmarker;
          setIsModelReady(true);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load pose model (WASM). Check your network.");
        }
      }
    }

    loadModel();

    return () => {
      cancelled = true;
      if (poseLandmarkerRef.current) {
        poseLandmarkerRef.current.close();
        poseLandmarkerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!isStreaming || !isModelReady) return;

    let animationFrameId;

    const renderLoop = () => {
      if (!videoRef.current || !outputCanvasRef.current || !poseLandmarkerRef.current) {
        animationFrameId = requestAnimationFrame(renderLoop);
        return;
      }

      const video = videoRef.current;
      const canvas = outputCanvasRef.current;
      const ctx = canvas.getContext("2d");

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        animationFrameId = requestAnimationFrame(renderLoop);
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const nowInMs = performance.now();
      const result = poseLandmarkerRef.current.detectForVideo(video, nowInMs);

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const drawingUtils = new DrawingUtils(ctx);

      if (result.landmarks && result.landmarks.length > 0) {
        for (const landmarks of result.landmarks) {
          drawingUtils.drawLandmarks(landmarks, {
            radius: 3,
            color: "#3fb0ff",
          });
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: "#ff3f81",
            lineWidth: 2,
          });
        }
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isStreaming, isModelReady]);

  return (
    <div className="app">
      <header className="header">
        <h1>FormAI Pose Demo</h1>
        <p>Webcam pose estimation fully in-browser with a WebAssembly MediaPipe model.</p>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>Webcam</h2>
          <div className="video-container">
            <video ref={videoRef} autoPlay playsInline muted className="video" />
          </div>
        </section>

        <section className="panel">
          <h2>Pose Model Output (WASM)</h2>
          <div className="video-container output">
            <canvas ref={outputCanvasRef} className="video" />
          </div>
        </section>
      </main>

      {(!isModelReady || !isStreaming) && !error && (
        <div className="placeholder" style={{ textAlign: "center", marginTop: "8px" }}>
          {!isStreaming
            ? "Waiting for webcam permissions…"
            : "Loading pose model (WASM)…"}
        </div>
      )}

      {error && (
        <div className="error-banner">
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

export default App;

