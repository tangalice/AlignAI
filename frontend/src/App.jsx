import React, { useEffect, useRef, useState } from "react";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import { comparePoseWithCoaching } from "./comparePose";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

function getYouTubeVideoId(url) {
  if (!url || typeof url !== "string") return null;
  const trimmed = url.trim();
  const watchMatch = trimmed.match(/(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})/);
  const shortMatch = trimmed.match(/(?:youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  const shortsMatch = trimmed.match(/(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/);
  const embedMatch = trimmed.match(/(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/);
  return watchMatch?.[1] ?? shortMatch?.[1] ?? shortsMatch?.[1] ?? embedMatch?.[1] ?? null;
}

function App() {
  const videoRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);

  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [error, setError] = useState(null);
  const [ytPlaying, setYtPlaying] = useState(false);
  const [ytProgress, setYtProgress] = useState(0);
  const [ytDuration, setYtDuration] = useState(0);
  const [ytVolume, setYtVolume] = useState(100);
  const [ytSpeed, setYtSpeed] = useState(1);
  const [ytLoading, setYtLoading] = useState(false);
  const [ytError, setYtError] = useState(null);
  const [preprocessLoading, setPreprocessLoading] = useState(false);
  const [preprocessResult, setPreprocessResult] = useState(null);
  const [preprocessError, setPreprocessError] = useState(null);
  const [copyFeedback, setCopyFeedback] = useState(false);
  const [coachingFeedback, setCoachingFeedback] = useState(null);

  // Tabs: "youtube" | "ai"
  const [activeTab, setActiveTab] = useState("youtube");
  const [aiDescription, setAiDescription] = useState("");
  const [aiVideoUrl, setAiVideoUrl] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [aiPlaying, setAiPlaying] = useState(false);
  const [aiProgress, setAiProgress] = useState(0);
  const [aiDuration, setAiDuration] = useState(0);
  const [aiVolume, setAiVolume] = useState(100);
  const [aiSpeed, setAiSpeed] = useState(1);

  const ytVideoRef = useRef(null);
  const aiVideoRef = useRef(null);
  const ytProgressIntervalRef = useRef(null);

  const youtubeVideoId = getYouTubeVideoId(youtubeUrl);
  const ytVideoSrc = youtubeVideoId ? `/api/youtube/${youtubeVideoId}` : null;

  const apiBase = import.meta.env.VITE_API_BASE || "";

  const handleGenerateAiVideo = async () => {
    const prompt = (aiDescription || "").trim();
    if (!prompt) {
      setAiError("Please enter a description for the workout/exercise video.");
      return;
    }
    setAiError(null);
    setAiLoading(true);
    if (aiVideoUrl) {
      URL.revokeObjectURL(aiVideoUrl);
      setAiVideoUrl(null);
    }
    try {
      const res = await fetch(`${apiBase}/api/ai-video/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Generation failed (${res.status})`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setAiVideoUrl(url);
    } catch (err) {
      setAiError(err.message || "Failed to generate video.");
    } finally {
      setAiLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (aiVideoUrl) URL.revokeObjectURL(aiVideoUrl);
    };
  }, [aiVideoUrl]);

  // Sync playback rate and volume when video element or state changes
  useEffect(() => {
    const el = ytVideoRef.current;
    if (!el) return;
    el.playbackRate = ytSpeed;
    el.volume = ytVolume / 100;
  }, [ytSpeed, ytVolume, ytVideoSrc]);

  useEffect(() => {
    const el = aiVideoRef.current;
    if (!el) return;
    el.playbackRate = aiSpeed;
    el.volume = aiVolume / 100;
  }, [aiSpeed, aiVolume, aiVideoUrl]);

  // Progress tick when playing
  useEffect(() => {
    if (!ytPlaying) return;
    const id = setInterval(() => {
      const el = ytVideoRef.current;
      if (!el) return;
      const d = el.duration;
      if (Number.isFinite(d) && d > 0) setYtProgress((el.currentTime / d) * 100);
    }, 250);
    return () => clearInterval(id);
  }, [ytPlaying]);

  useEffect(() => {
    if (!aiPlaying) return;
    const id = setInterval(() => {
      const el = aiVideoRef.current;
      if (!el) return;
      const d = el.duration;
      if (Number.isFinite(d) && d > 0) setAiProgress((el.currentTime / d) * 100);
    }, 250);
    return () => clearInterval(id);
  }, [aiPlaying]);

  const handleYtPlayPause = () => {
    const el = ytVideoRef.current;
    if (!el) return;
    if (ytPlaying) el.pause();
    else el.play().catch(() => {});
  };

  const handleYtTimeUpdate = () => {
    const el = ytVideoRef.current;
    if (!el) return;
    const d = el.duration;
    if (Number.isFinite(d) && d > 0) {
      setYtDuration(d);
      setYtProgress((el.currentTime / d) * 100);
    }
  };

  const handleYtSeek = (e) => {
    const value = Number(e.target.value);
    const el = ytVideoRef.current;
    if (!el || !Number.isFinite(el.duration)) return;
    el.currentTime = (value / 100) * el.duration;
    setYtProgress(value);
  };

  const handleYtVolume = (e) => {
    const value = Math.round(Number(e.target.value));
    setYtVolume(value);
    const el = ytVideoRef.current;
    if (el) el.volume = value / 100;
  };

  const handleYtSpeed = (e) => {
    const value = Number(e.target.value);
    setYtSpeed(value);
  };

  const handleAiPlayPause = () => {
    const el = aiVideoRef.current;
    if (!el) return;
    if (aiPlaying) el.pause();
    else el.play().catch(() => {});
  };

  const handleAiTimeUpdate = () => {
    const el = aiVideoRef.current;
    if (!el) return;
    const d = el.duration;
    if (Number.isFinite(d) && d > 0) {
      setAiDuration(d);
      setAiProgress((el.currentTime / d) * 100);
    }
  };

  const handleAiSeek = (e) => {
    const value = Number(e.target.value);
    const el = aiVideoRef.current;
    if (!el || !Number.isFinite(el.duration)) return;
    el.currentTime = (value / 100) * el.duration;
    setAiProgress(value);
  };

  const handleAiVolume = (e) => {
    const value = Math.round(Number(e.target.value));
    setAiVolume(value);
    const el = aiVideoRef.current;
    if (el) el.volume = value / 100;
  };

  const handleAiSpeed = (e) => {
    setAiSpeed(Number(e.target.value));
  };

  const handlePreprocess = async () => {
    const url = youtubeUrl.trim();
    if (!url) {
      setPreprocessError("Paste a YouTube link first.");
      return;
    }
    const video = ytVideoRef.current;
    if (!video) {
      setPreprocessError("Load the YouTube video first (press play once).");
      return;
    }
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      setPreprocessError("Video metadata not loaded yet. Wait until the video is ready.");
      return;
    }

    const seekTo = (timeSec) =>
      new Promise((resolve, reject) => {
        const el = ytVideoRef.current;
        if (!el) {
          reject(new Error("Video element not available"));
          return;
        }
        const onSeeked = () => {
          el.removeEventListener("seeked", onSeeked);
          el.removeEventListener("error", onError);
          resolve();
        };
        const onError = (e) => {
          el.removeEventListener("seeked", onSeeked);
          el.removeEventListener("error", onError);
          reject(e?.error || new Error("Seek failed"));
        };
        el.addEventListener("seeked", onSeeked);
        el.addEventListener("error", onError);
        const clamped = Math.min(Math.max(timeSec, 0), el.duration || timeSec);
        el.currentTime = clamped;
      });

    setPreprocessLoading(true);
    setPreprocessError(null);
    setPreprocessResult(null);

    const originalPaused = video.paused;
    const originalTime = video.currentTime;
    const originalRate = video.playbackRate;

    let offlineLandmarker = null;

    try {
      // Create a fresh pose landmarker in IMAGE mode so we don't share video
      // timestamps with the live webcam graph.
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
      offlineLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "IMAGE",
        numPoses: 1,
      });

      video.pause();
      video.playbackRate = 1;

      const duration = video.duration;
      const targetFps = 30; // logical timeline resolution for Supermemory
      const step = 1 / targetFps;

      const frames = [];
      let index = 0;

      for (let t = 0; t <= duration; t += step) {
        // eslint-disable-next-line no-await-in-loop
        await seekTo(t);
        const tsMs = t * 1000;
        const result = offlineLandmarker.detect(video);

        let landmarks = [];
        if (result.landmarks && result.landmarks.length > 0) {
          // Only keep the first pose; each landmark is [x, y, z].
          landmarks = result.landmarks[0].map((lm) => [lm.x, lm.y, lm.z]);
        }

        frames.push({
          index,
          t: Number(t.toFixed(3)),
          ms: Math.round(tsMs),
          landmarks,
        });
        index += 1;
      }

      const payload = {
        source_url: url,
        duration_sec: Number(duration.toFixed(3)),
        sample_fps: targetFps,
        frame_count: frames.length,
        frames,
      };

      setPreprocessResult(JSON.stringify(payload, null, 2));
    } catch (err) {
      setPreprocessError(
        err instanceof Error ? err.message : "Preprocess failed in browser. Try again or shorten the video."
      );
    } finally {
      try {
        if (offlineLandmarker) offlineLandmarker.close();
      } catch {
        // ignore
      }
      try {
        video.currentTime = originalTime;
        video.playbackRate = originalRate;
        if (!originalPaused) {
          // resume playback best-effort
          // eslint-disable-next-line @typescript-eslint/no-floating-promises
          video.play().catch(() => {});
        }
      } catch {
        // ignore restore errors
      }
      setPreprocessLoading(false);
    }
  };

  const handleCopyPreprocess = async () => {
    if (!preprocessResult) return;
    try {
      await navigator.clipboard.writeText(preprocessResult);
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 2000);
    } catch (_) {
      setPreprocessError("Copy failed");
    }
  };

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

  const referenceFramesRef = useRef(null);

  useEffect(() => {
    if (!preprocessResult) {
      referenceFramesRef.current = null;
      return;
    }
    try {
      const parsed = JSON.parse(preprocessResult);
      referenceFramesRef.current = parsed?.frames ?? null;
    } catch {
      referenceFramesRef.current = null;
    }
  }, [preprocessResult]);

  const lastCompareMsRef = useRef(-Infinity);

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

        const liveLandmarks = result.landmarks[0].map((lm) => [lm.x, lm.y, lm.z]);
        const now = performance.now();
        if (now - lastCompareMsRef.current >= 1000 / 8) {
          lastCompareMsRef.current = now;
          const refs = referenceFramesRef.current;
          const ytVideo = ytVideoRef.current;
          if (
            activeTab === "youtube" &&
            refs?.length &&
            ytVideo &&
            Number.isFinite(ytVideo.currentTime)
          ) {
            const videoT = ytVideo.currentTime;
            const closest = refs.reduce((a, b) =>
              Math.abs(a.t - videoT) <= Math.abs(b.t - videoT) ? a : b
            );
            if (closest?.landmarks?.length) {
              const { score, limbScores, feedback } = comparePoseWithCoaching(
                { landmarks: closest.landmarks },
                { landmarks: liveLandmarks }
              );
              setCoachingFeedback(feedback);
              fetch(`${apiBase}/api/compare-pose`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ video_t: videoT, score, limbScores, feedback }),
              }).catch(() => {});
            } else {
              setCoachingFeedback(null);
            }
          }
        }
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [isStreaming, isModelReady, activeTab, apiBase]);

  useEffect(() => {
    if (activeTab !== "youtube") setCoachingFeedback(null);
  }, [activeTab]);

  const leftPanelTitle = activeTab === "youtube" ? "YouTube Video" : "AI Generated Video";

  return (
    <div className="app">
      <header className="header">
        <h1>FormAI Pose Demo</h1>
        <p>
          {activeTab === "youtube"
            ? "Paste a YouTube video link and compare with live pose estimation from your webcam."
            : "Describe a workout or exercise and generate an AI video to compare with your webcam."}
        </p>
      </header>

      <div className="tabs">
        <button
          type="button"
          className={`tab ${activeTab === "youtube" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("youtube")}
        >
          YouTube Video
        </button>
        <button
          type="button"
          className={`tab ${activeTab === "ai" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("ai")}
        >
          AI Generated Video
        </button>
      </div>

      {activeTab === "youtube" && (
        <div className="input-row">
          <label className="input-label" htmlFor="youtube-url">
            YouTube video link
          </label>
          <input
            id="youtube-url"
            type="url"
            className="youtube-input"
            placeholder="https://www.youtube.com/watch?v=... or /shorts/..."
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
          />
        </div>
      )}

      {activeTab === "ai" && (
        <div className="input-row">
          <label className="input-label" htmlFor="ai-description">
            Describe the workout or exercise video
          </label>
          <textarea
            id="ai-description"
            className="ai-description-input"
            placeholder="e.g. Person doing jumping jacks in a gym, 10 seconds"
            value={aiDescription}
            onChange={(e) => setAiDescription(e.target.value)}
            rows={3}
          />
          <button
            type="button"
            className="generate-btn"
            onClick={handleGenerateAiVideo}
            disabled={aiLoading}
          >
            {aiLoading ? "Generating…" : "Generate Video"}
          </button>
          {aiError && <div className="ai-error">{aiError}</div>}
        </div>
      )}

      <main className="side-by-side">
        <section className="panel">
          <h2>{leftPanelTitle}</h2>
          <div className="youtube-video-wrap">
            {activeTab === "youtube" && ytVideoSrc ? (
              <>
                <div className="youtube-video-area">
                  <video
                    key={youtubeVideoId}
                    ref={ytVideoRef}
                    src={ytVideoSrc}
                    className="youtube-native-video"
                    playsInline
                    preload="auto"
                    poster={youtubeVideoId ? `https://img.youtube.com/vi/${youtubeVideoId}/hqdefault.jpg` : undefined}
                    onPlay={() => setYtPlaying(true)}
                    onPause={() => setYtPlaying(false)}
                    onTimeUpdate={handleYtTimeUpdate}
                    onLoadedMetadata={(e) => setYtDuration(e.target.duration ?? 0)}
                    onEnded={() => setYtProgress(0)}
                    onLoadStart={() => { setYtLoading(true); setYtError(null); }}
                    onCanPlay={() => setYtLoading(false)}
                    onError={(e) => {
                      setYtLoading(false);
                      setYtError("Video failed to load. Is the backend running on port 8001?");
                    }}
                  />
                  {ytLoading && (
                    <div className="youtube-loading-overlay">Loading video…</div>
                  )}
                  {ytError && (
                    <div className="youtube-error-overlay">{ytError}</div>
                  )}
                </div>
                <div className="player-controls">
                  <button
                    type="button"
                    className="control-btn play-pause"
                    onClick={handleYtPlayPause}
                    aria-label={ytPlaying ? "Pause" : "Play"}
                  >
                    {ytPlaying ? (
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <rect x="6" y="4" width="4" height="16" rx="1" />
                        <rect x="14" y="4" width="4" height="16" rx="1" />
                      </svg>
                    ) : (
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    )}
                  </button>
                  <div className="progress-wrap">
                    <input
                      type="range"
                      className="progress-range"
                      min="0"
                      max="100"
                      step="0.1"
                      value={ytProgress}
                      onInput={handleYtSeek}
                      aria-label="Seek"
                    />
                  </div>
                  <div className="volume-wrap">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71V20.77c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
                    </svg>
                    <input
                      type="range"
                      className="volume-range"
                      min="0"
                      max="100"
                      value={ytVolume}
                      onInput={handleYtVolume}
                      aria-label="Volume"
                    />
                  </div>
                  <div className="speed-wrap">
                    <span className="speed-label" title="Playback speed">Speed</span>
                    <input
                      type="range"
                      className="speed-range"
                      min="0.25"
                      max="2"
                      step="0.25"
                      value={ytSpeed}
                      onInput={handleYtSpeed}
                      aria-label="Playback speed"
                    />
                    <span className="speed-value">{ytSpeed}x</span>
                  </div>
                </div>
              </>
            ) : activeTab === "ai" && (aiLoading || aiVideoUrl) ? (
              <>
                <div className="youtube-video-area">
                  {aiLoading ? (
                    <div className="youtube-loading-overlay">
                      Generating video… This may take a few minutes.
                    </div>
                  ) : aiVideoUrl ? (
                    <video
                      ref={aiVideoRef}
                      src={aiVideoUrl}
                      className="youtube-native-video"
                      playsInline
                      preload="auto"
                      onPlay={() => setAiPlaying(true)}
                      onPause={() => setAiPlaying(false)}
                      onTimeUpdate={handleAiTimeUpdate}
                      onLoadedMetadata={(e) => setAiDuration(e.target.duration ?? 0)}
                      onEnded={() => setAiProgress(0)}
                    />
                  ) : null}
                </div>
                {aiVideoUrl && (
                  <div className="player-controls">
                    <button
                      type="button"
                      className="control-btn play-pause"
                      onClick={handleAiPlayPause}
                      aria-label={aiPlaying ? "Pause" : "Play"}
                    >
                      {aiPlaying ? (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                          <rect x="6" y="4" width="4" height="16" rx="1" />
                          <rect x="14" y="4" width="4" height="16" rx="1" />
                        </svg>
                      ) : (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                          <path d="M8 5v14l11-7z" />
                        </svg>
                      )}
                    </button>
                    <div className="progress-wrap">
                      <input
                        type="range"
                        className="progress-range"
                        min="0"
                        max="100"
                        step="0.1"
                        value={aiProgress}
                        onInput={handleAiSeek}
                        aria-label="Seek"
                      />
                    </div>
                    <div className="volume-wrap">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71V20.77c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
                      </svg>
                      <input
                        type="range"
                        className="volume-range"
                        min="0"
                        max="100"
                        value={aiVolume}
                        onInput={handleAiVolume}
                        aria-label="Volume"
                      />
                    </div>
                    <div className="speed-wrap">
                      <span className="speed-label" title="Playback speed">Speed</span>
                      <input
                        type="range"
                        className="speed-range"
                        min="0.25"
                        max="2"
                        step="0.25"
                        value={aiSpeed}
                        onInput={handleAiSpeed}
                        aria-label="Playback speed"
                      />
                      <span className="speed-value">{aiSpeed}x</span>
                    </div>
                  </div>
                )}
              </>
            ) : activeTab === "ai" ? (
              <div className="youtube-video-area">
                <div className="youtube-placeholder-top">
                  Enter a description and click Generate Video (Modal.com backend required)
                </div>
              </div>
            ) : (
              <div className="youtube-video-area">
                <div className="youtube-placeholder-top">
                  Paste a YouTube link to display the video (backend required)
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <h2>Pose Model Output (WASM)</h2>
          <div className="video-container output">
            <video ref={videoRef} autoPlay playsInline muted className="video hidden-video" aria-hidden="true" />
            <canvas ref={outputCanvasRef} className="video" />
          </div>
          {coachingFeedback && (
            <div className="coaching-feedback">{coachingFeedback.message}</div>
          )}
        </section>
      </main>

      <section className="panel preprocess-section">
        <h2>Preprocess for Supermemory</h2>
        <p className="preprocess-desc">
          Extract pose coordinates from the YouTube link above so you can paste them into Supermemory and compare later with your webcam.
        </p>
        <div className="preprocess-actions">
          <button
            type="button"
            className="preprocess-btn"
            onClick={handlePreprocess}
            disabled={preprocessLoading || !youtubeUrl.trim()}
          >
            {preprocessLoading ? "Extracting…" : "Extract pose coordinates"}
          </button>
        </div>
        {preprocessError && (
          <div className="preprocess-error">{preprocessError}</div>
        )}
        {preprocessResult && (
          <div className="preprocess-output">
            <textarea
              readOnly
              className="preprocess-textarea"
              value={preprocessResult}
              spellCheck={false}
              aria-label="Pose coordinates JSON"
            />
            <button
              type="button"
              className="copy-btn"
              onClick={handleCopyPreprocess}
            >
              {copyFeedback ? "Copied!" : "Copy for Supermemory"}
            </button>
          </div>
        )}
      </section>

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

