import React, { useCallback, useEffect, useRef, useState } from "react";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import { comparePoseWithCoaching } from "./comparePose";
import { startVoiceCoaching } from "./voiceCoaching";
import { Toast } from "./components/Toast";
import { ExerciseSearch } from "./components/ExerciseSearch";
import { useExerciseSearch } from "./hooks/useExerciseSearch";
import { loadWorkoutHistory, saveWorkoutToHistory, clearWorkoutHistory } from "./hooks/useWorkoutHistory";
import { CoachPanel } from "./components/CoachPanel";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

function scoreToColors(score) {
  if (score == null || score < 0) return { primary: "#3fb0ff", accent: "#ff3f81" };
  if (score >= 90) return { primary: "#22c55e", accent: "#4ade80" };
  if (score >= 80) return { primary: "#4ade80", accent: "#86efac" };
  if (score >= 70) return { primary: "#a3e635", accent: "#bef264" };
  if (score >= 50) return { primary: "#eab308", accent: "#facc15" };
  if (score >= 30) return { primary: "#f97316", accent: "#fb923c" };
  return { primary: "#ef4444", accent: "#f87171" };
}

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
  const apiBase = import.meta.env.VITE_API_BASE || "";
  const videoRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);

  const [referenceVideoUrl, setReferenceVideoUrl] = useState("");
  const [referenceExerciseId, setReferenceExerciseId] = useState(null);
  const [referenceExerciseMuscle, setReferenceExerciseMuscle] = useState("");
  const [referenceExerciseName, setReferenceExerciseName] = useState("");
  const [exerciseSearchQuery, setExerciseSearchQuery] = useState("");
  const [exerciseSearchFocused, setExerciseSearchFocused] = useState(false);
  const [exerciseVideoLoading, setExerciseVideoLoading] = useState(false);
  const [toast, setToast] = useState(null);
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
  const [coachingFeedback, setCoachingFeedback] = useState(null);
  const [voiceCoachOn, setVoiceCoachOn] = useState(true);
  const [llmCoachingAvailable, setLlmCoachingAvailable] = useState(false);
  const [workoutActive, setWorkoutActive] = useState(false);
  const [workoutSummary, setWorkoutSummary] = useState(null);
  const [workoutSummaryLoading, setWorkoutSummaryLoading] = useState(false);
  const [workoutSummaryError, setWorkoutSummaryError] = useState(null);
  const [isPastedYoutubeShort, setIsPastedYoutubeShort] = useState(false);
  const [workoutHistory, setWorkoutHistory] = useState(() => loadWorkoutHistory());
  const [repCount, setRepCount] = useState(0);

  // Tabs: "youtube" | "ai"
  const [activeTab, setActiveTab] = useState("youtube");
  const [supermemoryEnabled, setSupermemoryEnabled] = useState(false);

  const { results: exerciseResults, suggestions: exerciseSearchSuggestions, loading: exerciseSearchLoading, error: exerciseSearchError } = useExerciseSearch(apiBase, exerciseSearchQuery, activeTab);
  const [aiDescription, setAiDescription] = useState("");
  const [aiVideoUrl, setAiVideoUrl] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [aiPlaying, setAiPlaying] = useState(false);
  const [aiProgress, setAiProgress] = useState(0);
  const [aiDuration, setAiDuration] = useState(0);
  const [aiVolume, setAiVolume] = useState(100);
  const [aiSpeed, setAiSpeed] = useState(1);
  const [aiStage, setAiStage] = useState("");
  const [aiProgressText, setAiProgressText] = useState("");
  const [aiPercent, setAiPercent] = useState(0);
  const [aiGenId, setAiGenId] = useState(null);
  const [aiPreprocessLoading, setAiPreprocessLoading] = useState(false);
  const [aiPreprocessError, setAiPreprocessError] = useState(null);
  const [aiPreprocessReady, setAiPreprocessReady] = useState(false);

  const ytVideoRef = useRef(null);
  const aiVideoBlobRef = useRef(null);
  const aiReferenceFramesRef = useRef(null);
  const aiVideoRef = useRef(null);
  const ytProgressIntervalRef = useRef(null);
  const ytSkeletonCanvasRef = useRef(null);

  const ytVideoSrc = referenceVideoUrl || null;
  const isYoutubeVideo = Boolean(
    referenceVideoUrl && String(referenceVideoUrl).includes("/api/youtube/")
  );
  const youtubeVideoIdFromUrl = referenceVideoUrl?.match(/\/api\/youtube\/([^/?#]+)/)?.[1] ?? null;

  const [ttsEnabled, setTtsEnabled] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/api/coaching/llm/config`)
      .then((r) => r.json())
      .then((d) => {
        setLlmCoachingAvailable(Boolean(d.enabled));
        setSupermemoryEnabled(Boolean(d.supermemory));
        setTtsEnabled(Boolean(d.tts));
      })
      .catch(() => {
        setLlmCoachingAvailable(false);
        setSupermemoryEnabled(false);
        setTtsEnabled(false);
      });
  }, [apiBase]);

  // When user pastes a YouTube link in the exercise input, use it as the video source (no iframe; backend streams via redirect).
  useEffect(() => {
    if (activeTab !== "youtube") return;
    const trimmed = exerciseSearchQuery.trim();
    const videoId = getYouTubeVideoId(trimmed);
    if (videoId) {
      setReferenceVideoUrl(`${apiBase}/api/youtube/${videoId}`);
      setIsPastedYoutubeShort(trimmed.toLowerCase().includes("shorts"));
      setReferenceExerciseId(null);
      setReferenceExerciseMuscle("");
      setReferenceExerciseName("");
      setPreprocessError(null);
      setPreprocessResult(null);
      setExerciseSearchFocused(false);
      setExerciseResults([]);
    } else if (referenceVideoUrl && String(referenceVideoUrl).includes("/api/youtube/")) {
      setReferenceVideoUrl("");
      setIsPastedYoutubeShort(false);
      setReferenceExerciseId(null);
      setReferenceExerciseMuscle("");
      setReferenceExerciseName("");
      setPreprocessResult(null);
    }
  }, [exerciseSearchQuery, activeTab, apiBase, referenceVideoUrl]);

  // Search exercises (debounced, AbortController cancels stale requests); skip when input is a YouTube link
  const exerciseSearchTimeoutRef = useRef(null);
  const exerciseSearchAbortRef = useRef(null);
  useEffect(() => {
    if (exerciseSearchTimeoutRef.current) clearTimeout(exerciseSearchTimeoutRef.current);
    if (activeTab !== "youtube") return;
    const q = exerciseSearchQuery.trim();
    if (getYouTubeVideoId(q)) return;
    const doSearch = async () => {
      if (exerciseSearchAbortRef.current) exerciseSearchAbortRef.current.abort();
      exerciseSearchAbortRef.current = new AbortController();
      const signal = exerciseSearchAbortRef.current.signal;
      setExerciseSearchLoading(true);
      setExerciseSearchSuggestions([]);
      try {
        const res = await fetch(`${apiBase}/api/exercises/search?q=${encodeURIComponent(q)}&limit=15`, { signal });
        const data = await res.json();
        setExerciseResults(data.exercises || []);
        setExerciseSearchSuggestions(data.suggestions || []);
      } catch (err) {
        if (err.name === "AbortError") return;
        setExerciseResults([]);
        setExerciseSearchSuggestions([]);
      } finally {
        if (!signal.aborted) setExerciseSearchLoading(false);
      }
    };
    if (!q) {
      doSearch();
      return;
    }
    exerciseSearchTimeoutRef.current = setTimeout(doSearch, 80);
    return () => {
      if (exerciseSearchTimeoutRef.current) clearTimeout(exerciseSearchTimeoutRef.current);
    };
  }, [exerciseSearchQuery, activeTab, apiBase]);

  const exerciseVideoAbortRef = useRef(null);
  const handleSelectExercise = useCallback(async (exercise) => {
    if (exerciseVideoAbortRef.current) exerciseVideoAbortRef.current.abort();
    exerciseVideoAbortRef.current = new AbortController();
    const signal = exerciseVideoAbortRef.current.signal;
    setExerciseVideoLoading(true);
      setReferenceVideoUrl("");
      setReferenceExerciseId(null);
      setReferenceExerciseMuscle("");
      setReferenceExerciseName("");
      setPreprocessError(null);
      setPreprocessResult(null);
      setExerciseSearchFocused(false);
    try {
      const res = await fetch(`${apiBase}/api/exercises/video?id=${encodeURIComponent(exercise.id)}`, { signal });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not find video");
      setReferenceVideoUrl(data.video_url);
      setIsPastedYoutubeShort(false);
      setReferenceExerciseId(exercise.id);
      setReferenceExerciseMuscle((exercise.muscle || "").toLowerCase());
      setReferenceExerciseName(exercise.name || "");
      setExerciseSearchQuery("");
      fetch(`${apiBase}/api/supermemory/add-exercise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exercise_name: exercise.name,
          exercise_muscle: exercise.muscle,
          video_url: data.video_url,
          exercise_id: exercise.id,
        }),
      }).catch((e) => console.warn("[supermemory] add-exercise failed:", e));
    } catch (err) {
      if (err.name === "AbortError") return;
      setPreprocessError(err.message || "Could not find exercise video. Try another exercise.");
    } finally {
      if (!signal.aborted) setExerciseVideoLoading(false);
    }
  }, [apiBase]);

  const handleGenerateAiVideo = async () => {
    const prompt = (aiDescription || "").trim();
    if (!prompt) {
      setAiError("Please enter a description.");
      return;
    }
  
    setAiError(null);
    setAiLoading(true);
    setAiProgressText("Starting…");
    setAiPercent(0);
  
    if (aiVideoUrl) {
      URL.revokeObjectURL(aiVideoUrl);
      setAiVideoUrl(null);
    }
  
    try {
      const res = await fetch(`${apiBase}/api/ai-video/generate-sse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
  
      if (!res.ok || !res.body) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let genId = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // parse SSE frames split by blank line
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;

          const jsonStr = line.slice(6);
          let evt;
          try {
            evt = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          if (evt.type === "genId") {
            genId = evt.id;
          }

          if (evt.type === "progress") {
            setAiPercent(evt.percent ?? 0);
            setAiProgressText(evt.phase ? `${evt.phase}…` : "Working…");
          }

          if (evt.type === "synthesis") {
            setAiProgressText("Generating video…");
          }

          if (evt.type === "error") {
            throw new Error(evt.message || "Generation failed");
          }

          if (evt.type === "done") {
            genId = evt.id || genId;
          }
        }
      }

      if (!genId) throw new Error("No generation id returned.");

      // Download mp4 from YOUR server (FastAPI), not directly from Modal
      const videoRes = await fetch(`${apiBase}/api/ai-video/download/${genId}`);
      if (!videoRes.ok) throw new Error(await videoRes.text());

      const blob = await videoRes.blob();
      aiVideoBlobRef.current = blob;
      aiReferenceFramesRef.current = null;
      setAiPreprocessError(null);
      setAiPreprocessReady(false);
      const url = URL.createObjectURL(blob);
      setAiVideoUrl(url);
      setAiProgressText("Done!");
      setAiPercent(100);
  
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

  // Extract poses from AI-generated video via Modal (fast) when video is ready
  useEffect(() => {
    if (!aiVideoUrl || !aiVideoBlobRef.current || aiPreprocessLoading) return;
    const blob = aiVideoBlobRef.current;
    setAiPreprocessLoading(true);
    setAiPreprocessError(null);
    const formData = new FormData();
    formData.append("file", blob, "ai-workout.mp4");
    fetch(`${apiBase}/api/preprocess/video?sample_fps=8`, {
      method: "POST",
      body: formData,
    })
      .then(async (r) => {
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || data.error || "Preprocess failed");
        aiReferenceFramesRef.current = data.frames ?? null;
        setAiPreprocessReady(Boolean(data.frames?.length));
      })
      .catch((err) => {
        setAiPreprocessError(err.message || "Could not extract poses");
        setAiPreprocessReady(false);
        aiReferenceFramesRef.current = null;
      })
      .finally(() => setAiPreprocessLoading(false));
  }, [aiVideoUrl, apiBase]);

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

  const handlePreprocess = useCallback(async () => {
    if (!referenceVideoUrl) {
      setPreprocessError("Select an exercise first.");
      return;
    }
    // Try API preprocess first (Modal GPU) when we have a fetchable URL
    if (referenceVideoUrl.startsWith("http://") || referenceVideoUrl.startsWith("https://")) {
      setPreprocessLoading(true);
      setPreprocessError(null);
      setPreprocessResult(null);
      try {
        const res = await fetch(`${apiBase}/api/preprocess`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: referenceVideoUrl, sample_fps: 8 }),
        });
        const data = await res.json();
        if (res.ok) {
          setPreprocessResult(JSON.stringify({ source_url: referenceVideoUrl, frames: data.frames }));
          setPreprocessLoading(false);
          return;
        }
        throw new Error(data.detail || data.error || "Preprocess failed");
      } catch (err) {
        setPreprocessLoading(false);
        setPreprocessError(err.message || "API preprocess failed. Trying browser…");
        // Fall through to in-browser preprocess
      }
    }

    const video = ytVideoRef.current;
    if (!video) {
      setPreprocessError("Load the exercise video first.");
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

    const originalPaused = video.paused;
    const originalTime = video.currentTime;
    const originalRate = video.playbackRate;

    setPreprocessLoading(true);
    setPreprocessError(null);
    setPreprocessResult(null);
    video.pause();
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
        source_url: referenceVideoUrl,
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
  }, [referenceVideoUrl, apiBase]);

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
  const lastPreprocessedExerciseIdRef = useRef(null);

  // Auto-extract pose coordinates when exercise video loads
  useEffect(() => {
    if (
      activeTab !== "youtube" ||
      !referenceVideoUrl ||
      !referenceExerciseId ||
      !(ytDuration > 0) ||
      preprocessLoading ||
      lastPreprocessedExerciseIdRef.current === referenceExerciseId
    ) {
      return;
    }
    lastPreprocessedExerciseIdRef.current = referenceExerciseId;
    handlePreprocess();
  }, [activeTab, referenceVideoUrl, referenceExerciseId, ytDuration, preprocessLoading, handlePreprocess]);

  useEffect(() => {
    if (!referenceExerciseId) {
      lastPreprocessedExerciseIdRef.current = null;
    }
  }, [referenceExerciseId]);

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
  const lastFetchMsRef = useRef(-Infinity);
  const voiceCoachingRef = useRef(null);
  const voiceCoachOnRef = useRef(voiceCoachOn);
  voiceCoachOnRef.current = voiceCoachOn;
  const llmCoachingAvailableRef = useRef(llmCoachingAvailable);
  llmCoachingAvailableRef.current = llmCoachingAvailable;
  const lastLlmFetchMsRef = useRef(-Infinity);
  const compareScoreRef = useRef(null); // 0–100, for overlay and skeleton color
  const [comparisonScore, setComparisonScore] = useState(null); // synced for UI display
  const workoutSamplesRef = useRef([]);
  const workoutStartedAtRef = useRef(null);
  const workoutActiveRef = useRef(false);
  workoutActiveRef.current = workoutActive;
  const poseHistoryRef = useRef([]); // Rolling window: { score, worstLimb, worstScore }
  const temporalTrendRef = useRef(null); // "improving" | "worsening" | null
  const repStateRef = useRef({ phase: "high", lastRepAt: 0 });
  const repCountRef = useRef(0);
  repCountRef.current = repCount;

  useEffect(() => {
    if (!isStreaming || !isModelReady) return;
    voiceCoachingRef.current = startVoiceCoaching(() => {}, {
      apiBase,
      ttsEnabled,
    });
    return () => {
      if (voiceCoachingRef.current) voiceCoachingRef.current.reset();
      voiceCoachingRef.current = null;
    };
  }, [isStreaming, isModelReady, apiBase, ttsEnabled]);

  useEffect(() => {
    if (activeTab !== "youtube" && voiceCoachingRef.current) {
      voiceCoachingRef.current.reset();
    }
  }, [activeTab]);

  useEffect(() => {
    if (!voiceCoachOn && voiceCoachingRef.current) {
      voiceCoachingRef.current.cancel();
    }
  }, [voiceCoachOn]);

  const handleStartWorkout = () => {
    workoutSamplesRef.current = [];
    workoutStartedAtRef.current = performance.now() / 1000;
    repStateRef.current = { phase: "high", lastRepAt: 0 };
    setWorkoutActive(true);
    setWorkoutSummary(null);
    setWorkoutSummaryError(null);
    setRepCount(0);
    const video = ytVideoRef.current;
    if (video) video.play().catch(() => {});
  };

  const handleEndWorkout = async () => {
    setWorkoutActive(false);
    const video = ytVideoRef.current;
    if (video) video.pause();
    const samples = workoutSamplesRef.current;
    const startedAt = workoutStartedAtRef.current;
    const durationSec = startedAt != null ? performance.now() / 1000 - startedAt : null;

    if (!samples.length) {
      setWorkoutSummary("No form data was recorded during this workout. Start the exercise and mirror the demo to get a summary next time.");
      return;
    }

    setWorkoutSummaryLoading(true);
    setWorkoutSummaryError(null);
    try {
      const res = await fetch(`${apiBase}/api/workout/summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exercise_name: referenceExerciseName || "",
          exercise_muscle: referenceExerciseMuscle || "",
          duration_sec: durationSec,
          reps: repCountRef.current || undefined,
          samples: samples.map((s) => ({
            video_t: s.video_t,
            score: s.score,
            limbScores: s.limbScores,
            feedback: s.feedback,
          })),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Summary request failed");
      const summary = data.summary || "Your workout was recorded.";
      setWorkoutSummary(summary);
      const updated = saveWorkoutToHistory({
        exercise: referenceExerciseName,
        muscle: referenceExerciseMuscle,
        summary,
        durationSec,
        reps: repCountRef.current,
      });
      setWorkoutHistory(updated);
    } catch (err) {
      setWorkoutSummaryError(err.message || "Could not generate summary");
      setWorkoutSummary(null);
    } finally {
      setWorkoutSummaryLoading(false);
    }
  };

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
      const score = compareScoreRef.current;
      const colors = scoreToColors(score);

      if (result.landmarks && result.landmarks.length > 0) {
        for (const landmarks of result.landmarks) {
          drawingUtils.drawLandmarks(landmarks, {
            radius: 3,
            color: colors.primary,
          });
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: colors.accent,
            lineWidth: 2,
          });
        }

        const liveLandmarks = result.landmarks[0].map((lm) => [lm.x, lm.y, lm.z]);
        const now = performance.now();
        if (now - lastCompareMsRef.current >= 1000 / 15) {
          lastCompareMsRef.current = now;
          const refs = activeTab === "youtube" ? referenceFramesRef.current : aiReferenceFramesRef.current;
          const refVideo = activeTab === "youtube" ? ytVideoRef.current : aiVideoRef.current;
          if (
            refs?.length &&
            refVideo &&
            Number.isFinite(refVideo.currentTime)
          ) {
            const videoT = refVideo.currentTime;
            const closest = refs.reduce((a, b) =>
              Math.abs(a.t - videoT) <= Math.abs(b.t - videoT) ? a : b
            );
            if (closest?.landmarks?.length) {
              const { score, limbScores, feedback } = comparePoseWithCoaching(
                { landmarks: closest.landmarks },
                { landmarks: liveLandmarks },
                { exerciseMuscle: referenceExerciseMuscle }
              );
              const score100 = Math.round(score * 100);
              compareScoreRef.current = score100;
              setComparisonScore(score100);
              setCoachingFeedback(feedback);

              const worstLimb = feedback?.joint ?? null;
              const worstScore = worstLimb ? (limbScores[worstLimb] ?? 1) : 1;
              const history = poseHistoryRef.current;
              history.push({ score, worstLimb, worstScore });
              if (history.length > 30) history.shift();
              if (history.length >= 10 && worstLimb) {
                const half = Math.floor(history.length / 2);
                const recent = history.slice(-half).filter((h) => h.worstLimb === worstLimb);
                const older = history.slice(0, half).filter((h) => h.worstLimb === worstLimb);
                if (recent.length >= 3 && older.length >= 3) {
                  const recentAvg = recent.reduce((a, h) => a + h.worstScore, 0) / recent.length;
                  const olderAvg = older.reduce((a, h) => a + h.worstScore, 0) / older.length;
                  const diff = recentAvg - olderAvg;
                  temporalTrendRef.current = diff > 0.03 ? "improving" : diff < -0.03 ? "worsening" : null;
                } else {
                  temporalTrendRef.current = null;
                }
              } else {
                temporalTrendRef.current = null;
              }
              const worstScoreForLimb = worstLimb ? (limbScores[worstLimb] ?? 1) : 1;
              const substantialError = score < 0.75 && worstScoreForLimb < 0.75;
              if (voiceCoachOnRef.current && voiceCoachingRef.current) {
                const willFetchLlm =
                  llmCoachingAvailableRef.current &&
                  feedback?.message &&
                  substantialError &&
                  now - lastLlmFetchMsRef.current >= 5000;
                voiceCoachingRef.current.onFrame(
                  { score, limbScores, feedback },
                  { deferToLlm: willFetchLlm }
                );
              }
              if (
                llmCoachingAvailableRef.current &&
                feedback?.message &&
                substantialError &&
                now - lastLlmFetchMsRef.current >= 5000
              ) {
                lastLlmFetchMsRef.current = now;
                fetch(`${apiBase}/api/coaching/llm`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    score,
                    limbScores,
                    feedback_message: feedback.message,
                    worst_limb: feedback.joint,
                    exercise_name: referenceExerciseName,
                    exercise_muscle: referenceExerciseMuscle,
                    temporal_trend: temporalTrendRef.current,
                  }),
                })
                  .then(async (r) => {
                    const d = await r.json();
                    if (d.message) {
                      setCoachingFeedback({ joint: feedback.joint, message: d.message });
                      if (r.ok && voiceCoachOnRef.current && voiceCoachingRef.current?.speak) {
                        voiceCoachingRef.current.speak(d.message);
                      }
                    }
                    return d;
                  })
                  .catch((e) => console.warn("[llm-coaching] fetch failed:", e));
              }
              if (now - lastFetchMsRef.current >= 500) {
                lastFetchMsRef.current = now;
                fetch(`${apiBase}/api/compare-pose`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ video_t: videoT, score, limbScores, feedback }),
                }).catch((e) => console.warn("[compare-pose] fetch failed:", e));
              }
              if (workoutActiveRef.current && workoutSamplesRef.current.length < 500) {
                workoutSamplesRef.current.push({
                  video_t: videoT,
                  score,
                  limbScores: limbScores ? { ...limbScores } : undefined,
                  feedback: feedback ? { joint: feedback.joint, message: feedback.message } : undefined,
                });
              }
              const rs = repStateRef.current;
              if (workoutActiveRef.current && rs) {
                const minRepGap = 0.8;
                if (rs.phase === "high" && score < 0.65) rs.phase = "dipped";
                else if (rs.phase === "dipped" && score > 0.82 && now / 1000 - rs.lastRepAt >= minRepGap) {
                  rs.phase = "high";
                  rs.lastRepAt = now / 1000;
                  setRepCount((c) => c + 1);
                } else if (rs.phase === "dipped" && score > 0.82) rs.phase = "high";
              }
            } else {
              compareScoreRef.current = null;
              setComparisonScore(null);
              setCoachingFeedback(null);
              poseHistoryRef.current = [];
              temporalTrendRef.current = null;
            }
            } else {
              compareScoreRef.current = null;
              setComparisonScore(null);
              setCoachingFeedback(null);
              poseHistoryRef.current = [];
              temporalTrendRef.current = null;
            }
        }
      }

      if (score != null && score >= 0) {
        const pad = 12;
        const fontPx = Math.min(48, canvas.height * 0.12);
        ctx.font = `bold ${fontPx}px system-ui, sans-serif`;
        const text = `${score}%`;
        const metrics = ctx.measureText(text);
        const w = metrics.width + pad * 2;
        const h = fontPx + pad * 2;
        const x = canvas.width - w - pad;
        const y = pad;
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = colors.primary;
        ctx.fillText(text, x + pad, y + h - pad - 4);
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [isStreaming, isModelReady, activeTab, apiBase, referenceExerciseMuscle, referenceExerciseName]);

  useEffect(() => {
    if (activeTab !== "youtube") {
      setCoachingFeedback(null);
      setWorkoutSummary(null);
      setWorkoutSummaryError(null);
    }
  }, [activeTab]);

  // Draw skeleton overlay on YouTube video from reference frames
  useEffect(() => {
    if (activeTab !== "youtube") return;

    let animationFrameId;

    const drawYtSkeleton = () => {
      const video = ytVideoRef.current;
      const canvas = ytSkeletonCanvasRef.current;
      const refs = referenceFramesRef.current;

      if (!video || !canvas || !refs?.length || !Number.isFinite(video.currentTime)) {
        animationFrameId = requestAnimationFrame(drawYtSkeleton);
        return;
      }

      const parent = canvas.parentElement;
      if (parent) {
        const w = parent.clientWidth;
        const h = parent.clientHeight;
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
        }
      }

      const ctx = canvas.getContext("2d");
      const videoT = video.currentTime;
      const closest = refs.reduce((a, b) =>
        Math.abs(a.t - videoT) <= Math.abs(b.t - videoT) ? a : b
      );

      if (!closest?.landmarks?.length) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        animationFrameId = requestAnimationFrame(drawYtSkeleton);
        return;
      }

      const cw = canvas.width;
      const ch = canvas.height;
      const vw = video.videoWidth || 1;
      const vh = video.videoHeight || 1;
      const scale = Math.min(cw / vw, ch / vh);
      const drawW = vw * scale;
      const drawH = vh * scale;
      const offsetX = (cw - drawW) / 2;
      const offsetY = (ch - drawH) / 2;

      const toScreen = (x, y) => ({
        x: offsetX + x * drawW,
        y: offsetY + y * drawH,
      });

      ctx.clearRect(0, 0, cw, ch);

      const pts = closest.landmarks.map((lm) => toScreen(lm[0], lm[1]));

      ctx.strokeStyle = "#ff3f81";
      ctx.fillStyle = "#3fb0ff";
      ctx.lineWidth = 2;

      for (const [i, j] of PoseLandmarker.POSE_CONNECTIONS) {
        if (pts[i] && pts[j]) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }
      }
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(drawYtSkeleton);
    };

    animationFrameId = requestAnimationFrame(drawYtSkeleton);
    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [activeTab]);

  const leftPanelTitle = activeTab === "youtube" ? "Exercise Demo" : "AI Generated Video";

  return (
    <div className="app">
      <header className="header">
        <h1>FormAI Pose Demo</h1>
        <p>
          {activeTab === "youtube"
            ? "Search exercises from the YMove database, then compare your form with the demo video."
            : "Describe a workout or exercise and generate an AI video to compare with your webcam."}
        </p>
      </header>

      <div className="tabs">
        <button
          type="button"
          className={`tab ${activeTab === "youtube" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("youtube")}
        >
          Exercise Workout
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
        <div className="input-row exercise-search-row">
          <label className="input-label" htmlFor="exercise-search">
            Search exercises or paste a YouTube link
          </label>
          <div className="exercise-search-wrap">
            <input
              id="exercise-search"
              type="text"
              className="youtube-input"
              placeholder="e.g. Bicycle Crunches, Push-Ups, Squats — or https://youtube.com/watch?v=..."
              value={exerciseSearchQuery}
              onChange={(e) => setExerciseSearchQuery(e.target.value)}
              onFocus={() => setExerciseSearchFocused(true)}
              onBlur={() => setTimeout(() => setExerciseSearchFocused(false), 150)}
              disabled={exerciseVideoLoading}
            />
            {exerciseVideoLoading && (
              <span className="exercise-search-loading">Loading video…</span>
            )}
          </div>
          {exerciseSearchFocused && exerciseResults.length > 0 && (
            <ul className="exercise-results">
              {exerciseResults.map((ex) => (
                <li key={`${ex.name}-${ex.muscle}`}>
                  <button
                    type="button"
                    className="exercise-result-btn"
                    onClick={() => handleSelectExercise(ex)}
                    disabled={exerciseVideoLoading}
                  >
                    <span className="exercise-name">{ex.name}</span>
                    <span className="exercise-meta">{ex.muscle} · {ex.level}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {exerciseSearchFocused && exerciseSearchQuery.trim() && exerciseSearchLoading && exerciseResults.length === 0 && (
            <div className="exercise-search-hint">Searching…</div>
          )}
          {exerciseSearchFocused && exerciseSearchQuery.trim() && !exerciseSearchLoading && exerciseResults.length === 0 && (
            <div className="exercise-search-empty">
              No exercises found for &quot;{exerciseSearchQuery}&quot;.
              {exerciseSearchSuggestions.length > 0 && (
                <p className="exercise-search-suggestions">
                  Try: {exerciseSearchSuggestions.map((s) => (
                    <button
                      key={s}
                      type="button"
                      className="suggestion-btn"
                      onClick={() => setExerciseSearchQuery(s)}
                    >
                      {s}
                    </button>
                  ))}
                </p>
              )}
            </div>
          )}
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
            {aiLoading ? `Generating… ${aiPercent}%` : "Generate Video"}
          </button>
          {aiLoading && (
            <div className="ai-progress">
              <div className="ai-progress-top">
                <span className="ai-progress-stage">{aiStage || "working"}</span>
                <span className="ai-progress-percent">{aiPercent}%</span>
              </div>
              <div className="ai-progress-bar">
                <div className="ai-progress-fill" style={{ width: `${aiPercent}%` }} />
              </div>
            </div>
          )}
          {aiError && <div className="ai-error">{aiError}</div>}
        </div>
      )}

      <main className="side-by-side">
        <section className="panel">
          <h2>{leftPanelTitle}</h2>
          <div className="youtube-video-wrap">
            {activeTab === "youtube" && ytVideoSrc ? (
              <>
                <div
                  className={`youtube-video-area ${isYoutubeVideo && !isPastedYoutubeShort ? "video-area-youtube" : "video-area-preset"}`}
                >
                  <video
                    key={referenceExerciseId || referenceVideoUrl}
                    ref={ytVideoRef}
                    src={ytVideoSrc}
                    className="youtube-native-video"
                    playsInline
                    loop
                    preload="auto"
                    crossOrigin="anonymous"
                    poster={youtubeVideoIdFromUrl ? `https://img.youtube.com/vi/${youtubeVideoIdFromUrl}/hqdefault.jpg` : undefined}
                    onPlay={() => setYtPlaying(true)}
                    onPause={() => setYtPlaying(false)}
                    onTimeUpdate={handleYtTimeUpdate}
                    onLoadedMetadata={(e) => setYtDuration(e.target.duration ?? 0)}
                    onEnded={() => setYtProgress(0)}
                    onLoadStart={() => { setYtLoading(true); setYtError(null); }}
                    onCanPlay={() => setYtLoading(false)}
                    onError={() => {
                      setYtLoading(false);
                      setYtError("Video failed to load. Is the backend running?");
                    }}
                  />
                  <canvas
                    ref={ytSkeletonCanvasRef}
                    className="youtube-skeleton-overlay"
                    aria-hidden="true"
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
              <div className="youtube-video-area video-area-preset">
                <div className="youtube-placeholder-top">
                  Search for an exercise or paste a YouTube link to load the demo video
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <h2>Your Pose (Webcam)</h2>
          <div className="video-container output">
            <video ref={videoRef} autoPlay playsInline muted className="video hidden-video" aria-hidden="true" />
            <canvas ref={outputCanvasRef} className="video" />
          </div>
          <div className="comparison-display">
            {comparisonScore != null && comparisonScore >= 0 ? (
              <div className="comparison-score-row">
                <span className="comparison-score" style={{ color: scoreToColors(comparisonScore).primary }}>
                  Match: {comparisonScore}%
                </span>
                {workoutActive && (
                  <span className="rep-count">Reps: {repCount}</span>
                )}
              </div>
            ) : activeTab === "youtube" && ytVideoSrc && (
              <div className={`comparison-hint ${preprocessError ? "error" : ""}`}>
                {preprocessLoading
                  ? "Extracting reference poses…"
                  : preprocessError
                    ? preprocessError
                    : "Play the video and mirror the pose to see your score."}
              </div>
            )}
            {activeTab === "ai" && aiVideoUrl && (
              <div className={`comparison-hint ${aiPreprocessError ? "error" : ""}`}>
                {aiPreprocessLoading
                  ? "Extracting reference poses…"
                  : aiPreprocessError
                    ? aiPreprocessError
                    : aiPreprocessReady
                      ? "Play the video and mirror the pose to see your score."
                      : "Extracting reference poses…"}
              </div>
            )}
            {coachingFeedback && (
              <div className="coaching-feedback">{coachingFeedback.message}</div>
            )}
            <label className="voice-coach-toggle">
              <input
                type="checkbox"
                checked={voiceCoachOn}
                onChange={(e) => setVoiceCoachOn(e.target.checked)}
                aria-label="Toggle voice coaching"
              />
              <span>Voice coach{supermemoryEnabled ? " (with form guides)" : ""}{ttsEnabled ? " · ElevenLabs" : ""}</span>
            </label>
            {activeTab === "youtube" && ytVideoSrc && (
              <div className="workout-actions">
                {!workoutActive ? (
                  <button type="button" className="btn-workout btn-start" onClick={handleStartWorkout}>
                    Start workout
                  </button>
                ) : (
                  <button type="button" className="btn-workout btn-end" onClick={handleEndWorkout} disabled={workoutSummaryLoading}>
                    {workoutSummaryLoading ? "Generating summary…" : "End workout"}
                  </button>
                )}
              </div>
            )}
          </div>
        </section>

        {(workoutSummary || workoutSummaryError || workoutSummaryLoading) && (
          <section className="panel workout-summary-panel">
            <h2>Workout summary</h2>
            {workoutSummaryLoading && <div className="workout-summary-loading">Generating your AI summary…</div>}
            {workoutSummary && !workoutSummaryLoading && (
              <>
                <div className="workout-summary-content">{workoutSummary}</div>
                {repCount > 0 && <div className="workout-reps">Reps completed: {repCount}</div>}
              </>
            )}
            {workoutSummaryError && !workoutSummaryLoading && (
              <div className="workout-summary-error">{workoutSummaryError}</div>
            )}
          </section>
        )}

        {workoutHistory.length > 0 && activeTab === "youtube" && (
          <section className="panel workout-history-panel">
            <h2>Workout history</h2>
            <ul className="workout-history-list">
              {workoutHistory.slice(0, 10).map((entry) => (
                <li key={entry.id} className="workout-history-item">
                  <span className="workout-history-date">{new Date(entry.date).toLocaleDateString()}</span>
                  <span className="workout-history-exercise">{entry.exercise}</span>
                  {entry.reps > 0 && <span className="workout-history-reps">{entry.reps} reps</span>}
                </li>
              ))}
            </ul>
            <button type="button" className="btn-clear-history" onClick={() => { clearWorkoutHistory(); setWorkoutHistory([]); }}>
              Clear history
            </button>
          </section>
        )}
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

      <Toast message={toast} onDismiss={() => setToast(null)} type="error" />

      <CoachPanel />
    </div>
  );
}

export default App;

