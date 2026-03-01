/**
 * Live webcam pose encoding at a fixed sample rate.
 * Uses MediaPipe PoseLandmarker (VIDEO mode) with GPU acceleration.
 * Samples at 8 FPS; does not process every animation frame.
 *
 * @param {HTMLVideoElement} videoElement - Webcam video (srcObject from getUserMedia)
 * @param {import("@mediapipe/tasks-vision").PoseLandmarker} poseLandmarker - PoseLandmarker in VIDEO mode
 * @param { (frameData: { index: number; t: number; ms: number; landmarks: [number, number, number][] }) => void } onFrame - Called at sample rate
 * @param {{ sampleFps?: number }} options - Optional: sampleFps (default 8)
 * @returns { () => void } stop - Call to stop encoding
 */
export function startLivePoseEncoding(videoElement, poseLandmarker, onFrame, options = {}) {
  const sampleFps = options.sampleFps ?? 8;
  const intervalMs = 1000 / sampleFps;

  let index = 0;
  let lastSampleMs = -Infinity;
  let startTimeMs = null;
  let rafId = null;
  let stopped = false;

  function tick() {
    if (stopped) return;

    rafId = requestAnimationFrame(tick);

    if (!videoElement || videoElement.readyState < 2) return;
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) return;

    const now = performance.now();
    if (now - lastSampleMs < intervalMs) return;

    lastSampleMs = now;
    if (startTimeMs == null) startTimeMs = now;
    const elapsedMs = now - startTimeMs;
    const t = elapsedMs / 1000;
    const ms = Math.round(elapsedMs);

    const result = poseLandmarker.detectForVideo(videoElement, now);

    const landmarks =
      result?.landmarks?.length > 0
        ? result.landmarks[0].map((lm) => [lm.x, lm.y, lm.z])
        : [];

    onFrame({
      index,
      t: Number(t.toFixed(3)),
      ms,
      landmarks,
    });

    index += 1;
  }

  rafId = requestAnimationFrame(tick);

  return function stop() {
    stopped = true;
    if (rafId != null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };
}
