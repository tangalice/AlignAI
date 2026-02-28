/**
 * Low-latency browser-only voice coaching engine.
 * Uses Web Speech API (SpeechSynthesis). Zero backend TTS.
 *
 * Auto-unlocks on first click/touch/keydown — no button needed.
 */

export type LimbGroup = "leftArm" | "rightArm" | "leftLeg" | "rightLeg" | "torso";

const LIMB_GROUPS: LimbGroup[] = ["leftArm", "rightArm", "leftLeg", "rightLeg", "torso"];

const LIMB_TO_RAW: Record<LimbGroup, string[]> = {
  leftArm: ["left_upper_arm", "left_forearm"],
  rightArm: ["right_upper_arm", "right_forearm"],
  leftLeg: ["left_thigh", "left_shin"],
  rightLeg: ["right_thigh", "right_shin"],
  torso: ["torso"],
};

export interface FrameScoreData {
  score: number;
  limbScores: Record<string, number>;
}

export interface VoiceCoachingOutput {
  score: number;
  worstLimb: LimbGroup | null;
  severity: number;
  spoken: boolean;
}

/** Only speak when error exceeds this. 0.30 = speak when limb score below 70%. Higher = less sensitive. */
const ERROR_THRESHOLD = 0.30;
const COOLDOWN_MS = 2500;

function mapToGroupScores(limbScores: Record<string, number>): Record<LimbGroup, number> {
  const out = {} as Record<LimbGroup, number>;
  for (const limb of LIMB_GROUPS) {
    const raw = LIMB_TO_RAW[limb];
    const scores = raw.map((r) => limbScores[r]).filter((s) => typeof s === "number");
    out[limb] = scores.length > 0 ? Math.min(...scores) : 1;
  }
  return out;
}

export function analyzeErrors(_score: number, limbScores: Record<string, number>): { limb: LimbGroup; severity: number } | null {
  const groups = mapToGroupScores(limbScores);
  let worstLimb: LimbGroup | null = null;
  let worstError = 0;
  for (const limb of LIMB_GROUPS) {
    const error = 1 - groups[limb];
    if (error > ERROR_THRESHOLD && error > worstError) {
      worstError = error;
      worstLimb = limb;
    }
  }
  if (!worstLimb) return null;
  return { limb: worstLimb, severity: worstError };
}

// ---------------------------------------------------------------------------
// Module-level auto-unlock for Chrome SpeechSynthesis
// ---------------------------------------------------------------------------

let _unlocked = false;
let _defaultVoice: SpeechSynthesisVoice | null = null;

function loadVoices(): void {
  if (typeof speechSynthesis === "undefined") return;
  const list = speechSynthesis.getVoices();
  if (list.length === 0) return;
  const en = list.filter((v) => v.lang.startsWith("en"));
  _defaultVoice = en.find((v) => v.localService) ?? en[0] ?? list[0] ?? null;
  console.log("[VoiceCoach] voices loaded:", list.length, "default:", _defaultVoice?.name ?? "none");
}

function onFirstInteraction(): void {
  if (_unlocked) return;
  if (typeof window === "undefined" || !window.speechSynthesis) return;

  const synth = window.speechSynthesis;
  synth.cancel();
  if (synth.paused) synth.resume();

  loadVoices();

  const u = new SpeechSynthesisUtterance(".");
  u.volume = 0.01;
  u.rate = 2;
  if (_defaultVoice) u.voice = _defaultVoice;
  synth.speak(u);

  _unlocked = true;
  console.log("[VoiceCoach] audio unlocked on user gesture");

  document.removeEventListener("click", onFirstInteraction, true);
  document.removeEventListener("touchstart", onFirstInteraction, true);
  document.removeEventListener("keydown", onFirstInteraction, true);
}

/**
 * Call once at app startup. Registers one-time listeners that auto-unlock
 * speech on first click/touch/keydown. No button needed.
 */
export function initVoiceCoachAutoUnlock(): void {
  if (typeof document === "undefined") return;

  if (typeof speechSynthesis !== "undefined") {
    loadVoices();
    speechSynthesis.addEventListener("voiceschanged", loadVoices, { once: true });
  }

  document.addEventListener("click", onFirstInteraction, true);
  document.addEventListener("touchstart", onFirstInteraction, true);
  document.addEventListener("keydown", onFirstInteraction, true);
  console.log("[VoiceCoach] auto-unlock listeners registered");
}

/** @deprecated Use initVoiceCoachAutoUnlock() instead. */
export function preloadVoices(): void {
  initVoiceCoachAutoUnlock();
}

// ---------------------------------------------------------------------------
// VoiceCoach
// ---------------------------------------------------------------------------

const MESSAGES: Record<LimbGroup, { soft: string; strong: string }> = {
  leftArm: { soft: "Adjust left arm.", strong: "Fix your left arm." },
  rightArm: { soft: "Adjust right arm.", strong: "Fix your right arm." },
  leftLeg: { soft: "Adjust left leg.", strong: "Fix your left leg." },
  rightLeg: { soft: "Adjust right leg.", strong: "Fix your right leg." },
  torso: { soft: "Adjust your torso.", strong: "Fix your torso." },
};

export class VoiceCoach {
  speak(message: string): void {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    if (!message.trim()) return;

    if (!_unlocked) {
      console.log("[VoiceCoach] speech blocked: not unlocked yet");
      return;
    }

    const synth = window.speechSynthesis;
    synth.cancel();

    const u = new SpeechSynthesisUtterance(message.trim());
    u.lang = "en-US";
    u.rate = 1.05;
    u.volume = 1;
    u.pitch = 1;
    if (_defaultVoice) u.voice = _defaultVoice;
    synth.speak(u);

    console.log("[VoiceCoach] speaking:", message.trim());
  }

  cancel(): void {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }

  getMessage(limb: LimbGroup, severity: number): string {
    const m = MESSAGES[limb];
    return severity > 0.5 ? m.strong : m.soft;
  }
}

// ---------------------------------------------------------------------------
// Engine — processes frames and fires coaching speech
// ---------------------------------------------------------------------------

export function createVoiceCoachingEngine() {
  const voice = new VoiceCoach();
  let lastSpokenAt = 0;

  function processFrame(data: FrameScoreData): VoiceCoachingOutput {
    const ts = performance.now();
    const error = analyzeErrors(data.score, data.limbScores);

    if (!error) {
      return { score: data.score, worstLimb: null, severity: 0, spoken: false };
    }

    const elapsed = ts - lastSpokenAt;
    if (elapsed < COOLDOWN_MS) {
      return { score: data.score, worstLimb: error.limb, severity: error.severity, spoken: false };
    }

    const msg = voice.getMessage(error.limb, error.severity);
    voice.speak(msg);
    lastSpokenAt = ts;

    return { score: data.score, worstLimb: error.limb, severity: error.severity, spoken: true };
  }

  return {
    processFrame,
    cancel: () => voice.cancel(),
    reset: () => { voice.cancel(); lastSpokenAt = 0; },
  };
}

export function startVoiceCoaching(onFrameScore: (data: FrameScoreData) => void) {
  const engine = createVoiceCoachingEngine();

  function onFrame(data: FrameScoreData): void {
    onFrameScore(data);
    engine.processFrame(data);
  }

  return {
    onFrame,
    processFrame: engine.processFrame,
    cancel: engine.cancel,
    reset: engine.reset,
  };
}
