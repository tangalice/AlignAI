/**
 * Voice coaching engine. Uses ElevenLabs TTS (backend) when enabled,
 * otherwise falls back to Web Speech API (SpeechSynthesis).
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

export interface PrimaryFeedback {
  joint: string;
  message: string;
}

export interface FrameScoreData {
  score: number;
  limbScores: Record<string, number>;
  /** From comparePose – single source of truth for what to coach. When set, voice uses this instead of analyzeErrors. */
  feedback?: PrimaryFeedback | null;
}

export interface VoiceCoachingOutput {
  score: number;
  worstLimb: LimbGroup | null;
  severity: number;
  spoken: boolean;
}

/** Only speak when error exceeds this. 0.35 = speak when limb score below 65%. Higher = less sensitive. */
const ERROR_THRESHOLD = 0.35;
const COOLDOWN_MS = 3500; // ~3 per 10 seconds max

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

const MESSAGES: Record<LimbGroup, { soft: string[]; strong: string[] }> = {
  leftArm: {
    soft: ["Left arm — match the demo.", "Adjust your left arm position.", "Bring your left arm in line."],
    strong: ["Fix your left arm — it's off.", "Your left arm needs correction.", "Align your left arm with the reference."],
  },
  rightArm: {
    soft: ["Right arm — match the demo.", "Adjust your right arm position.", "Bring your right arm in line."],
    strong: ["Fix your right arm — it's off.", "Your right arm needs correction.", "Align your right arm with the reference."],
  },
  leftLeg: {
    soft: ["Left leg — align with the demo.", "Adjust your left leg position.", "Check your left leg angle."],
    strong: ["Fix your left leg — it's off.", "Your left leg needs correction.", "Align your left leg with the reference."],
  },
  rightLeg: {
    soft: ["Right leg — align with the demo.", "Adjust your right leg position.", "Check your right leg angle."],
    strong: ["Fix your right leg — it's off.", "Your right leg needs correction.", "Align your right leg with the reference."],
  },
  torso: {
    soft: ["Torso — adjust to match.", "Straighten your posture.", "Align your torso with the demo."],
    strong: ["Fix your torso position.", "Your posture is off.", "Adjust your torso to match the reference."],
  },
};

export interface VoiceCoachConfig {
  apiBase?: string;
  ttsEnabled?: boolean;
}

export class VoiceCoach {
  private config: VoiceCoachConfig = {};
  private currentAudio: HTMLAudioElement | null = null;
  /** Queue of messages to speak; next starts only after current finishes (no overlap). */
  private speechQueue: string[] = [];
  private busy = false;
  private _requestId = 0;

  constructor(config?: VoiceCoachConfig) {
    this.config = config || {};
  }

  private onSpeechDone(): void {
    this.busy = false;
    this.currentAudio = null;
    if (this.speechQueue.length > 0) {
      const next = this.speechQueue.shift()!;
      this.doSpeak(next);
    }
  }

  speak(message: string): void {
    if (typeof window === "undefined") return;
    const trimmed = message.trim();
    if (!trimmed) return;

    if (!_unlocked) {
      console.log("[VoiceCoach] speech blocked: not unlocked yet");
      return;
    }

    if (this.busy) {
      this.speechQueue.push(trimmed);
      return;
    }
    this.doSpeak(trimmed);
  }

  private doSpeak(text: string): void {
    this.busy = true;

    if (this.config.ttsEnabled) {
      this.speakViaElevenLabs(text);
      return;
    }

    if (!window.speechSynthesis) {
      this.busy = false;
      return;
    }
    const synth = window.speechSynthesis;
    synth.cancel();

    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.rate = 1.05;
    u.volume = 1;
    u.pitch = 1;
    if (_defaultVoice) u.voice = _defaultVoice;
    u.onend = () => this.onSpeechDone();
    u.onerror = () => this.onSpeechDone();
    synth.speak(u);
    console.log("[VoiceCoach] speaking (browser):", text);
  }

  private speakViaElevenLabs(text: string): void {
    const base = (this.config.apiBase || "").replace(/\/$/, "");
    const url = base ? `${base}/api/tts` : "/api/tts";
    const requestId = ++this._requestId;

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`TTS ${r.status}`);
        return r.blob();
      })
      .then((blob) => {
        if (requestId !== this._requestId) {
          this.onSpeechDone();
          return;
        }
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        this.currentAudio = audio;
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          if (this.currentAudio === audio) this.currentAudio = null;
          this.onSpeechDone();
        };
        audio.onerror = (e) => {
          console.warn("[VoiceCoach] ElevenLabs play failed:", e);
          URL.revokeObjectURL(audioUrl);
          if (this.currentAudio === audio) this.currentAudio = null;
          this.onSpeechDone();
        };
        audio.play();
        console.log("[VoiceCoach] speaking (ElevenLabs):", text.slice(0, 50) + (text.length > 50 ? "…" : ""));
      })
      .catch((e) => {
        console.warn("[VoiceCoach] ElevenLabs fetch failed:", e);
        if (requestId !== this._requestId) {
          this.onSpeechDone();
          return;
        }
        if (typeof window !== "undefined" && window.speechSynthesis) {
          const u = new SpeechSynthesisUtterance(text);
          u.lang = "en-US";
          if (_defaultVoice) u.voice = _defaultVoice;
          u.onend = () => this.onSpeechDone();
          u.onerror = () => this.onSpeechDone();
          window.speechSynthesis.speak(u);
          console.log("[VoiceCoach] fallback to browser speech");
        } else {
          this.onSpeechDone();
        }
      });
  }

  cancel(): void {
    this._requestId++;
    this.speechQueue = [];
    this.busy = false;
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }

  getMessage(limb: LimbGroup, severity: number): string {
    const m = MESSAGES[limb];
    const arr = severity > 0.5 ? m.strong : m.soft;
    const idx = Math.floor(Math.random() * arr.length);
    return arr[idx];
  }
}

// Only intervene when form difference is VERY different (score < 0.55)
const COACH_SCORE_THRESHOLD = 0.55;

// ---------------------------------------------------------------------------
// Engine — processes frames and fires coaching speech
// Uses comparePose feedback as single source of truth when provided.
// ---------------------------------------------------------------------------

export function createVoiceCoachingEngine(config?: VoiceCoachConfig) {
  const voice = new VoiceCoach(config);
  let lastSpokenAt = 0;

  function processFrame(data: FrameScoreData, options?: { deferToLlm?: boolean }): VoiceCoachingOutput {
    const ts = performance.now();
    const { score, limbScores, feedback } = data;

    // Unified checkpoint: only coach when feedback exists and score is below threshold
    const shouldCoach = feedback?.message && score < COACH_SCORE_THRESHOLD;
    if (!shouldCoach) {
      return { score, worstLimb: null, severity: 0, spoken: false };
    }

    // When AI coach will provide the message, don't speak here – caller will invoke speak() when LLM returns
    if (options?.deferToLlm) {
      return { score, worstLimb: "torso", severity: 1 - score, spoken: false };
    }

    const elapsed = ts - lastSpokenAt;
    if (elapsed < COOLDOWN_MS) {
      return { score, worstLimb: "torso", severity: 1 - score, spoken: false };
    }

    voice.speak(feedback.message);
    lastSpokenAt = ts;
    return { score, worstLimb: "torso", severity: 1 - score, spoken: true };
  }

  return {
    processFrame,
    speak: (msg: string) => {
      voice.speak(msg);
    },
    cancel: () => voice.cancel(),
    reset: () => {
      voice.cancel();
      lastSpokenAt = 0;
    },
  };
}

export function startVoiceCoaching(
  onFrameScore: (data: FrameScoreData) => void,
  config?: VoiceCoachConfig
) {
  const engine = createVoiceCoachingEngine(config);

  function onFrame(data: FrameScoreData, options?: { deferToLlm?: boolean }): void {
    onFrameScore(data);
    engine.processFrame(data, options);
  }

  return {
    onFrame,
    speak: engine.speak,
    cancel: engine.cancel,
    reset: engine.reset,
  };
}
