/**
 * Pose comparison module for MediaPipe Pose (33 landmarks).
 * Translation- and scale-invariant comparison using limb vector cosine similarity.
 * Limb-based feedback reports which limb is off (lowest similarity).
 */

export interface PoseFrame {
  landmarks: [number, number, number][]; // [x, y, z] × 33 or []
}

export interface CompareResult {
  score: number; // 0–1
  limbScores: Record<string, number>;
}

export interface PrimaryIssue {
  joint: string;
  message: string;
}

export interface CoachingResult {
  score: number;
  feedback: PrimaryIssue | null;
}

// MediaPipe Pose landmark indices
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_ELBOW = 13;
const RIGHT_ELBOW = 14;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;
const LEFT_KNEE = 25;
const RIGHT_KNEE = 26;
const LEFT_ANKLE = 27;
const RIGHT_ANKLE = 28;

type Vec3 = [number, number, number];

function vecSub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vecLen(v: Vec3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function vecDot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalizePose(landmarks: Vec3[]): Vec3[] {
  const root: Vec3 = [
    (landmarks[LEFT_HIP][0] + landmarks[RIGHT_HIP][0]) / 2,
    (landmarks[LEFT_HIP][1] + landmarks[RIGHT_HIP][1]) / 2,
    (landmarks[LEFT_HIP][2] + landmarks[RIGHT_HIP][2]) / 2,
  ];
  const translated = landmarks.map((p) => vecSub(p, root));
  const ls = translated[LEFT_SHOULDER];
  const rs = translated[RIGHT_SHOULDER];
  const scale = vecLen([
    rs[0] - ls[0],
    rs[1] - ls[1],
    rs[2] - ls[2],
  ]);
  if (scale < 1e-8) return translated;
  return translated.map((p) => [p[0] / scale, p[1] / scale, p[2] / scale]);
}

function limbVector(landmarks: Vec3[], from: number, to: number): Vec3 {
  return vecSub(landmarks[to], landmarks[from]);
}

function cosineSimilarity(a: Vec3, b: Vec3): number {
  const la = vecLen(a);
  const lb = vecLen(b);
  if (la < 1e-8 || lb < 1e-8) return 0;
  const cos = vecDot(a, b) / (la * lb);
  return Math.max(-1, Math.min(1, cos));
}

const LIMBS: { name: string; from: number; to: number; weight: number }[] = [
  { name: "left_upper_arm", from: LEFT_SHOULDER, to: LEFT_ELBOW, weight: 1.2 },
  { name: "right_upper_arm", from: RIGHT_SHOULDER, to: RIGHT_ELBOW, weight: 1.2 },
  { name: "left_forearm", from: LEFT_ELBOW, to: LEFT_WRIST, weight: 1.2 },
  { name: "right_forearm", from: RIGHT_ELBOW, to: RIGHT_WRIST, weight: 1.2 },
  { name: "left_thigh", from: LEFT_HIP, to: LEFT_KNEE, weight: 1.5 },
  { name: "right_thigh", from: RIGHT_HIP, to: RIGHT_KNEE, weight: 1.5 },
  { name: "left_shin", from: LEFT_KNEE, to: LEFT_ANKLE, weight: 1.5 },
  { name: "right_shin", from: RIGHT_KNEE, to: RIGHT_ANKLE, weight: 1.5 },
  { name: "left_torso", from: LEFT_SHOULDER, to: LEFT_HIP, weight: 1.0 },
  { name: "right_torso", from: RIGHT_SHOULDER, to: RIGHT_HIP, weight: 1.0 },
];

/** Limb definitions for Jiggle Wiggle style: shoulder_mid→hip_mid for torso. */
const LIMBS_JW: { name: string; weight: number; getVec: (lm: Vec3[]) => Vec3 }[] = [
  { name: "left_upper_arm", weight: 1.2, getVec: (lm) => limbVector(lm, LEFT_SHOULDER, LEFT_ELBOW) },
  { name: "left_forearm", weight: 1.2, getVec: (lm) => limbVector(lm, LEFT_ELBOW, LEFT_WRIST) },
  { name: "right_upper_arm", weight: 1.2, getVec: (lm) => limbVector(lm, RIGHT_SHOULDER, RIGHT_ELBOW) },
  { name: "right_forearm", weight: 1.2, getVec: (lm) => limbVector(lm, RIGHT_ELBOW, RIGHT_WRIST) },
  { name: "left_thigh", weight: 1.5, getVec: (lm) => limbVector(lm, LEFT_HIP, LEFT_KNEE) },
  { name: "left_shin", weight: 1.5, getVec: (lm) => limbVector(lm, LEFT_KNEE, LEFT_ANKLE) },
  { name: "right_thigh", weight: 1.5, getVec: (lm) => limbVector(lm, RIGHT_HIP, RIGHT_KNEE) },
  { name: "right_shin", weight: 1.5, getVec: (lm) => limbVector(lm, RIGHT_KNEE, RIGHT_ANKLE) },
  {
    name: "torso",
    weight: 1.0,
    getVec: (lm) => {
      const shoulderMid: Vec3 = [
        (lm[LEFT_SHOULDER][0] + lm[RIGHT_SHOULDER][0]) / 2,
        (lm[LEFT_SHOULDER][1] + lm[RIGHT_SHOULDER][1]) / 2,
        (lm[LEFT_SHOULDER][2] + lm[RIGHT_SHOULDER][2]) / 2,
      ];
      const hipMid: Vec3 = [
        (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2,
        (lm[LEFT_HIP][1] + lm[RIGHT_HIP][1]) / 2,
        (lm[LEFT_HIP][2] + lm[RIGHT_HIP][2]) / 2,
      ];
      return vecSub(shoulderMid, hipMid); // shoulder_mid → hip_mid (upward)
    },
  },
];

export interface ScorePoseState {
  previousScore: number;
}

/**
 * Jiggle Wiggle style pose similarity scoring.
 * Limb vectors, cosine similarity per limb, weighted average, optional exponential smoothing.
 */
export function scorePoseSimilarity(
  reference: PoseFrame,
  live: PoseFrame,
  options?: { alpha?: number; state?: ScorePoseState }
): { score: number; limbScores: Record<string, number> } {
  const limbScores: Record<string, number> = {};

  if (
    !reference.landmarks?.length ||
    !live.landmarks?.length ||
    reference.landmarks.length < 33 ||
    live.landmarks.length < 33
  ) {
    return { score: 0, limbScores };
  }

  const refNorm = normalizePose(reference.landmarks as Vec3[]);
  const liveNorm = normalizePose(live.landmarks as Vec3[]);

  let weightedSum = 0;
  let totalWeight = 0;

  for (const limb of LIMBS_JW) {
    const vRef = limb.getVec(refNorm);
    const vLive = limb.getVec(liveNorm);
    const cos = cosineSimilarity(vRef, vLive);
    limbScores[limb.name] = (cos + 1) / 2; // map [-1,1] → [0,1]
    weightedSum += cos * limb.weight;
    totalWeight += limb.weight;
  }

  const weightedMean = totalWeight > 0 ? weightedSum / totalWeight : 0;
  let score = (weightedMean + 1) / 2; // map [-1,1] → [0,1]
  score = Math.pow(Math.max(0, score), SCORE_STRICTNESS);

  // Exponential smoothing
  const alpha = options?.alpha ?? 0.3;
  const state = options?.state;
  if (state != null && typeof state.previousScore === "number") {
    score = alpha * score + (1 - alpha) * state.previousScore;
    state.previousScore = score;
  } else if (state) {
    state.previousScore = score;
  }

  return { score, limbScores };
}

/**
 * Compare two pose frames. Returns score 0–1 (1 = identical) and per-limb scores.
 */
export function comparePose(reference: PoseFrame, live: PoseFrame): CompareResult {
  const limbScores: Record<string, number> = {};

  if (
    !reference.landmarks?.length ||
    !live.landmarks?.length ||
    reference.landmarks.length < 33 ||
    live.landmarks.length < 33
  ) {
    return { score: 0, limbScores };
  }

  const refNorm = normalizePose(reference.landmarks as Vec3[]);
  const liveNorm = normalizePose(live.landmarks as Vec3[]);

  let weightedSum = 0;
  let totalWeight = 0;

  for (const limb of LIMBS) {
    const vRef = limbVector(refNorm, limb.from, limb.to);
    const vLive = limbVector(liveNorm, limb.from, limb.to);
    const cos = cosineSimilarity(vRef, vLive);
    limbScores[limb.name] = (cos + 1) / 2; // per-limb 0–1
    weightedSum += cos * limb.weight;
    totalWeight += limb.weight;
  }

  const weightedCosine = totalWeight > 0 ? weightedSum / totalWeight : 0;
  const score = (weightedCosine + 1) / 2; // normalize final to 0–1

  return { score, limbScores };
}

const LIMB_FEEDBACK_THRESHOLD = 0.75; // below this = limb is "off" (was 0.85, relaxed)

/** Margin from frame edge (0–1) for visibility. Relaxed from 0.08 so limbs at edges still count. */
const FRAME_MARGIN = 0.03;

/** Score power curve. 1.0 = linear; lower = more forgiving. Was 1.4, now 1.1. */
const SCORE_STRICTNESS = 1.1;

/** Upper-body muscle groups: arms and torso matter most, legs minimal. */
const UPPER_MUSCLES = new Set([
  "chest", "back", "shoulders", "biceps", "triceps", "forearms", "core", "abs",
  "anterior deltoid", "posterior deltoids", "deltoids", "lats", "middle back",
  "lower back", "neck", "obliques", "transverse abdominis", "rotator cuff",
]);
/** Lower-body muscle groups: legs matter most, arms minimal. */
const LOWER_MUSCLES = new Set([
  "quadriceps", "hamstrings", "glutes", "calves", "abductors", "adductors",
  "hip flexors", "quads", "legs",
]);

/** Limb names that are legs. */
const LEG_LIMBS = new Set(["left_thigh", "right_thigh", "left_shin", "right_shin"]);
/** Limb names that are arms. */
const ARM_LIMBS = new Set([
  "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
]);
const TORSO_LIMBS = new Set(["left_torso", "right_torso", "torso"]);

/** Limb name -> [fromIdx, toIdx] for visibility check. */
const LIMB_INDICES: Record<string, [number, number]> = {
  left_upper_arm: [LEFT_SHOULDER, LEFT_ELBOW],
  right_upper_arm: [RIGHT_SHOULDER, RIGHT_ELBOW],
  left_forearm: [LEFT_ELBOW, LEFT_WRIST],
  right_forearm: [RIGHT_ELBOW, RIGHT_WRIST],
  left_thigh: [LEFT_HIP, LEFT_KNEE],
  right_thigh: [RIGHT_HIP, RIGHT_KNEE],
  left_shin: [LEFT_KNEE, LEFT_ANKLE],
  right_shin: [RIGHT_KNEE, RIGHT_ANKLE],
  left_torso: [LEFT_SHOULDER, LEFT_HIP],
  right_torso: [RIGHT_SHOULDER, RIGHT_HIP],
};

/** In-frame check for limb visibility. Uses relaxed margin so limbs at edges still count. */
function isPointInFrame(p: Vec3): boolean {
  const x = p[0], y = p[1];
  return x >= FRAME_MARGIN && x <= 1 - FRAME_MARGIN && y >= FRAME_MARGIN && y <= 1 - FRAME_MARGIN;
}

/** Relaxed in-frame check: x,y within expanded bounds (MediaPipe can go slightly outside 0–1). */
function isPointInFrameRelaxed(p: Vec3): boolean {
  const x = p[0], y = p[1];
  return x >= -FRAME_MARGIN && x <= 1 + FRAME_MARGIN && y >= -FRAME_MARGIN && y <= 1 + FRAME_MARGIN;
}

/**
 * Returns true if the user is sufficiently in frame. Exercise-aware:
 * - Upper body: only shoulders need to be visible (hips often cut off)
 * - Lower body: only hips need to be visible
 * - Full body / unknown: shoulders OR hips visible
 */
function isUserInFrame(landmarks: Vec3[], exerciseMuscle: string): boolean {
  const shoulderMid: Vec3 = [
    (landmarks[LEFT_SHOULDER][0] + landmarks[RIGHT_SHOULDER][0]) / 2,
    (landmarks[LEFT_SHOULDER][1] + landmarks[RIGHT_SHOULDER][1]) / 2,
    0,
  ];
  const hipMid: Vec3 = [
    (landmarks[LEFT_HIP][0] + landmarks[RIGHT_HIP][0]) / 2,
    (landmarks[LEFT_HIP][1] + landmarks[RIGHT_HIP][1]) / 2,
    0,
  ];
  const shouldersOk = isPointInFrameRelaxed(shoulderMid);
  const hipsOk = isPointInFrameRelaxed(hipMid);

  const m = exerciseMuscle.toLowerCase();
  if (UPPER_MUSCLES.has(m)) return shouldersOk;
  if (LOWER_MUSCLES.has(m)) return hipsOk;
  return shouldersOk || hipsOk;
}

/** Returns true if both endpoints of the limb are in frame. Legs often cut off in upper-body framing. */
function isLimbVisible(landmarks: Vec3[], limbName: string): boolean {
  if (limbName === "torso") {
    const hipMid: Vec3 = [
      (landmarks[LEFT_HIP][0] + landmarks[RIGHT_HIP][0]) / 2,
      (landmarks[LEFT_HIP][1] + landmarks[RIGHT_HIP][1]) / 2,
      (landmarks[LEFT_HIP][2] + landmarks[RIGHT_HIP][2]) / 2,
    ];
    const shoulderMid: Vec3 = [
      (landmarks[LEFT_SHOULDER][0] + landmarks[RIGHT_SHOULDER][0]) / 2,
      (landmarks[LEFT_SHOULDER][1] + landmarks[RIGHT_SHOULDER][1]) / 2,
      0,
    ];
    return isPointInFrame(hipMid) && isPointInFrame(shoulderMid);
  }
  const idx = LIMB_INDICES[limbName];
  if (!idx) return true;
  return isPointInFrame(landmarks[idx[0]]) && isPointInFrame(landmarks[idx[1]]);
}

/**
 * Returns the weight for a limb given the exercise. 0 = ignore, 1 = normal, 2 = primary focus.
 * For biceps/triceps: arms = 2, torso = 0.3, legs = 0.
 * For squats: legs = 2, torso = 0.5, arms = 0.
 * For chest/back: arms + torso matter, legs = 0.
 */
function getLimbWeightForExercise(limbName: string, exerciseMuscle: string): number {
  if (!exerciseMuscle) return 1;
  const m = exerciseMuscle.toLowerCase();

  if (m === "full_body" || m === "core") return 1;

  if (UPPER_MUSCLES.has(m)) {
    if (ARM_LIMBS.has(limbName)) return 2.0;
    if (TORSO_LIMBS.has(limbName)) return 0.5;
    if (LEG_LIMBS.has(limbName)) return 0;
    return 0.5;
  }
  if (LOWER_MUSCLES.has(m)) {
    if (LEG_LIMBS.has(limbName)) return 2.0;
    if (TORSO_LIMBS.has(limbName)) return 0.5;
    if (ARM_LIMBS.has(limbName)) return 0;
    return 0.5;
  }
  return 1;
}

function isLimbRelevantForExercise(limbName: string, exerciseMuscle: string): boolean {
  return getLimbWeightForExercise(limbName, exerciseMuscle) > 0;
}

const FEEDBACK_MESSAGES: Record<string, string[]> = {
  left_upper_arm: ["Left arm angle is off — match the demo", "Adjust your left arm position", "Bring your left arm in line with the reference"],
  right_upper_arm: ["Right arm angle is off — match the demo", "Adjust your right arm position", "Bring your right arm in line with the reference"],
  left_forearm: ["Left forearm needs adjustment", "Align your left forearm with the demo", "Check your left arm position"],
  right_forearm: ["Right forearm needs adjustment", "Align your right forearm with the demo", "Check your right arm position"],
  left_thigh: ["Left leg position is off", "Adjust your left leg to match the demo", "Check your left knee alignment"],
  right_thigh: ["Right leg position is off", "Adjust your right leg to match the demo", "Check your right knee alignment"],
  left_shin: ["Left lower leg — align with the demo", "Adjust your left leg angle"],
  right_shin: ["Right lower leg — align with the demo", "Adjust your right leg angle"],
  left_torso: ["Torso angle needs adjustment", "Straighten or angle your torso to match"],
  right_torso: ["Torso alignment is off", "Adjust your torso to match the reference"],
  torso: ["Torso position is off", "Adjust your posture to match the demo", "Straighten your torso"],
};

function getFeedbackMessage(limbName: string, score: number): string {
  const messages = FEEDBACK_MESSAGES[limbName] ?? [limbName.replace(/_/g, " ") + " is off"];
  const idx = Math.min(Math.floor((1 - score) * messages.length), messages.length - 1);
  return messages[idx];
}

export interface CompareCoachingOptions {
  exerciseMuscle?: string;
}

/**
 * Compare poses with limb-based feedback. Reports which limb is off (lowest similarity).
 * Only considers limbs that are: (1) in frame, (2) relevant for the exercise (e.g. no legs for upper-body).
 * If landmarks empty: "Limbs not in frame".
 */
export function comparePoseWithCoaching(
  reference: PoseFrame,
  live: PoseFrame,
  options?: CompareCoachingOptions
): CoachingResult & CompareResult {
  const limbScores: Record<string, number> = {};
  const exerciseMuscle = (options?.exerciseMuscle || "").trim();

  if (
    !reference.landmarks?.length ||
    !live.landmarks?.length ||
    reference.landmarks.length < 33 ||
    live.landmarks.length < 33
  ) {
    return {
      score: 0,
      feedback: { joint: "frame", message: "Limbs not in frame" },
      limbScores,
    };
  }

  const liveNorm = live.landmarks as Vec3[];

  if (!isUserInFrame(liveNorm, exerciseMuscle)) {
    return {
      score: 0,
      feedback: { joint: "frame", message: "Step into frame so your upper body is visible" },
      limbScores: {},
    };
  }

  const { limbScores: scores } = scorePoseSimilarity(reference, live);

  let totalWeight = 0;
  let weightedSum = 0;
  for (const limb of LIMBS_JW) {
    const limbWeight = getLimbWeightForExercise(limb.name, exerciseMuscle);
    if (limbWeight === 0) {
      limbScores[limb.name] = 1;
      continue;
    }

    const rawScore = scores[limb.name] ?? 1;
    const visible = isLimbVisible(liveNorm, limb.name);
    const strictScore = Math.pow(Math.max(0, rawScore), SCORE_STRICTNESS);
    const final = visible ? strictScore : Math.min(1, Math.max(0.5, rawScore) * 0.9);
    limbScores[limb.name] = final;
    totalWeight += limbWeight;
    weightedSum += final * limbWeight;
  }
  const score = totalWeight > 0 ? weightedSum / totalWeight : 0;

  let worstLimb: string | null = null;
  let worstScore = 1;
  for (const [limb, s] of Object.entries(limbScores)) {
    if (s < worstScore && s < LIMB_FEEDBACK_THRESHOLD) {
      worstScore = s;
      worstLimb = limb;
    }
  }

  const primaryIssue: PrimaryIssue | null =
    worstLimb != null
      ? {
          joint: worstLimb,
          message: getFeedbackMessage(worstLimb, worstScore),
        }
      : null;

  return { score, feedback: primaryIssue, limbScores };
}
