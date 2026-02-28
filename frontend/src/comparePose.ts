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

const LIMB_FEEDBACK_THRESHOLD = 0.85; // below this = limb is "off"

/**
 * Compare poses with limb-based feedback. Reports which limb is off (lowest similarity).
 * If landmarks empty: "Limbs not in frame".
 */
export function comparePoseWithCoaching(
  reference: PoseFrame,
  live: PoseFrame
): CoachingResult & CompareResult {
  const limbScores: Record<string, number> = {};

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

  const { score, limbScores: scores } = scorePoseSimilarity(reference, live);
  Object.assign(limbScores, scores);

  // Find limb with lowest score
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
          message: worstLimb.replace(/_/g, " ") + " is off",
        }
      : null;

  return { score, feedback: primaryIssue, limbScores };
}
