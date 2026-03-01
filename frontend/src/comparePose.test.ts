import { describe, it, expect } from "vitest";
import { comparePose, comparePoseWithCoaching, scorePoseSimilarity } from "./comparePose";

// MediaPipe Pose landmark indices
const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
const LEFT_ELBOW = 13, RIGHT_ELBOW = 14;
const LEFT_WRIST = 15, RIGHT_WRIST = 16;
const LEFT_HIP = 23, RIGHT_HIP = 24;
const LEFT_KNEE = 25, RIGHT_KNEE = 26;
const LEFT_ANKLE = 27, RIGHT_ANKLE = 28;

/** Create a minimal 33-landmark pose. Indices 11-16, 23-28 are used; rest are placeholders. */
function makePose(overrides) {
  const pts = Array(33).fill(null).map((_, i) => [0.5, 0.5, 0]);
  // Default: standing pose with hips/shoulders in frame (0.2-0.8)
  pts[LEFT_SHOULDER] = [0.35, 0.3, 0];
  pts[RIGHT_SHOULDER] = [0.65, 0.3, 0];
  pts[LEFT_ELBOW] = [0.25, 0.4, 0];
  pts[RIGHT_ELBOW] = [0.75, 0.4, 0];
  pts[LEFT_WRIST] = [0.2, 0.5, 0];
  pts[RIGHT_WRIST] = [0.8, 0.5, 0];
  pts[LEFT_HIP] = [0.4, 0.6, 0];
  pts[RIGHT_HIP] = [0.6, 0.6, 0];
  pts[LEFT_KNEE] = [0.4, 0.8, 0];
  pts[RIGHT_KNEE] = [0.6, 0.8, 0];
  pts[LEFT_ANKLE] = [0.4, 1.0, 0];
  pts[RIGHT_ANKLE] = [0.6, 1.0, 0];
  if (overrides && typeof overrides === "object") {
    for (const [k, v] of Object.entries(overrides)) pts[Number(k)] = v;
  }
  return pts;
}

describe("comparePose", () => {
  it("returns score 1 for identical poses", () => {
    const pose = makePose();
    const result = comparePose({ landmarks: pose }, { landmarks: pose });
    expect(result.score).toBeGreaterThanOrEqual(0.99);
    expect(result.score).toBeLessThanOrEqual(1);
  });

  it("returns score 0 for empty reference", () => {
    const live = makePose();
    const result = comparePose({ landmarks: [] }, { landmarks: live });
    expect(result.score).toBe(0);
    expect(Object.keys(result.limbScores)).toHaveLength(0);
  });

  it("returns score 0 for empty live", () => {
    const ref = makePose();
    const result = comparePose({ landmarks: ref }, { landmarks: [] });
    expect(result.score).toBe(0);
  });

  it("returns lower score for different poses", () => {
    const ref = makePose();
    const live = makePose({ [LEFT_ELBOW]: [0.1, 0.2, 0], [RIGHT_ELBOW]: [0.9, 0.2, 0] });
    const result = comparePose({ landmarks: ref }, { landmarks: live });
    expect(result.score).toBeLessThan(1);
    expect(result.score).toBeGreaterThan(0);
  });

  it("returns limb scores for each limb", () => {
    const pose = makePose();
    const result = comparePose({ landmarks: pose }, { landmarks: pose });
    expect(result.limbScores).toHaveProperty("left_upper_arm");
    expect(result.limbScores).toHaveProperty("right_forearm");
    expect(result.limbScores).toHaveProperty("left_thigh");
    expect(result.limbScores).toHaveProperty("left_torso");
  });
});

describe("comparePoseWithCoaching", () => {
  it("returns 0 and 'Limbs not in frame' for empty landmarks", () => {
    const result = comparePoseWithCoaching(
      { landmarks: [] },
      { landmarks: makePose() }
    );
    expect(result.score).toBe(0);
    expect(result.feedback?.joint).toBe("frame");
    expect(result.feedback?.message).toContain("frame");
  });

  it("returns 0 when user is out of frame (lower-body exercise, hips off-screen)", () => {
    const outOfFrame = makePose({
      [LEFT_HIP]: [-0.3, 0.6, 0],
      [RIGHT_HIP]: [-0.2, 0.6, 0],
      [LEFT_SHOULDER]: [-0.3, 0.3, 0],
      [RIGHT_SHOULDER]: [-0.2, 0.3, 0],
    });
    const ref = makePose();
    const result = comparePoseWithCoaching(
      { landmarks: ref },
      { landmarks: outOfFrame },
      { exerciseMuscle: "quads" }
    );
    expect(result.score).toBe(0);
    expect(result.feedback?.message).toContain("frame");
  });

  it("returns high score for matching poses", () => {
    const pose = makePose();
    const result = comparePoseWithCoaching(
      { landmarks: pose },
      { landmarks: pose }
    );
    // Score is reduced by power curve and limb visibility; matching poses should score well
    expect(result.score).toBeGreaterThan(0.6);
    expect(result.limbScores).toBeDefined();
  });

  it("returns feedback when a limb is off", () => {
    const ref = makePose();
    const live = makePose({
      [LEFT_ELBOW]: [0.05, 0.1, 0],
      [LEFT_WRIST]: [0.0, 0.2, 0],
    });
    const result = comparePoseWithCoaching(
      { landmarks: ref },
      { landmarks: live }
    );
    expect(result.score).toBeLessThan(0.98);
    expect(result.feedback).not.toBeNull();
    expect(result.feedback?.joint).toBeDefined();
    expect(result.feedback?.message).toBeDefined();
  });
});

describe("scorePoseSimilarity", () => {
  it("supports optional state for smoothing", () => {
    const pose = makePose();
    const state = { previousScore: 0.8 };
    const result = scorePoseSimilarity(
      { landmarks: pose },
      { landmarks: pose },
      { alpha: 0.3, state }
    );
    expect(result.score).toBeGreaterThan(0);
    expect(state.previousScore).toBe(result.score);
  });
});
