import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { analyzeErrors, initVoiceCoachAutoUnlock } from "./voiceCoaching";

describe("analyzeErrors", () => {
  it("returns null when all limb scores are above threshold", () => {
    const limbScores = {
      left_upper_arm: 0.9,
      right_upper_arm: 0.9,
      left_forearm: 0.9,
      right_forearm: 0.9,
      left_thigh: 0.9,
      right_thigh: 0.9,
      left_shin: 0.9,
      right_shin: 0.9,
      torso: 0.9,
    };
    expect(analyzeErrors(0.9, limbScores)).toBeNull();
  });

  it("returns worst limb when one is below threshold", () => {
    const limbScores = {
      left_upper_arm: 0.5,
      right_upper_arm: 0.9,
      left_forearm: 0.9,
      right_forearm: 0.9,
      left_thigh: 0.9,
      right_thigh: 0.9,
      left_shin: 0.9,
      right_shin: 0.9,
      torso: 0.9,
    };
    const result = analyzeErrors(0.7, limbScores);
    expect(result).not.toBeNull();
    expect(result?.limb).toBe("leftArm");
    expect(result?.severity).toBeGreaterThan(0.35);
  });

  it("returns limb with highest error when multiple are bad", () => {
    const limbScores = {
      left_upper_arm: 0.9,
      right_upper_arm: 0.3,
      left_forearm: 0.9,
      right_forearm: 0.4,
      left_thigh: 0.9,
      right_thigh: 0.9,
      left_shin: 0.9,
      right_shin: 0.9,
      torso: 0.9,
    };
    const result = analyzeErrors(0.5, limbScores);
    expect(result).not.toBeNull();
    expect(result?.limb).toBe("rightArm");
    expect(result?.severity).toBeGreaterThan(0.5);
  });

  it("maps limb groups correctly (forearm -> arm)", () => {
    const limbScores = {
      left_upper_arm: 0.9,
      right_upper_arm: 0.9,
      left_forearm: 0.3,
      right_forearm: 0.9,
      left_thigh: 0.9,
      right_thigh: 0.9,
      left_shin: 0.9,
      right_shin: 0.9,
      torso: 0.9,
    };
    const result = analyzeErrors(0.6, limbScores);
    expect(result?.limb).toBe("leftArm");
  });
});

describe("initVoiceCoachAutoUnlock", () => {
  const addEventListenerSpy = vi.spyOn(document, "addEventListener");
  const removeEventListenerSpy = vi.spyOn(document, "removeEventListener");

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("registers click, touchstart, keydown listeners", () => {
    initVoiceCoachAutoUnlock();
    expect(addEventListenerSpy).toHaveBeenCalledWith("click", expect.any(Function), true);
    expect(addEventListenerSpy).toHaveBeenCalledWith("touchstart", expect.any(Function), true);
    expect(addEventListenerSpy).toHaveBeenCalledWith("keydown", expect.any(Function), true);
  });
});
