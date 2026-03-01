const STORAGE_KEY = "formai_workout_history";
const MAX_ENTRIES = 50;

export function loadWorkoutHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.slice(0, MAX_ENTRIES) : [];
  } catch {
    return [];
  }
}

export function saveWorkoutToHistory(entry) {
  const history = loadWorkoutHistory();
  const newEntry = {
    id: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    date: new Date().toISOString(),
    exercise: entry.exercise || "Unknown",
    muscle: entry.muscle || "",
    summary: entry.summary || "",
    durationSec: entry.durationSec ?? null,
  };
  const updated = [newEntry, ...history].slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return updated;
  } catch {
    return history;
  }
}

export function clearWorkoutHistory() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    return [];
  } catch {
    return loadWorkoutHistory();
  }
}
