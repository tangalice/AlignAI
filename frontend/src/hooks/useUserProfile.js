const PROFILE_KEY = "alignai_profile";
const USER_ID_KEY = "alignai_user_id";

function generateUserId() {
  return crypto.randomUUID?.() || `user_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

export function getUserId() {
  try {
    let id = localStorage.getItem(USER_ID_KEY);
    if (!id) {
      id = generateUserId();
      localStorage.setItem(USER_ID_KEY, id);
    }
    return id;
  } catch {
    return "default";
  }
}

export function loadProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return defaultProfile();
    const p = JSON.parse(raw);
    return {
      height: p.height ?? "",
      weight: p.weight ?? "",
      gender: p.gender ?? "",
      ptEmail: p.ptEmail ?? "",
      userEmail: p.userEmail ?? "",
    };
  } catch {
    return defaultProfile();
  }
}

function defaultProfile() {
  return {
    height: "",
    weight: "",
    gender: "",
    ptEmail: "",
    userEmail: "",
  };
}

export function saveProfile(profile) {
  try {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    return true;
  } catch {
    return false;
  }
}
