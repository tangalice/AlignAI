import React, { useState, useEffect } from "react";
import { loadProfile, saveProfile, getUserId } from "../hooks/useUserProfile";

export function SettingsModal({ open, onClose, apiBase, onProfileSaved }) {
  const [userEmail, setUserEmail] = useState("");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [gender, setGender] = useState("");
  const [ptEmail, setPtEmail] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (open) {
      const p = loadProfile();
      setUserEmail(p.userEmail);
      setHeight(p.height);
      setWeight(p.weight);
      setGender(p.gender);
      setPtEmail(p.ptEmail);
      setSaved(false);
    }
  }, [open]);

  const handleSave = async () => {
    const profile = {
      userEmail: userEmail.trim(),
      height: height.trim(),
      weight: weight.trim(),
      gender: gender.trim(),
      ptEmail: ptEmail.trim(),
    };
    saveProfile(profile);
    setSaving(true);
    setSaved(false);
    try {
      const userId = getUserId();
      const res = await fetch(`${apiBase}/api/user/profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          user_email: profile.userEmail,
          height: profile.height,
          weight: profile.weight,
          gender: profile.gender,
          pt_email: profile.ptEmail,
        }),
      });
      if (res.ok) {
        setSaved(true);
        onProfileSaved?.(profile);
      }
    } catch (e) {
      console.warn("[settings] sync to supermemory failed:", e);
      setSaved(true);
      onProfileSaved?.(profile);
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  return (
    <div className="settings-backdrop" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h3>Profile & Settings</h3>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="settings-body">
          <p className="settings-hint">Your profile helps personalize the PT report and coach. Data is stored locally and synced to your account.</p>
          <div className="settings-field">
            <label>Your Email</label>
            <input
              type="email"
              placeholder="you@example.com"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
            />
            <span className="settings-hint-small">Displayed in header when signed in</span>
          </div>
          <div className="settings-field">
            <label>Height</label>
            <input
              type="text"
              placeholder="e.g. 5'10&quot; or 178 cm"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
            />
          </div>
          <div className="settings-field">
            <label>Weight</label>
            <input
              type="text"
              placeholder="e.g. 160 lbs or 72 kg"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
            />
          </div>
          <div className="settings-field">
            <label>Gender</label>
            <select value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="">Select</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
              <option value="prefer_not">Prefer not to say</option>
            </select>
          </div>
          <div className="settings-field">
            <label>Physical Therapist Email</label>
            <input
              type="email"
              placeholder="pt@clinic.com"
              value={ptEmail}
              onChange={(e) => setPtEmail(e.target.value)}
            />
            <span className="settings-hint-small">Used when you click &quot;Send to PT&quot;</span>
          </div>
        </div>
        <div className="settings-footer">
          {saved && <span className="settings-saved">Saved</span>}
          <button type="button" className="settings-save-btn" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
