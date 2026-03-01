import React, { useEffect } from "react";

/**
 * Simple toast for transient error messages. Auto-dismisses after 4 seconds.
 */
export function Toast({ message, onDismiss, type = "error" }) {
  useEffect(() => {
    if (!message) return;
    const id = setTimeout(() => onDismiss?.(), 4000);
    return () => clearTimeout(id);
  }, [message, onDismiss]);

  if (!message) return null;

  return (
    <div className={`toast toast-${type}`} role="alert">
      <span>{message}</span>
      <button type="button" className="toast-dismiss" onClick={onDismiss} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
