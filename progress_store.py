"""Simple JSON file store for workout progress. Used for charts and injury/decline detection."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROGRESS_FILE = Path(__file__).parent / "data" / "progress.json"
DEFAULT_USER = "default"


def _ensure_dir():
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load() -> dict:
    _ensure_dir()
    if not PROGRESS_FILE.exists():
        return {"users": {}}
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"users": {}}


def _save(data: dict) -> None:
    _ensure_dir()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_progress(
    user_id: str,
    *,
    date: str,
    exercise: str,
    muscle: str = "",
    avg_score: float,
    min_score: float,
    max_score: float,
    reps: int = 0,
    duration_sec: float = 0,
    worst_limbs: list[str] | None = None,
) -> None:
    data = _load()
    if "users" not in data:
        data["users"] = {}
    if user_id not in data["users"]:
        data["users"][user_id] = []
    entry = {
        "date": date,
        "exercise": exercise,
        "muscle": muscle,
        "avg_score": avg_score,
        "min_score": min_score,
        "max_score": max_score,
        "reps": reps,
        "duration_sec": duration_sec,
        "worst_limbs": worst_limbs or [],
    }
    data["users"][user_id].append(entry)
    data["users"][user_id].sort(key=lambda x: x["date"])
    _save(data)


def get_progress(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    data = _load()
    entries = data.get("users", {}).get(user_id, [])
    return entries[-limit:] if limit else entries


def get_progress_summary(user_id: str) -> dict[str, Any]:
    """Returns aggregated stats for charts and trend analysis."""
    entries = get_progress(user_id, limit=50)
    if not entries:
        return {"entries": [], "trend": "unknown", "alert": None}

    by_date: dict[str, list[dict]] = {}
    for e in entries:
        d = e["date"]
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(e)

    chart_data = []
    for d in sorted(by_date.keys()):
        items = by_date[d]
        chart_data.append({
            "date": d,
            "avg_score": sum(x["avg_score"] for x in items) / len(items),
            "reps": sum(x.get("reps", 0) for x in items),
            "sessions": len(items),
        })

    trend = "unknown"
    if len(chart_data) >= 3:
        recent = chart_data[-3:]
        older = chart_data[-6:-3] if len(chart_data) >= 6 else chart_data[:-3]
        if recent and older:
            recent_avg = sum(x["avg_score"] for x in recent) / len(recent)
            older_avg = sum(x["avg_score"] for x in older) / len(older)
            if recent_avg > older_avg + 0.05:
                trend = "improving"
            elif recent_avg < older_avg - 0.05:
                trend = "declining"

    alert = None
    if trend == "declining" and len(chart_data) >= 5:
        alert = "progress_declining"
    if chart_data and chart_data[-1]["avg_score"] < 0.5 and len(chart_data) >= 3:
        alert = "low_form_score"

    return {
        "entries": chart_data,
        "trend": trend,
        "alert": alert,
        "total_sessions": len(entries),
    }
