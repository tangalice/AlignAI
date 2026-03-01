import React from "react";

export function ExerciseSearch({
  query,
  onQueryChange,
  results,
  suggestions,
  loading,
  error,
  focused,
  onFocus,
  onBlur,
  onSelectExercise,
  videoLoading,
}) {
  return (
    <div className="input-row exercise-search-row">
      <label className="input-label" htmlFor="exercise-search">
        Search workout (YMove Exercise API)
      </label>
      <div className="exercise-search-wrap">
        <input
          id="exercise-search"
          type="text"
          className="youtube-input"
          placeholder="e.g. Bicycle Crunches, Push-Ups, Squats..."
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onFocus={onFocus}
          onBlur={onBlur}
          disabled={videoLoading}
        />
        {videoLoading && (
          <span className="exercise-search-loading">Loading video…</span>
        )}
      </div>
      {focused && results.length > 0 && (
        <ul className="exercise-results">
          {results.map((ex) => (
            <li key={`${ex.name}-${ex.muscle}`}>
              <button
                type="button"
                className="exercise-result-btn"
                onClick={() => onSelectExercise(ex)}
                disabled={videoLoading}
              >
                <span className="exercise-name">{ex.name}</span>
                <span className="exercise-meta">{ex.muscle} · {ex.level}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {focused && query.trim() && loading && results.length === 0 && (
        <div className="exercise-search-hint">Searching…</div>
      )}
      {error && <div className="exercise-search-error">{error}</div>}
      {focused && query.trim() && !loading && results.length === 0 && !error && (
        <div className="exercise-search-empty">
          No exercises found for &quot;{query}&quot;.
          {suggestions.length > 0 && (
            <p className="exercise-search-suggestions">
              Try: {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="suggestion-btn"
                  onClick={() => onQueryChange(s)}
                >
                  {s}
                </button>
              ))}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
