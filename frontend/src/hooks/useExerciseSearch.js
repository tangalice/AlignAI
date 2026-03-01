import { useEffect, useRef, useState } from "react";

const CLIENT_CACHE_MAX = 25;
const DEBOUNCE_MS = 60;

/**
 * Debounced exercise search via YMove API.
 * Uses client-side cache for instant results when typing (e.g. "cur" shows "curl" results).
 * Returns { results, suggestions, loading, error }.
 */
export function useExerciseSearch(apiBase, query, activeTab) {
  const [results, setResults] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const timeoutRef = useRef(null);
  const abortRef = useRef(null);
  const clientCacheRef = useRef(new Map()); // query -> { exercises, suggestions }

  useEffect(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (activeTab !== "youtube") return;

    const q = query.trim();
    const cache = clientCacheRef.current;

    const applyFromCache = (cached) => {
      if (cached) {
        setResults(cached.exercises || []);
        setSuggestions(cached.suggestions || []);
        setError(null);
      }
    };

    const doSearch = async () => {
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;

      // Instant: show cached results if we have a prefix match (user typing "cur" → show "curl" cache)
      if (q.length >= 2) {
        let best = null;
        for (const [cachedQ, cached] of cache) {
          if (cachedQ.toLowerCase().startsWith(q.toLowerCase()) && cachedQ.length > q.length) {
            if (!best || cachedQ.length < best.length) best = cachedQ;
          }
        }
        if (best) applyFromCache(cache.get(best));
      }

      setLoading(true);
      if (!q) setSuggestions([]);

      try {
        const res = await fetch(
          `${apiBase}/api/exercises/search?q=${encodeURIComponent(q)}&limit=15`,
          { signal }
        );
        const data = await res.json();
        if (!res.ok) {
          setResults([]);
          setSuggestions([]);
          setError(data.detail || data.error || `Search failed (${res.status})`);
          return;
        }
        const payload = { exercises: data.exercises || [], suggestions: data.suggestions || [] };
        setResults(payload.exercises);
        setSuggestions(payload.suggestions);
        setError(null);

        if (cache.size >= CLIENT_CACHE_MAX) {
          const firstKey = cache.keys().next().value;
          cache.delete(firstKey);
        }
        cache.set(q, payload);
      } catch (err) {
        if (err.name === "AbortError") return;
        setResults([]);
        setSuggestions([]);
        setError("Search failed. Check your connection and try again.");
      } finally {
        if (!signal.aborted) setLoading(false);
      }
    };

    if (!q) {
      doSearch();
      return;
    }
    timeoutRef.current = setTimeout(doSearch, DEBOUNCE_MS);
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [query, activeTab, apiBase]);

  return { results, suggestions, loading, error };
}
