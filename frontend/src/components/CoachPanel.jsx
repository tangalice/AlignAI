import React, { useState, useRef, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export function CoachPanel({ apiBase: apiBaseProp }) {
  const apiBase = apiBaseProp ?? import.meta.env.VITE_API_BASE ?? "";
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState("chat"); // "chat" | "progress" | "report"
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! I'm your AlignAI coach. I know your workout history and can help with form, progress, or anything fitness-related. What's on your mind?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [coachAvailable, setCoachAvailable] = useState(true);
  const [progress, setProgress] = useState({ entries: [], trend: "unknown", alert: null });
  const [ptReport, setPtReport] = useState(null);
  const [ptLoading, setPtLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => { scrollToBottom(); }, [messages]);

  useEffect(() => {
    if (open) {
      fetch(`${apiBase}/api/coach/status`)
        .then((r) => r.json())
        .then((d) => setCoachAvailable(d.available !== false))
        .catch(() => setCoachAvailable(false));
    }
  }, [open, apiBase]);

  useEffect(() => {
    if (open) {
      fetch(`${apiBase}/api/progress?user_id=default`)
        .then((r) => r.json())
        .then(setProgress)
        .catch(() => setProgress({ entries: [], trend: "unknown", alert: null }));
    }
  }, [open, apiBase]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    try {
      const res = await fetch(`${apiBase}/api/coach/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, user_id: "default" }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      let data = {};
      try {
        data = await res.json();
      } catch {
        data = { message: "Invalid response from server." };
      }
      const msg = data.message || data.error || data.detail || "Sorry, I couldn't respond.";
      setMessages((m) => [...m, { role: "assistant", content: msg }]);
    } catch (err) {
      clearTimeout(timeoutId);
      const isTimeout = err?.name === "AbortError";
      setMessages((m) => [
        ...m,
        { role: "assistant", content: isTimeout ? "Request timed out. Please try again." : "Connection error. Check that the server is running and try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const requestPTReport = async () => {
    if (ptLoading) return;
    setPtLoading(true);
    try {
      const res = await fetch(`${apiBase}/api/coach/pt-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: progress.alert || "user_requested", user_id: "default" }),
      });
      const data = await res.json();
      setPtReport(data.report || "");
      setTab("report");
    } catch {
      setPtReport("Failed to generate report.");
      setTab("report");
    } finally {
      setPtLoading(false);
    }
  };

  const chartData = (progress.entries || []).map((e) => ({
    date: e.date,
    score: Math.round((e.avg_score || 0) * 100),
    reps: e.reps || 0,
  }));

  return (
    <>
      <button
        type="button"
        className="coach-fab"
        onClick={() => setOpen(!open)}
        aria-label={open ? "Close coach" : "Open AI coach"}
      >
        {open ? "×" : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
      )}
      </button>

      {open && (
        <div className="coach-panel">
          <div className="coach-panel-header">
            <h3>AlignAI Coach</h3>
            <button type="button" className="coach-close" onClick={() => setOpen(false)} aria-label="Close">×</button>
          </div>

          <div className="coach-tabs">
            <button type="button" className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
            <button type="button" className={tab === "progress" ? "active" : ""} onClick={() => setTab("progress")}>Progress</button>
            <button
              type="button"
              className={`${tab === "report" ? "active" : ""} ${progress.alert ? "alert" : ""}`}
              onClick={() => setTab("report")}
            >
              PT Report
            </button>
          </div>

          {tab === "chat" && (
            <div className="coach-chat">
              {!coachAvailable && (
                <div className="coach-unavailable">
                  Coach is unavailable. Set OPENAI_API_KEY in the server .env and restart the server.
                </div>
              )}
              <div className="coach-messages">
                {messages.map((m, i) => (
                  <div key={i} className={`coach-msg ${m.role}`}>
                    <span className="coach-msg-content">{m.content}</span>
                  </div>
                ))}
                {loading && <div className="coach-msg assistant"><span className="coach-msg-content">Thinking…</span></div>}
                <div ref={messagesEndRef} />
              </div>
              <div className="coach-input-wrap">
                <div className="coach-input-row">
                  <input
                    ref={inputRef}
                    type="text"
                    className="coach-input"
                    placeholder={coachAvailable ? "Ask about your form..." : "Coach unavailable"}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={loading || !coachAvailable}
                  />
                  <button type="button" className="coach-send" onClick={handleSend} disabled={loading || !input.trim() || !coachAvailable}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 2L11 13" /><path d="M22 2L15 22 11 13 2 9 22 2z" /></svg>
                  </button>
                </div>
              </div>
            </div>
          )}

          {tab === "progress" && (
            <div className="coach-progress">
              {progress.alert && (
                <div className="coach-alert">
                  <p>We noticed a possible concern with your progress. Consider generating a PT report to share with a physical therapist.</p>
                  <button type="button" className="coach-pt-btn" onClick={requestPTReport} disabled={ptLoading}>
                    {ptLoading ? "Generating…" : "Generate PT Report"}
                  </button>
                </div>
              )}
              {chartData.length > 0 ? (
                <>
                  <p className="coach-trend">Trend: {progress.trend}</p>
                  <div className="coach-chart">
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" stroke="rgba(255,255,255,0.6)" fontSize={11} />
                        <YAxis stroke="rgba(255,255,255,0.6)" fontSize={11} domain={[0, 100]} />
                        <Tooltip contentStyle={{ background: "#1a1f2e", border: "1px solid rgba(255,255,255,0.2)" }} />
                        <Line type="monotone" dataKey="score" stroke="#3fb0ff" strokeWidth={2} dot={{ r: 4 }} name="Form score %" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  {chartData.some((d) => d.reps > 0) && (
                    <div className="coach-chart">
                      <ResponsiveContainer width="100%" height={150}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="date" stroke="rgba(255,255,255,0.6)" fontSize={11} />
                          <YAxis stroke="rgba(255,255,255,0.6)" fontSize={11} />
                          <Tooltip contentStyle={{ background: "#1a1f2e", border: "1px solid rgba(255,255,255,0.2)" }} />
                          <Line type="monotone" dataKey="reps" stroke="#4ade80" strokeWidth={2} dot={{ r: 4 }} name="Reps" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </>
              ) : (
                <p className="coach-empty">Complete workouts to see your progress here.</p>
              )}
            </div>
          )}

          {tab === "report" && (
            <div className="coach-report">
              {ptReport ? (
                <>
                  <p className="coach-report-desc">Share this report with your physical therapist:</p>
                  <pre className="coach-report-content">{ptReport}</pre>
                  <button type="button" className="coach-copy" onClick={() => navigator.clipboard?.writeText(ptReport)}>Copy to clipboard</button>
                </>
              ) : (
                <>
                  <p className="coach-report-desc">
                    {progress.alert
                      ? "We detected a possible concern. Generate a formal report to share with a physical therapist."
                      : "Generate a formal report summarizing your workout history and form data for a physical therapist."}
                  </p>
                  <button type="button" className="coach-pt-btn" onClick={requestPTReport} disabled={ptLoading}>
                    {ptLoading ? "Generating…" : "Generate PT Report"}
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}
