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
import { jsPDF } from "jspdf";
import { getUserId, loadProfile } from "../hooks/useUserProfile";
export function CoachPanel({ apiBase: apiBaseProp, onOpenSettings }) {
  const apiBase = apiBaseProp ?? import.meta.env.VITE_API_BASE ?? "";
  const userId = getUserId();
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState("chat");
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! I'm your AlignAI coach. I know your workout history and can help with form, progress, or anything fitness-related. What's on your mind?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [coachAvailable, setCoachAvailable] = useState(true);
  const [progress, setProgress] = useState({ entries: [], trend: "unknown", alert: null });
  const [ptReport, setPtReport] = useState(null);
  const [ptLoading, setPtLoading] = useState(false);
  const [sendEmailLoading, setSendEmailLoading] = useState(false);
  const [sendEmailError, setSendEmailError] = useState(null);
  const [sendEmailSuccess, setSendEmailSuccess] = useState(false);
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
      fetch(`${apiBase}/api/progress?user_id=${encodeURIComponent(userId)}`)
        .then((r) => r.json())
        .then(setProgress)
        .catch(() => setProgress({ entries: [], trend: "unknown", alert: null }));
    }
  }, [open, apiBase, userId]);

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
        body: JSON.stringify({ message: text, user_id: userId }),
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
    setSendEmailError(null);
    try {
      const res = await fetch(`${apiBase}/api/coach/pt-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: progress.alert || "user_requested", user_id: userId }),
      });
      let data = {};
      try {
        data = await res.json();
      } catch {
        data = {};
      }
      if (!res.ok) {
        const errMsg = data.error || data.detail || `Request failed (${res.status})`;
        setPtReport("");
        setSendEmailError(errMsg);
      } else {
        setPtReport(data.report || "");
        if (!data.report) {
          setSendEmailError("Report was empty. Try again or check OPENAI_API_KEY in server .env.");
        }
      }
      setTab("report");
    } catch (err) {
      setPtReport("");
      setSendEmailError(err?.message || "Network error. Check that the server is running.");
      setTab("report");
    } finally {
      setPtLoading(false);
    }
  };

  const downloadPdf = () => {
    if (!ptReport) return;
    const doc = new jsPDF();
    const lines = doc.splitTextToSize(ptReport.replace(/\r/g, ""), 180);
    let y = 20;
    for (const line of lines) {
      if (y > 270) {
        doc.addPage();
        y = 20;
      }
      doc.text(line, 15, y);
      y += 7;
    }
    doc.save("PT_Referral_Report.pdf");
  };

  const sendToPT = async () => {
    if (!ptReport) return;
    const profile = loadProfile();
    const ptEmail = profile.ptEmail?.trim();
    if (!ptEmail || !ptEmail.includes("@")) {
      setSendEmailError("Add your PT email in Settings first.");
      onOpenSettings?.();
      return;
    }
    setSendEmailLoading(true);
    setSendEmailError(null);
    setSendEmailSuccess(false);
    try {
      const doc = new jsPDF();
      const lines = doc.splitTextToSize(ptReport.replace(/\r/g, ""), 180);
      let y = 20;
      for (const line of lines) {
        if (y > 270) {
          doc.addPage();
          y = 20;
        }
        doc.text(line, 15, y);
        y += 7;
      }
      const pdfBase64 = doc.output("datauristring").split(",")[1];
      const res = await fetch(`${apiBase}/api/pt-report/send-email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pt_email: ptEmail,
          report_text: ptReport,
          pdf_base64: pdfBase64,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || "Failed to send email.");
      }
      setSendEmailSuccess(true);
    } catch (err) {
      setSendEmailError(err.message || "Failed to send email.");
    } finally {
      setSendEmailLoading(false);
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
            <div className="coach-header-actions">
              <button type="button" className="coach-settings-btn" onClick={() => onOpenSettings?.()} aria-label="Settings">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
              </button>
              <button type="button" className="coach-close" onClick={() => setOpen(false)} aria-label="Close">×</button>
            </div>
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
                  <div className="coach-report-actions">
                    <button type="button" className="coach-copy" onClick={() => navigator.clipboard?.writeText(ptReport)}>Copy</button>
                    <button type="button" className="coach-download-pdf" onClick={downloadPdf}>Download PDF</button>
                    <button type="button" className="coach-send-pt" onClick={sendToPT} disabled={sendEmailLoading}>
                      {sendEmailLoading ? "Sending…" : "Send Email to PT"}
                    </button>
                  </div>
                  {sendEmailError && <p className="coach-send-error">{sendEmailError}</p>}
                  {sendEmailSuccess && <p className="coach-send-success">Email sent to your PT.</p>}
                </>
              ) : (
                <>
                  <p className="coach-report-desc">
                    {progress.alert
                      ? "We detected a possible concern. Generate a formal report to share with a physical therapist."
                      : "Generate a formal report summarizing your workout history and form data for a physical therapist."}
                  </p>
                  {sendEmailError && <p className="coach-send-error">{sendEmailError}</p>}
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
