import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import accounticon from "../../../assets/icons/account_icon.png";
import teamsLogo from "../../../assets/teams.png";
import googleMeetLogo from "../../../assets/logos_google-meet.png";
import videoicon from "../../../assets/videoicon.png";
import RecentSignals from "./RecentSignals";
import TalkingPoints from "./TalkingPoints";
import { fetchMeetings, fetchTranscript, fetchTranscriptFromVexa } from "../../../api/client";
import "../../../styles/account.css";
import "../../../styles/opportunity.css";

// Generate a consistent pastel colour for a speaker name
function speakerColor(name) {
  const palette = [
    "#4F46E5", "#0891B2", "#059669", "#D97706",
    "#DC2626", "#7C3AED", "#DB2777", "#065F46",
  ];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return palette[h % palette.length];
}

function initials(name) {
  return name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

function fmtTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: true });
}

function MeetingDetails() {
  const { id } = useParams();
  const [meeting, setMeeting] = useState(null);
  const [transcript, setTranscript] = useState(null);   // { status, started_at, … }
  const [segments, setSegments] = useState([]);
  const [txError, setTxError] = useState(null);
  const [activeTab, setActiveTab] = useState("prep");   // "prep" | "transcript"
  const [fetchingTx, setFetchingTx] = useState(false);
  const [fetchTxMsg, setFetchTxMsg] = useState(null);
  const bottomRef = useRef(null);
  const pollRef = useRef(null);

  const loadTranscript = useCallback(() => {
    if (!id) return;
    fetchTranscript(id)
      .then((data) => {
        setTxError(null);
        setTranscript(data?.transcript ?? null);
        setSegments(data?.segments ?? []);
      })
      .catch((err) => {
        // 404 = no transcript yet — not an error to surface noisily
        if (!err?.message?.includes("404")) setTxError(err.message);
      });
  }, [id]);

  // Start polling; stop when transcript is finalized/terminated
  useEffect(() => {
    loadTranscript();
    pollRef.current = setInterval(() => {
      if (transcript?.status && transcript.status !== "in_progress") {
        clearInterval(pollRef.current);
        return;
      }
      loadTranscript();
    }, 5000);
    return () => clearInterval(pollRef.current);
  }, [loadTranscript]);                       // eslint-disable-line

  // Stop polling once transcript reaches a terminal state
  useEffect(() => {
    if (transcript?.status && transcript.status !== "in_progress") {
      clearInterval(pollRef.current);
    }
  }, [transcript?.status]);

  // Auto-scroll to bottom when new segments arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [segments.length]);

  useEffect(() => {
    fetchMeetings()
      .then((data) => {
        const items = data?.meetings || data || [];
        const found = items.find((m) => String(m.meeting_id) === String(id));
        if (found) {
          const start = new Date(found.meeting_start_time);
          const end = new Date(found.meeting_end_time);
          const formatTime = (d) =>
            d.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
            });
          const formatDate = (d) => {
            const dd = String(d.getDate()).padStart(2, "0");
            const mm = String(d.getMonth() + 1).padStart(2, "0");
            const yy = String(d.getFullYear()).slice(-2);
            return `${dd}/${mm}/${yy}`;
          };
          const durationMin = Math.round((end - start) / 60000);
          const duration =
            durationMin >= 60
              ? `${Math.floor(durationMin / 60)}hr${durationMin % 60 ? ` ${durationMin % 60}min` : ""}`
              : `${durationMin}min`;
          const platform = found.platform || "";
          const isGoogle = platform.toLowerCase().includes("google");
          setMeeting({
            title: found.title,
            accountName: found.account_name || "-",
            startTime: formatTime(start),
            endTime: formatTime(end),
            date: formatDate(start),
            duration,
            platform:
              platform.replace(/^(Internal|External)\s*/i, "").trim() || "Teams",
            platformIcon: isGoogle ? googleMeetLogo : teamsLogo,
            attendeesEmails: found.attendees_emails || "",
            meetingUrl: found.body || null,
            botStatus: found.bot_status || null,
            vexaBotId: found.vexa_bot_id || null,
          });
        }
      })
      .catch((err) => console.error("Failed to fetch meeting:", err));
  }, [id]);

  return (
    <div className="dv-page">
      <div className="main">
        <div className="quarter-bar">
          Q2 FY2024 · Week 10 of 12 ·{" "}
          <span className="closure-badge">Closure Phase</span>
        </div>

        <div className="page-header">
          <div className="page-header-row" style={{ alignItems: "center" }}>
            <span className="topbar-title">Execute: Meeting Prep</span>
            {meeting && (
              <div className="topbar-actions">
                <button
                  className="sd-btn-join-ghost"
                  onClick={() => {
                    if (meeting.meetingUrl) window.open(meeting.meetingUrl, "_blank");
                  }}
                >
                  <img src={videoicon} alt="Video" className="sd-prep-icon" />{" "}
                  Join
                </button>
              </div>
            )}
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 13,
              color: "#374151",
              marginTop:"-10px",
              paddingLeft:"25px"
            }}
          >
            {meeting && (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <img
                    src={meeting.platformIcon}
                    alt={meeting.platform}
                    style={{ width: 18, height: 18 }}
                  />
                  <span style={{ fontWeight: 500 }}>{meeting.platform}</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span>{meeting.date}  {meeting.startTime}  {meeting.endTime}  {meeting.duration}</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div
                    className="sd-avatar-group"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                    }}
                  >
                    {meeting.attendeesEmails.split(",").filter(Boolean).map((email, i) => {
                      const initials = email
                        .trim()
                        .split("@")[0]
                        .split(".")
                        .map((w) => w[0]?.toUpperCase() || "")
                        .join("")
                        .slice(0, 2);
                      return (
                        <div
                          key={i}
                          className="sd-avatar sd-avatar-initials-ra"
                          title={email.trim()}
                        >
                          {initials}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Tab bar */}
        <div style={{
          display: "flex", gap: 0, borderBottom: "2px solid #e8eaed",
          background: "#fff", paddingLeft: 24,
        }}>
          {[
            { key: "prep", label: "Meeting Prep" },
            { key: "transcript", label: "Transcript" },
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              style={{
                padding: "10px 20px", fontSize: 13, fontWeight: 600, cursor: "pointer",
                border: "none", background: "none",
                borderBottom: activeTab === t.key ? "2px solid #1a73e8" : "2px solid transparent",
                color: activeTab === t.key ? "#1a73e8" : "#64748b",
                marginBottom: -2,
              }}
            >
              {t.label}
              {t.key === "transcript" && transcript?.status === "in_progress" && (
                <span style={{
                  marginLeft: 6, display: "inline-block", width: 7, height: 7,
                  borderRadius: "50%", background: "#DC2626",
                  animation: "txPulse 1.2s ease-in-out infinite",
                }} />
              )}
              {t.key === "transcript" && segments.length > 0 && transcript?.status !== "in_progress" && (
                <span style={{
                  marginLeft: 6, background: "#1a73e8", color: "#fff",
                  borderRadius: 999, fontSize: 10, fontWeight: 700,
                  padding: "1px 6px",
                }}>{segments.length}</span>
              )}
            </button>
          ))}
        </div>

        <div className="content">
          {/* Left Column */}
          <div className="col-main">
            {/* Deal Summary */}
            {activeTab === "prep" && <>
            <div className="card">
              <div className="card-header-row" style={{ marginBottom: 16 }}>
                <div className="card-title">Deal Summary</div>
              </div>
              <div className="deal-inner">
                <div className="deal-content">
                  <img src={accounticon} alt="Account" className="deal-image" />
                  <div className="deal-desc">
                    Ford is Lenovo's largest automotive enterprise account in
                    North America with 177,000+ employees. Key decision maker
                    Brian Novak (VP IT Infrastructure) has completed three prior
                    Lenovo purchases totaling $12M. The account has 3 open
                    opportunities and an annual revenue potential of $25M.
                    Strong executive relationship — Brian met our CTO at Lenovo
                    Accelerate conference last quarter. Manufacturing division
                    urgently needs data center expansion for new production line
                    launching August 1.
                  </div>
                </div>
              </div>
            </div>
            {/* Talking Points */}
            <div className="card">
              <div className="card-title">Talking Points</div>
              <TalkingPoints />
            </div>

            {/* Recent Signals */}
            <div className="card">
              <RecentSignals />
            </div>
            </>}

            {/* Live Transcript — shown as tab panel when activeTab=transcript, hidden in prep tab */}
            {activeTab === "transcript" &&
            <div className="card" style={{ padding: 0, overflow: "hidden", flex: 1, display: "flex", flexDirection: "column" }}>
              {/* header */}
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "14px 18px 10px", borderBottom: "1px solid #f1f5f9",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontWeight: 700, fontSize: 15 }}>Live Transcript</span>
                  {transcript?.status === "in_progress" && (
                    <span style={{
                      display: "inline-flex", alignItems: "center", gap: 5,
                      background: "#FEF2F2", color: "#DC2626",
                      fontSize: 11, fontWeight: 600, padding: "2px 8px",
                      borderRadius: 999, letterSpacing: "0.04em",
                    }}>
                      <span style={{
                        width: 7, height: 7, borderRadius: "50%",
                        background: "#DC2626",
                        animation: "txPulse 1.2s ease-in-out infinite",
                        display: "inline-block",
                      }} />
                      LIVE
                    </span>
                  )}
                  {transcript?.status === "finalized" && (
                    <span style={{
                      background: "#F0FDF4", color: "#15803D",
                      fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 999,
                    }}>Recording ended</span>
                  )}
                  {transcript?.status === "terminated_partial" && (
                    <span style={{
                      background: "#FFFBEB", color: "#B45309",
                      fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 999,
                    }}>Partial recording</span>
                  )}
                </div>
                {transcript?.started_at && (
                  <span style={{ fontSize: 11, color: "#94A3B8" }}>
                    Started {fmtTime(transcript.started_at)}
                  </span>
                )}
                <button
                  onClick={() => {
                    setFetchingTx(true);
                    setFetchTxMsg(null);
                    fetchTranscriptFromVexa(id)
                      .then((res) => {
                        setFetchTxMsg(`Fetched ${res.segments_found} segment${res.segments_found !== 1 ? "s" : ""}`);
                        loadTranscript();
                      })
                      .catch((err) => setFetchTxMsg(`Error: ${err.message}`))
                      .finally(() => setFetchingTx(false));
                  }}
                  disabled={fetchingTx}
                  style={{
                    marginLeft: "auto", padding: "5px 14px", fontSize: 12, fontWeight: 600,
                    border: "1px solid #1a73e8", borderRadius: 6, background: "#fff",
                    color: "#1a73e8", cursor: fetchingTx ? "not-allowed" : "pointer",
                    opacity: fetchingTx ? 0.6 : 1,
                  }}
                >
                  {fetchingTx ? "Fetching…" : "⬇ Fetch from Vexa"}
                </button>
              </div>
              {fetchTxMsg && (
                <div style={{ padding: "4px 18px", fontSize: 11, color: fetchTxMsg.startsWith("Error") ? "#DC2626" : "#15803D" }}>
                  {fetchTxMsg}
                </div>
              )}

              {/* body */}
              <div style={{
                flex: 1, overflowY: "auto",
                padding: "12px 18px", display: "flex", flexDirection: "column", gap: 14,
              }}>
                <style>{`
                  @keyframes txPulse {
                    0%,100% { opacity:1; } 50% { opacity:0.3; }
                  }
                `}</style>

                {/* no transcript yet */}
                {!transcript && !txError && (
                  <div style={{ textAlign: "center", padding: "32px 0", color: "#94A3B8", fontSize: 13 }}>
                    <div style={{ fontSize: 28, marginBottom: 8 }}>🎙️</div>
                    {!meeting?.vexaBotId ? (
                      <>
                        <div style={{ fontWeight: 600, color: "#374151", marginBottom: 6 }}>Bot not yet dispatched</div>
                        <div>Click <strong>Join</strong> on the meeting card to send the recording bot into the call, then click <strong>⬇ Fetch from Vexa</strong> once the call ends.</div>
                      </>
                    ) : meeting?.botStatus === "scheduled" || meeting?.botStatus === "joining" ? (
                      <>
                        <div style={{ fontWeight: 600, color: "#374151", marginBottom: 6 }}>Bot is joining the call…</div>
                        <div>Once the call starts recording, click <strong>⬇ Fetch from Vexa</strong> after the call ends to load the transcript.</div>
                      </>
                    ) : (
                      "Waiting for bot to join and start recording…"
                    )}
                  </div>
                )}

                {/* API error */}
                {txError && (
                  <div style={{ color: "#DC2626", fontSize: 12, padding: "8px 0" }}>
                    Could not load transcript: {txError}
                  </div>
                )}

                {/* segments */}
                {segments.map((seg, i) => {
                  const color = speakerColor(seg.speaker_name || "?");
                  return (
                    <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                      <div style={{
                        width: 30, height: 30, borderRadius: "50%",
                        background: color, color: "#fff",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 11, fontWeight: 700, flexShrink: 0, marginTop: 2,
                      }}>
                        {initials(seg.speaker_name || "?")}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          display: "flex", alignItems: "baseline", gap: 8, marginBottom: 2,
                        }}>
                          <span style={{ fontWeight: 600, fontSize: 13, color }}>{seg.speaker_name}</span>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>{fmtTime(seg.start_time)}</span>
                        </div>
                        <p style={{
                          margin: 0, fontSize: 13, color: "#1E293B", lineHeight: 1.55,
                        }}>
                          {seg.utterance_text}
                        </p>
                      </div>
                    </div>
                  );
                })}

                {/* finalized empty */}
                {transcript?.status === "finalized" && segments.length === 0 && (
                  <div style={{ textAlign: "center", color: "#94A3B8", fontSize: 13, padding: "24px 0" }}>
                    No transcript segments recorded.
                  </div>
                )}

                <div ref={bottomRef} />
              </div>

              {/* footer segment count */}
              {segments.length > 0 && (
                <div style={{
                  borderTop: "1px solid #f1f5f9", padding: "8px 18px",
                  fontSize: 11, color: "#94A3B8",
                }}>
                  {segments.length} segment{segments.length !== 1 ? "s" : ""} · refreshes every 5 s
                </div>
              )}
            </div>}
          </div>

          {/* Right Column */}
          <div className="col-side">
            {/* Account Overview */}
            <div className="card account-card">
              <div className="health-header">
                <div className="card-title" style={{ marginBottom: 0 }}>
                  Account Overview
                </div>
              </div>
              <div
                style={{
                  fontSize: 13,
                  color: "#555",
                  lineHeight: 1.6,
                  marginTop: 8,
                }}
              >
                <p>
                  Ford is Lenovo's largest automotive enterprise account in
                  South Asia with 177,000+ employees. Key decision maker Brian
                  Novak has completed three prior Lenovo purchases. The account
                  has 3 open opportunities and an annual revenue potential of
                  $25M. Strong executive relationship — Amit met the CIO at
                  Lenovo Transform last quarter.
                </p>
              </div>
            </div>

            {/* Competitors */}
            <div className="card">
              <div
                className="account-header"
                style={{ fontSize: 15, fontWeight: 700 }}
              >
                Competitors
              </div>
              <div className="health-body">
                <div className="health-row">
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    <span style={{ fontWeight: 600 }}>Dell</span>
                    <span style={{ fontSize: 11, color: "#888" }}>
                      Dell Technologies
                    </span>
                  </div>
                  <span
                    className="dv-badge dv-badge-warning"
                    style={{
                      borderRadius: "999px",
                      background: "#FFF7ED",
                      color: "#C2410C",
                    }}
                  >
                    Direct
                  </span>
                </div>
                <div className="health-row">
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    <span style={{ fontWeight: 600 }}>HP</span>
                    <span style={{ fontSize: 11, color: "#888" }}>HP Inc.</span>
                  </div>
                  <span
                    className="dv-badge dv-badge-warning"
                    style={{
                      borderRadius: "999px",
                      background: "#FFF7ED",
                      color: "#C2410C",
                    }}
                  >
                    Direct
                  </span>
                </div>
              </div>
            </div>

            {/* Deal Information */}
            <div className="card">
              <div className="health-header">
                <div className="card-title" style={{ marginBottom: 0 }}>
                  Deal Information
                </div>
              </div>
              <div className="expipeline compact" style={{ marginBottom: 12 }}>
                <div className="stage">
                  <div className="stage-dot done">✓</div>
                </div>
                <div className="stage">
                  <div className="stage-dot done">✓</div>
                </div>
                <div className="stage">
                  <div className="stage-dot active"></div>
                </div>
                <div className="stage">
                  <div className="stage-dot inactive"></div>
                </div>
                <div className="stage">
                  <div className="stage-dot inactive"></div>
                </div>
              </div>
              <div className="health-row">
                <span>Deal Value</span>
                <span className="health-pct">$8,500,000</span>
              </div>
              <div className="health-row">
                <span>Close Date</span>
                <span className="health-pct">06/30/26</span>
              </div>
              <div className="health-row">
                <span>Forecast</span>
                <span className="health-pct">Best Case</span>
              </div>
              <div className="health-row">
                <span>Owner</span>
                <span className="health-pct">Amit Sharma</span>
              </div>
              <div className="health-row">
                <span>Region</span>
                <span className="health-pct">North America</span>
              </div>
              <div className="health-row">
                <span>Lead Origin</span>
                <span className="health-pct">Partner Referral</span>
              </div>
              <div className="health-row">
                <span>Partner</span>
                <span className="health-pct">Yes</span>
              </div>
            </div>

            {/* Notes */}
            <div className="card">
              <div className="health-header">
                <div className="card-title" style={{ marginBottom: 0 }}>
                  Files & Notes
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <textarea
                  className="dv-note-textarea"
                  placeholder="Text area for Amit to add personal notes"
                  style={{
                    width: "100%",
                    minHeight: 100,
                    resize: "vertical",
                    border: "2px solid #e2e8f0",
                    borderRadius: 8,
                    padding: 12,
                    fontSize: 13,
                    color: "#111827",
                    background: "#F8FAFC",
                    marginBottom: 12,
                  }}
                ></textarea>
                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    className="ct-btn-cancel"
                    style={{
                      padding: "6px 14px",
                      fontSize: 12,
                      border: "none",
                      borderRadius: 6,
                      background: "#CBD5E1",
                      color: "#374151",
                      cursor: "pointer",
                    }}
                  >
                    Clear
                  </button>
                  <button
                    className="sd-action-btn"
                    style={{
                      padding: "6px 14px",
                      fontSize: 12,
                      borderRadius: 6,
                    }}
                  >
                    Draft
                  </button>
                </div>
              </div>

              <div
                style={{
                  marginTop: 16,
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                {[
                  { name: "Meeting prep notes", added: "Added 3 days ago" },
                  { name: "Proposal draft v2", added: "Added 5 days ago" },
                  { name: "Technical specifications", added: "Added 1 week ago" },
                  { name: "TCO comparison vs Dell", added: "Added 2 weeks ago" },
                ].map((file, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: 10,
                      border: "1px solid #e2e8f0",
                      borderRadius: 8,
                    }}
                  >
                    <div
                      style={{
                        width: 36,
                        height: 36,
                        borderRadius: 6,
                        background: "#f3f4f6",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexShrink: 0,
                      }}
                    >
                      <i
                        className="bi bi-file-text"
                        style={{ fontSize: 16, color: "#6b7280" }}
                      ></i>
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: 13,
                          fontWeight: 500,
                          color: "#111827",
                        }}
                      >
                        {file.name}
                      </div>
                      <div style={{ fontSize: 11, color: "#6b7280" }}>
                        {file.added}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MeetingDetails;
