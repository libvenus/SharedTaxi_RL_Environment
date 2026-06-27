import { useState, useRef, useEffect } from "react";
import accountIcon from "../assets/icons/account_icon.png";

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [isSending, setIsSending] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  const handleSend = async () => {
    if (!input.trim() || isSending) return;
    const userText = input.trim();
    const userMsg = { id: Date.now(), sender: "user", text: userText };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsSending(true);

    try {
      const res = await fetch("/v1/orchestrator/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: userText,
          conversation_id: conversationId || "convo_id",
          trace_id: "string",
        }),
      });
      const data = await res.json();
      if (data.meta?.conversation_id) setConversationId(data.meta.conversation_id);
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, sender: "bot", text: data.display?.text || "No response" },
      ]);

      // Persist scheduled / rescheduled meetings to the calendar DB
      const actions = Array.isArray(data.actions) ? data.actions : [];
      for (const action of actions) {
        if (
          action.status === "success" &&
          (action.type === "schedule_meeting" || action.type === "reschedule_meeting")
        ) {
          const p = action.payload || {};
          try {
            await fetch("/ai-api/meeting-prep", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                meeting_start_time: p.meeting_start_time_utc || p.meeting_start_time,
                meeting_end_time: p.meeting_end_time_utc || p.meeting_end_time,
                platform: p.platform || "Teams",
                title: p.title || "Meeting",
                account_name: p.account_name || "",
                attendees_emails: p.attendees || "",
                meeting_url: p.body || "",
                seller_id: "055DAFE7-9840-451D-8328-5F70A6326C03",
              }),
            });
          } catch (e) {
            console.warn("Failed to persist chatbot-scheduled meeting to calendar:", e);
          }
        }
      }
    } catch (err) {
      console.error("Chat API error:", err);
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, sender: "bot", text: "Sorry, something went wrong. Please try again." },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <style>{`
        @keyframes chatDot {
          0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
      {/* Chat Toggle Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          style={{
            position: "fixed",
            bottom: 24,
            right: 24,
            zIndex: 9999,
            width: 56,
            height: 56,
            borderRadius: "50%",
            background: "linear-gradient(135deg, #1D4ED8, #3B82F6)",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 4px 16px rgba(29, 78, 216, 0.4)",
          }}
          aria-label="Open chat"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      )}

      {/* Chat Panel */}
      {isOpen && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            right: 24,
            zIndex: 9999,
            width: 380,
            height: 520,
            borderRadius: 16,
            background: "#fff",
            boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            border: "1px solid #e2e8f0",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "12px 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderBottom: "1px solid #e2e8f0",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <img src={accountIcon} alt="Lenovo" style={{ width: 28, height: 28 }} />
              <span style={{ fontWeight: 700, fontSize: 14, color: "#0f172a" }}>Lenovo Sales Agent</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button
                onClick={async () => {
                  if (!conversationId) return;
                  try {
                    await fetch(`/v1/orchestrator/reset/${conversationId}`, { method: "POST" });
                  } catch (err) {
                    console.error("Reset error:", err);
                  }
                  setMessages([]);
                  setConversationId(null);
                }}
                style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b", fontSize: 20 }}
                aria-label="Reset chat"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="1 4 1 10 7 10" />
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                </svg>
              </button>
              <button
                onClick={() => setIsOpen(false)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b", fontSize: 20 }}
                aria-label="Close chat"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          </div>

          {/* Content */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "24px 20px",
              display: "flex",
              flexDirection: "column",
              background: "#f8fafc",
            }}
          >
            {messages.length === 0 ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
                {/* AI Icon */}
                <img src={accountIcon} alt="AI" style={{ width: 48, height: 48 }} />

                {/* Greeting */}
                <h3 style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", margin: 0, textAlign: "center" }}>
                  Hello, how may I assist you today?
                </h3>

                {/* Subtext */}
                <p style={{ fontSize: 13, color: "#64748b", margin: 0, textAlign: "center" }}>
                  Below are some ideas to get started
                </p>

                {/* Suggestion Cards */}
                <div style={{ display: "flex", gap: 10, width: "100%", marginTop: 8 }}>
                  <div
                    onClick={() => setInput("Show my pipeline summary")}
                    style={{
                      flex: 1,
                      padding: "14px 12px",
                      background: "#fff",
                      border: "1px solid #e2e8f0",
                      borderRadius: 10,
                      cursor: "pointer",
                      fontSize: 12,
                      color: "#334155",
                      fontWeight: 500,
                      textAlign: "center",
                      lineHeight: 1.4,
                      boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
                    }}
                  >
                    Schedule Meeting
                  </div>
                  <div
                    onClick={() => setInput("What are my priority actions?")}
                    style={{
                      flex: 1,
                      padding: "14px 12px",
                      background: "#fff",
                      border: "1px solid #e2e8f0",
                      borderRadius: 10,
                      cursor: "pointer",
                      fontSize: 12,
                      color: "#334155",
                      fontWeight: 500,
                      textAlign: "center",
                      lineHeight: 1.4,
                      boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
                    }}
                  >
                    What are my priority actions?
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    style={{
                      display: "flex",
                      justifyContent: msg.sender === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    <div
                      style={{
                        maxWidth: "75%",
                        padding: "10px 14px",
                        borderRadius: msg.sender === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                        background: msg.sender === "user" ? "#1D4ED8" : "#fff",
                        color: msg.sender === "user" ? "#fff" : "#0f172a",
                        fontSize: 13,
                        lineHeight: 1.5,
                        boxShadow: msg.sender === "bot" ? "0 1px 3px rgba(0,0,0,0.08)" : "none",
                        border: msg.sender === "bot" ? "1px solid #e2e8f0" : "none",
                      }}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
                {isSending && (
                  <div style={{ display: "flex", justifyContent: "flex-start" }}>
                    <div style={{ padding: "10px 14px", borderRadius: "14px 14px 14px 4px", background: "#fff", border: "1px solid #e2e8f0", boxShadow: "0 1px 3px rgba(0,0,0,0.08)", display: "flex", gap: 4, alignItems: "center" }}>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#94a3b8", animation: "chatDot 1.4s infinite", animationDelay: "0s" }}></span>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#94a3b8", animation: "chatDot 1.4s infinite", animationDelay: "0.2s" }}></span>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#94a3b8", animation: "chatDot 1.4s infinite", animationDelay: "0.4s" }}></span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Footer Input */}
          <div
            style={{
              padding: "12px 16px",
              borderTop: "1px solid #e2e8f0",
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: "#fff",
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question"
              style={{
                flex: 1,
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                padding: "10px 14px",
                fontSize: 13,
                outline: "none",
                background: "#f8fafc",
                color:"black"
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              style={{
                width: 36,
                height: 36,
                borderRadius: 8,
                background: input.trim() ? "#1D4ED8" : "#cbd5e1",
                border: "none",
                cursor: input.trim() ? "pointer" : "not-allowed",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              aria-label="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  );
}
