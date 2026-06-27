import accounticon from "../../../assets/icons/account_icon.png";

function RecentSignals() {
  const signals = [
    {
      title: "Budget Approved",
      description: "Finance team scheduled a budget review for Q3 allocations.",
      date: "Jun 10, 2026",
      severity: "High",
    },
    {
      title: "Competitor Activity",
      description: "Schedule a 1:1 with CIO Brian Novak to address technical concerns raised in last call.",
      date: "Jun 8, 2026",
      severity: "Medium",
    },
    {
      title: "Delivery Timeline Concerns",
      description: "Share TechData case study showing 40% faster deployment timeline.",
      date: "Jun 6, 2026",
      severity: "High",
    },
  ];

  const aiSuggestions = [
    {
      title: "Accelerate Proposal",
      description:
        "Q3 budget review may delay procurement. Monitor finance team decisions.",
    },
    {
      title: "Engage CIO Directly",
      description:
        "Dell demo scheduled for Jan 12. Need to present before that date.",
    },
    {
      title: "Leverage Partner Reference",
      description:
        "Brian Novak on leave Jun 15-22. All approvals must happen before Jun 14.",
    },
  ];

  const risksToWatch = [
    {
      title: "Budget Freeze Risk",
      description:
        "Budget approval is critical milestone for $8.5M deal closure.",
      severity: "High",
    },
    {
      title: "Competitor Engagement",
      description:
        "Dell competitive threat requires immediate response with TCO comparison.",
      severity: "Medium",
    },
    {
      title: "Decision Maker OOO",
      description:
        "Production line launches August 1 - delivery by July 25 is non-negotiable.",
      severity: "High",
    },
  ];

  const maxLength = Math.max(signals.length, aiSuggestions.length, risksToWatch.length);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="card-title" style={{ marginBottom: 0 }}>Recent Signals</div>

      {Array.from({ length: maxLength }).map((_, index) => {
        const signal = signals[index];
        const ai = aiSuggestions[index];
        const risk = risksToWatch[index];

        return (
          <div className="rs-card" key={index} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {/* Signal row - bold title + severity badge */}
              {signal && (
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
                  <div className="rs-card-desc" style={{ flex: 1, fontWeight: 600 }}>Signal {index + 1}: {signal.title} - {signal.severity} Priority</div>
                  <span className={`rs-severity rs-severity-${signal.severity.toLowerCase()}`} style={{ whiteSpace: "nowrap" }}>
                    {signal.severity}
                  </span>
                </div>
              )}

              {/* Signal description */}
              {signal && (
                <div style={{ border: "1px solid #E2E8F0", borderRadius: 8, padding: "8px 12px", display: "flex", alignItems: "flex-start", gap: 8 }}>
                  <span style={{ fontSize: 14 }}>📧</span>
                  <div className="rs-card-desc" style={{ flex: 1, fontSize: 12 }}>{signal.description}</div>
                </div>
              )}

              {/* AI Suggestion / Risk warning */}
              {ai && (
                <div style={{ border: "1px solid #E2E8F0", borderRadius: 8, padding: "8px 12px", display: "flex", alignItems: "flex-start", gap: 8 }}>
                  <span style={{ fontSize: 14 }}>⚠️</span>
                  <div className="rs-card-desc" style={{ flex: 1, fontSize: 12 }}>{ai.description}</div>
                </div>
              )}

              {/* Why it is shown */}
              {risk && (
                <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "4px 0" }}>
                  <span style={{ fontSize: 14 }}>💡</span>
                  <span style={{ fontSize: 12, color: "#6B7280" }}><strong>Why it is shown:</strong> {risk.description}</span>
                </div>
              )}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end" }}>
                <button
                  style={{
                    background: "linear-gradient(135deg, rgba(173, 26, 152, 0.15), rgba(55, 104, 199, 0.15))",
                    border: "1px solid #CBD5E1",
                    borderRadius: 50,
                    padding: "6px 12px",
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#374151",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  Complete Task
                </button>
              </div>
          </div>
        );
      })}
    </div>
  );
}

export default RecentSignals;
