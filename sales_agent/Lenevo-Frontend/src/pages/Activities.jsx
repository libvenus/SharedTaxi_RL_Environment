import "../styles/account.css";
import leftArrowIcon from "../assets/icons/left.png";
import rightArrowIcon from "../assets/icons/right1.png";

const staticActivities = [
  {
    id: 1,
    type: "email",
    direction: "inbound",
    subject: "Re: ThinkPad Fleet Refresh Proposal",
    body: "Hi Rachel, we've reviewed the proposal and would like to proceed with the 1,400 unit order. Can we schedule a call to discuss delivery timelines?",
    eventDate: "2026-06-20T10:30:00",
  },
  {
    id: 2,
    type: "meeting",
    direction: "outbound",
    subject: "Quarterly Business Review - Infosys",
    body: "Conducted QBR with Infosys procurement team. Discussed expansion opportunities and renewal timeline for existing ThinkCentre fleet.",
    eventDate: "2026-06-19T14:00:00",
  },
  {
    id: 3,
    type: "email",
    direction: "outbound",
    subject: "Pricing Update: ThinkStation P360 Bundle",
    body: "Sent updated pricing for the ThinkStation P360 bundle including 3-year warranty and DaaS options as requested by the client.",
    eventDate: "2026-06-18T09:15:00",
  },
  {
    id: 4,
    type: "crm",
    direction: "inbound",
    subject: "Deal Stage Updated to Propose",
    body: "Opportunity moved from Develop to Propose stage after successful technical validation and stakeholder alignment.",
    eventDate: "2026-06-17T16:45:00",
  },
  {
    id: 5,
    type: "meeting",
    direction: "inbound",
    subject: "Technical Demo - HPC Cluster Configuration",
    body: "Walked through proposed HPC cluster topology with Infosys architecture team. They confirmed compatibility with their existing infrastructure.",
    eventDate: "2026-06-16T11:00:00",
  },
  {
    id: 6,
    type: "email",
    direction: "outbound",
    subject: "Follow-up: Contract Terms Discussion",
    body: "Sent revised MSA with updated clauses addressing legal team's feedback. Awaiting procurement sign-off by end of week.",
    eventDate: "2026-06-15T08:30:00",
  },
];

function getBadgeClass(type) {
  switch (type) {
    case "email":
      return "dv-badge-success";
    case "meeting":
      return "dv-badge-info";
    case "crm":
      return "dv-badge-warning";
    default:
      return "dv-badge-info";
  }
}

function getTypeLabel(type) {
  switch (type) {
    case "email":
      return "Email";
    case "meeting":
      return "Meeting";
    case "crm":
      return "CRM Update";
    default:
      return type;
  }
}

function getBadge(type) {
  switch (type) {
    case "email":
      return "Expansion signal";
    case "meeting":
      return "High-confidence close";
    case "crm":
      return "Record Updated";
    default:
      return "Activity";
  }
}

function formatDate(date) {
  return new Date(date).toLocaleString();
}

export default function Activities() {
  return (
    <>
      <style>{`
        .activities-page {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          font-size: 13px;
          background: #f5f6f8;
          color: #1a1a2e;
          height: 100vh;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }
        .activities-page .main { height: 100%; display: flex; flex-direction: column; overflow: hidden; }
        .activities-page .quarter-bar { background: #f5f6f8; border-bottom: 1px solid #e8eaed; padding: 4px 24px; font-size: 12px; color: #555; display: flex; align-items: center; gap: 6px; }
        .activities-page .closure-badge { color: #e2001a; font-weight: 600; }
        .activities-page .page-header { background: #f5f6f8; border-bottom: 1px solid #e8eaed; padding: 10px 24px 16px; margin-top: 38px; }
        .activities-page .topbar-title { font-size: 28px; font-weight: 700; color: #111; }
        .activities-page .content { flex: 1; overflow-y: auto; padding: 20px 24px; }

        .dv-activity-container { padding: 0; background: #fafafa; border-radius: 12px; }
        .dv-activity-timeline { background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 24px; }
        .dv-timeline { display: flex; flex-direction: column; gap: 0; }
        .dv-convo-row { display: grid; grid-template-columns: 1fr 48px 1fr; align-items: stretch; gap: 8px; }
        .dv-convo-row .dv-activity-card { margin-bottom: 0; }
        .dv-activity-card { background: white; border: 1px solid #cbd5e1; border-radius: 12px; margin-bottom: 24px; overflow: hidden; }
        .dv-convo-track { position: relative; width: 48px; min-height: 100%; }
        .dv-convo-line { width: 2px; height: 100%; background: #cbd5e1; position: absolute; left: 50%; top: 0; transform: translateX(-50%); }
        .dv-convo-dot { width: 24px; height: 24px; background: #1d4ed8; border: 2px solid #e2e8f0; border-radius: 12px; position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 1; }
        .dv-badge { display: inline-flex; align-items: center; gap: 4px; padding: 4px 12px 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; line-height: 1.2; white-space: nowrap; flex-shrink: 0; }
        .dv-badge-success { background: #ecfdf5; color: #0f172a; }
        .dv-badge-info { background: #eff6ff; color: #0f172a; }
        .dv-badge-warning { background: #fff7ed; color: #0f172a; }
        .dv-activity-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; border-bottom: 1px solid #cbd5e1; min-height: 56px; flex-wrap: nowrap; }
        .dv-activity-type { display: flex; align-items: center; gap: 8px; font-size: 16px; flex: 1; min-width: 0; white-space: nowrap; }
        .dv-activity-body { padding: 16px; }
        .dv-activity-timestamp { font-size: 10px; color: #6a7282; margin-bottom: 8px; }
        .dv-activity-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
        .dv-activity-description { font-size: 14px; line-height: 22px; color: #334155; }
        .dv-sub-tabs { display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 16px; background: transparent; }
        .dv-sub-title { padding: 6px 12px; font-size: 16px; color: #334155; font-weight: 600; line-height: 1; white-space: nowrap; display: inline-flex; align-items: center; }
        .dv-sub-icon { width: 18px; height: 18px; object-fit: contain; vertical-align: middle; }
      `}</style>

      <div className="activities-page">
        <div className="main">
          <div className="quarter-bar">
            Q2 FY2024 · Week 10 of 12 ·{" "}
            <span className="closure-badge">Closure Phase</span>
          </div>

          <div className="page-header">
            <span className="topbar-title">Activities</span>
          </div>

          <div className="content">
            <div className="dv-activity-container">
              <div className="dv-activity-timeline">
                <div className="dv-sub-tabs">
                  <span className="dv-sub-title">
                    <img
                      src={leftArrowIcon}
                      alt="left"
                      className="dv-sub-icon"
                      style={{ marginRight: 5 }}
                    />
                    Inbound
                  </span>
                  <span className="dv-sub-title">
                    Outbound
                    <img
                      src={rightArrowIcon}
                      alt="right"
                      className="dv-sub-icon"
                      style={{ marginLeft: 5 }}
                    />
                  </span>
                </div>

                <div className="dv-timeline">
                  {staticActivities.map((act) => (
                    <div className="dv-convo-row" key={act.id}>
                      {/* LEFT SIDE (Inbound) */}
                      <div className="dv-convo-side">
                        {act.direction === "inbound" && (
                          <div className="dv-activity-card">
                            <div className="dv-activity-header">
                              <div className="dv-activity-type">
                                <span>{getTypeLabel(act.type)}</span>
                              </div>
                              <span className={`dv-badge ${getBadgeClass(act.type)}`}>
                                {getBadge(act.type)}
                              </span>
                            </div>
                            <div className="dv-activity-body">
                              <div className="dv-activity-timestamp">
                                {formatDate(act.eventDate)}
                              </div>
                              <div className="dv-activity-title">
                                {act.subject}
                              </div>
                              <div className="dv-activity-description">
                                {act.body}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* CENTER LINE */}
                      <div className="dv-convo-track">
                        <div className="dv-convo-line"></div>
                        <div className="dv-convo-dot"></div>
                      </div>

                      {/* RIGHT SIDE (Outbound) */}
                      <div className="dv-convo-side">
                        {act.direction === "outbound" && (
                          <div className="dv-activity-card">
                            <div className="dv-activity-header">
                              <div className="dv-activity-type">
                                <span>{getTypeLabel(act.type)}</span>
                              </div>
                              <span className={`dv-badge ${getBadgeClass(act.type)}`}>
                                {getBadge(act.type)}
                              </span>
                            </div>
                            <div className="dv-activity-body">
                              <div className="dv-activity-timestamp">
                                {formatDate(act.eventDate)}
                              </div>
                              <div className="dv-activity-title">
                                {act.subject}
                              </div>
                              <div className="dv-activity-description">
                                {act.body}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
