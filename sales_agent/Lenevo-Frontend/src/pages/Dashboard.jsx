import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/dashboard.css";
import "bootstrap-icons/font/bootstrap-icons.css";
import googleMeetLogo from "../assets/logos_google-meet.png";
import teamsLogo from "../assets/teams.png";
import prepIcon from "../assets/Group.png";
import avatar1 from "../assets/avatar1.jpg";
import avatar2 from "../assets/avatar2.jpg";
import avatar3 from "../assets/avatar3.jpg";
import accountIcon from "../assets/icons/account_icon.png";
import { fetchQuarterPulse, fetchNotifications, fetchTodos, fetchMeetings, fetchOpportunities } from "../api/client";

/**
 * SalesDashboard
 * - Left column: Priority Actions + Accounts Needing Attention (3 rows, expandable)
 * - Right column: Q3 Pulse + Today's Meetings (3 cards)
 */
export default function SalesDashboard() {
  const navigate = useNavigate();
  const [expandedRows, setExpandedRows] = useState({});
  const [quarterPulse, setQuarterPulse] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [showAllNotifications, setShowAllNotifications] = useState(false);
  const [todoTasks, setTodoTasks] = useState([]);
  const [todayMeetings, setTodayMeetings] = useState([]);
  const [attnAccounts, setAttnAccounts] = useState([]);

  const today = new Date().toISOString().split("T")[0];
  const SELLER_ID = "055DAFE7-9840-451D-8328-5F70A6326C03";

  useEffect(() => {
    fetchQuarterPulse(SELLER_ID)
      .then((res) => setQuarterPulse(res))
      .catch((err) => console.error("Failed to fetch quarter pulse:", err));

    fetchNotifications(SELLER_ID, 6)
      .then((res) => setNotifications(res?.items || []))
      .catch((err) => console.error("Failed to fetch notifications:", err));

    fetchTodos({ filter_type: "all", seller_id: SELLER_ID })
      .then((res) => setTodoTasks(res?.tasks || []))
      .catch((err) => console.error("Failed to fetch todos:", err));

    fetchMeetings({ filter: "all_meetings", seller_id: SELLER_ID })
      .then((data) => {
        const items = data?.meetings || data || [];
        const todayStr = new Date().toISOString().split("T")[0];
        const filtered = items.filter((m) => m.meeting_start_time && m.meeting_start_time.startsWith(todayStr));
        setTodayMeetings(filtered);
      })
      .catch((err) => console.error("Failed to fetch meetings:", err));

    // Accounts needing attention: seller-scoped opps sorted by lowest deal health
    fetchOpportunities({ pageSize: 50, ownerId: SELLER_ID })
      .then((data) => {
        const opps = (data?.items || []).slice().sort((a, b) => (a.dealHealth ?? 100) - (b.dealHealth ?? 100));
        const seen = new Set();
        const grouped = [];
        for (const o of opps) {
          if (!o.accountName || seen.has(o.accountName)) continue;
          seen.add(o.accountName);
          const health = typeof o.dealHealth === "number" ? o.dealHealth : 100;
          let statusLabel, statusClass, dotClass;
          if (health < 40) {
            statusLabel = "At Risk"; statusClass = "sd-status-at-risk"; dotClass = "sd-status-dot-red";
          } else if (health < 65) {
            statusLabel = "Needs Attention"; statusClass = "sd-status-needs-attention"; dotClass = "sd-status-dot-amber";
          } else {
            statusLabel = "Opportunity"; statusClass = "sd-status-opportunity"; dotClass = "sd-status-dot-blue";
          }
          const daysAgo = o.lastActivity
            ? Math.floor((Date.now() - new Date(o.lastActivity)) / 86400000)
            : null;
          const activityNote = daysAgo !== null
            ? `Last activity ${daysAgo} day${daysAgo !== 1 ? "s" : ""} ago.`
            : "No recent activity logged.";
          const closeNote = o.closeDate
            ? `Close date ${new Date(o.closeDate).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}.`
            : "";
          grouped.push({
            key: o.accountId || o.accountName,
            name: o.accountName,
            deal: o.name,
            stage: typeof o.stage === "object" ? (o.stage?.label || o.stage?.raw || "") : (o.stage || ""),
            health,
            riskCount: o.riskCount || 0,
            statusLabel, statusClass, dotClass,
            description: [activityNote, closeNote, o.riskCount > 0 ? `${o.riskCount} risk flag${o.riskCount !== 1 ? "s" : ""} detected.` : ""].filter(Boolean).join(" "),
          });
          if (grouped.length >= 3) break;
        }
        setAttnAccounts(grouped);
      })
      .catch((err) => console.error("Failed to fetch accounts needing attention:", err));
  }, []);

  const pendingTasks = todoTasks.filter((t) => t.status?.toLowerCase() !== "completed");
  const dueTodayTasks = pendingTasks.filter((t) => t.due_date === today);
  const priorityActions = pendingTasks.slice(0, 4);

  const toggleRow = (key) => {
    setExpandedRows((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="sd-wrapper">
      <div className="row g-4">
        <div className="col-12 col-lg-8">
          <div className="sd-priority-panel">
            <div className="sd-priority-header">
              <h2 className="sd-priority-title">
                Priority Actions
                <span className="sd-priority-meta">
                  {pendingTasks.length} total actions ·{" "}
                  <span className="sd-priority-due-today">{dueTodayTasks.length} due today</span>
                </span>
              </h2>

              <a href="#" className="sd-view-all-link" onClick={(e) => { e.preventDefault(); navigate("/todo"); }}>
                View All
                <svg
                  viewBox="0 0 16 16"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M6 3l5 5-5 5"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </a>
            </div>

            <div className="sd-actions-grid">
              {priorityActions.map((task) => {
                const isDueToday = task.due_date === today;
                const isFuture = task.due_date && task.due_date > today;
                const typeTag = (task.type_tag || "").toLowerCase();
                const actionLabel = typeTag === "outreach" ? "Draft Mail" : typeTag === "document" ? "Generate Draft" : "Review and Update";
                return (
                  <div className="sd-action-card" key={task.id}>
                    <div className="sd-action-card-header">
                      <span className="sd-action-company">{task.task_title || "Task"}</span>
                      {isDueToday ? (
                        <span className="sd-badge-due-today">
                          <span className="sd-badge-icon-red">
                            <i className="bi bi-exclamation-circle-fill"></i>
                          </span>
                          Due Today
                        </span>
                      ) : isFuture ? (
                        <span className="sd-badge-due-soon">
                          <span className="sd-badge-icon-warning">
                            <i className="bi bi-exclamation-triangle-fill"></i>
                          </span>
                          Due Soon
                        </span>
                      ) : task.due_date ? (
                        <span className="sd-badge-due-today">
                          <span className="sd-badge-icon-red">
                            <i className="bi bi-exclamation-circle-fill"></i>
                          </span>
                          Overdue
                        </span>
                      ) : (
                        <span className="sd-badge-due-soon">
                          <span className="sd-badge-icon-warning">
                            <i className="bi bi-exclamation-triangle-fill"></i>
                          </span>
                          No date
                        </span>
                      )}
                    </div>
                    <p className="sd-action-desc">{task.notes}</p>
                    <button className="sd-action-btn" onClick={() => typeTag === "action" ? navigate("/execute") : navigate(`/todo/${task.id}`, { state: { selectedTask: task } })}>{actionLabel}</button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="sd-accounts-panel">
            <div className="sd-accounts-header">
              <h2 className="sd-accounts-title">
                Accounts Needing Attention
                <span className="sd-accounts-meta">
                  {attnAccounts.length > 0 ? (
                    <>
                      {attnAccounts.length} shown ·{" "}
                      <span className="sd-accounts-risk">
                        {attnAccounts.filter((a) => a.statusLabel === "At Risk").length} at risk
                      </span>
                    </>
                  ) : "Loading…"}
                </span>
              </h2>

              <a href="#" className="sd-view-all-link" onClick={(e) => { e.preventDefault(); navigate("/accounts"); }}>
                View All
                <svg className="sd-chevron-svg" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </a>
            </div>

            {attnAccounts.map((acct) => (
              <div key={acct.key} className="sd-account-row">
                <div className="sd-account-row-header">
                  <span className="sd-account-name">{acct.name}</span>
                  <span className={acct.statusClass}>
                    {acct.statusLabel} <span className={acct.dotClass}></span>
                  </span>
                </div>

                <div className="sd-account-task-row">
                  <div className="sd-account-task-body">
                    <div className="sd-account-task-title">
                      {acct.deal}
                      {acct.stage && <span style={{ fontWeight: 400, color: "#64748b", fontSize: 11, marginLeft: 6 }}>· {acct.stage}</span>}
                      <span style={{ fontWeight: 400, color: "#94a3b8", fontSize: 11, marginLeft: 6 }}>Health {acct.health}</span>
                    </div>
                    <p className={`sd-account-task-desc${expandedRows[acct.key] ? " sd-account-task-desc-expanded" : ""}`}>
                      {acct.description}
                    </p>
                  </div>

                  <button
                    type="button"
                    className={`sd-account-chevron-btn${expandedRows[acct.key] ? " is-expanded" : ""}`}
                    onClick={() => toggleRow(acct.key)}
                    aria-label="Toggle account details"
                  >
                    <svg className="sd-account-chevron sd-chevron-svg" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M6 3l5 5-5 5" stroke="#9ca3af" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}

            {attnAccounts.length === 0 && (
              <p style={{ padding: "12px 0", color: "#94a3b8", fontSize: 13 }}>No accounts flagged for attention.</p>
            )}
          </div>
        </div>

        <div className="col-12 col-lg-4">
          <div className="sd-q3-panel">
            <div className="sd-q3-header">
              <h2 className="sd-q3-title">{quarterPulse ? `${quarterPulse.quarterLabel} Pulse` : "Quarter Pulse"}</h2>
              <span className="sd-q3-days-left">{quarterPulse?.daysLeftInQuarter ?? 0} days left</span>
            </div>

            <div className="sd-q3-metric">
              <div className="sd-q3-metric-header">
                <span className="sd-q3-metric-label">Quota Attainment</span>
                <span className="sd-q3-metric-value">{quarterPulse?.quotaAttainment?.displayValue ?? "0%"}</span>
              </div>
              <div className="sd-progress-track">
                <div
                  className="sd-progress-fill-blue"
                  style={{ width: `${quarterPulse?.quotaAttainment?.progressFillPercent ?? 0}%` }}
                ></div>
              </div>
            </div>

            <div className="sd-q3-metric">
              <div className="sd-q3-metric-header">
                <span className="sd-q3-metric-label">Pipeline Coverage</span>
                <span className="sd-q3-metric-value">{quarterPulse?.pipelineCoverage?.displayValue ?? "0x"}</span>
              </div>
              <div className="sd-progress-track">
                <div
                  className="sd-progress-fill-yellow"
                  style={{ width: `${quarterPulse?.pipelineCoverage?.progressFillPercent ?? 0}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="sd-meetings-panel">
            <div className="sd-meetings-header">
              <h2 className="sd-meetings-title">Today's Meetings</h2>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <button
                  className="sd-schedule-btn"
                  onClick={() => navigate("/execute/add-meeting")}
                  title="Schedule a meeting"
                >
                  <i className="bi bi-plus-lg"></i> Schedule
                </button>
                <a href="#" className="sd-view-all-link" onClick={(e) => { e.preventDefault(); navigate("/execute"); }}>
                  View All
                  <svg
                    className="sd-chevron-svg"
                    viewBox="0 0 16 16"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M6 3l5 5-5 5"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </a>
              </div>
            </div>

            {todayMeetings.length === 0 && (
              <p style={{ padding: 16, color: "#64748b", fontSize: 13 }}>No meetings found for today</p>
            )}

            {todayMeetings.map((meeting) => {
              const platform = meeting.platform || "";
              const isGoogle = platform.toLowerCase().includes("google");
              const platformIcon = isGoogle ? googleMeetLogo : teamsLogo;
              const platformLabel = platform.replace(/^(Internal|External)\s*/i, "").trim() || "Teams";
              const typeLabel = platform.toLowerCase().includes("external") ? "External" : "Internal";
              const formatTime = (dateStr) => {
                const d = new Date(dateStr);
                return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: true });
              };
              return (
                <div className="sd-meeting-card" key={meeting.meeting_id}>
                  <div className="sd-meeting-card-header">
                    <div className="sd-meeting-platform">
                      <div>
                        <img src={platformIcon} alt={platformLabel} className="sd-platform-icon sd-platform-meet" />
                      </div>
                      {typeLabel} · {platformLabel}
                    </div>
                    <span className="sd-meeting-time1">{formatTime(meeting.meeting_start_time)} - {formatTime(meeting.meeting_end_time)}</span>
                  </div>

                  <div className="sd-meeting-title">{meeting.title}</div>

                  <div className="sd-meeting-avatars">
                    <div className="sd-avatar-group">
                      {meeting.attendees_emails && (
                        <div className="sd-avatar sd-avatar-initials-ra">
                          {meeting.attendees_emails.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="sd-meeting-actions">
                    <button className="sd-btn-prep" onClick={() => navigate(`/execute/meeting/${meeting.meeting_id}`)}>
                      <img src={prepIcon} alt="Prep" className="sd-prep-icon" />
                      Prep
                    </button>
                    <button className="sd-btn-join-primary" onClick={() => meeting.meeting_url && window.open(meeting.meeting_url, "_blank")}>Join</button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="sd-deadlines-panel">
            <div className="sd-deadlines-header">
              <h2 className="sd-deadlines-title">What Changed</h2>
              <a href="#" className="sd-view-all-link" onClick={(e) => { e.preventDefault(); navigate("/activity-timeline"); }}>
                View All
                <svg
                  className="sd-chevron-svg"
                  viewBox="0 0 16 16"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M6 3l5 5-5 5"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </a>
            </div>

            {(showAllNotifications ? notifications : notifications.slice(0, 4)).map((item) => (
              <div className="sd-summary-line" key={item.id}>
                <div>
                  <div className="sd-summary-label-row">
                    <div className="sd-summary-label">
                      <img src={accountIcon} alt="AI" className="sd-summary-icon" />
                      {item.title}
                    </div>
                    <span className="sd-summary-time">{item.categoryLabel}</span>
                  </div>
                  <div className="sd-summary-desc">
                    {item.summary}
                  </div>
                  <div className="sd-summary-desc" style={{ color: "#6b7280", fontSize: 11 }}>
                    {item.opportunityName}
                  </div>
                </div>
              </div>
            ))}

          </div>
        </div>
      </div>
    </div>
  );
}
