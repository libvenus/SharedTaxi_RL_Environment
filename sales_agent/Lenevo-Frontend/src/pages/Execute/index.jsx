import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../../styles/execute.css";
import prepIcon from "../../assets/Group.png";
import teamsLogo from "../../assets/teams.png";
import googleMeetLogo from "../../assets/logos_google-meet.png";
import videoicon from "../../assets/videoicon.png";
import { fetchMeetings, fetchPostMeetingReview, joinMeetingBot } from "../../api/client";
function Execute() {
  const navigate = useNavigate();
  const [meetings, setMeetings] = useState([]);
  const [postMeetings, setPostMeetings] = useState([]);
  const [activeFilter, setActiveFilter] = useState("all_meetings");

  const filters = [
    { key: "today", label: "Today" },
    { key: "tomorrow", label: "Tomorrow" },
    { key: "this_week", label: "This Week" },
    { key: "all_meetings", label: "All Meetings" },
    { key: "total_time", label: "Total Time Taken" },
  ];

  useEffect(() => {
    fetchMeetings({ filter: activeFilter })
      .then((data) => {
        const items = data?.meetings || data || [];
        const staticData = [
          { accountName: "Ford Motor Company", deal: "$2.95M", stage: "Develop" },
          { accountName: "AT&T Inc.", deal: "$1.75M", stage: "Qualify" },
          { accountName: "Infosys Ltd.", deal: "$2.1M Workstation Refresh", stage: "Discovery" },
          { accountName: "Microsoft Corp.", deal: "$6.8M Enterprise Fleet", stage: "Qualification" },
        ];
        const filtered = items.filter((m) => m.title?.toLowerCase() !== "test");
        const mapped = filtered.map((m, idx) => {
          const extra = staticData[idx % staticData.length];
          const start = new Date(m.meeting_start_time);
          const end = new Date(m.meeting_end_time);
          const formatTime = (d) =>
            d.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
            });
          const platform = m.platform || "";
          const isGoogle = platform.toLowerCase().includes("google");
          return {
            meeting_id: m.meeting_id,
            startTime: formatTime(start),
            endTime: formatTime(end),
            platform:
              platform.replace(/^(Internal|External)\s*/i, "").trim() ||
              "Teams",
            platformIcon: isGoogle ? googleMeetLogo : teamsLogo,
            type: platform.toLowerCase().includes("external")
              ? "External"
              : "Internal",
            title: m.title,
            accountName: m.account_name || extra.accountName,
            deal: extra.deal,
            stage: extra.stage,
            company: m.organiser_name || m.account_name || "",
            attendeesEmails: m.attendees_emails || "",
            attendees: (Array.isArray(m.attendees) ? m.attendees : []).map(
              (a) => a.name,
            ),
            avatars: (Array.isArray(m.attendees) ? m.attendees : []).map(
              (a) => ({
                type: "initials",
                text: a.name
                  .split(" ")
                  .map((w) => w[0])
                  .join("")
                  .slice(0, 2)
                  .toUpperCase(),
              }),
            ),
            meetingUrl: m.body || null,
            passcode: m.passcode || "",
            joinMeetingId: m.join_meeting_id || "",
          };
        });
        setMeetings(mapped);
      })
      .catch((err) => console.error("Failed to fetch meetings:", err));
  }, [activeFilter]);

  useEffect(() => {
    fetchPostMeetingReview()
      .then((data) => setPostMeetings(data?.meetings || data || []))
      .catch((err) => console.error("Failed to fetch post-meeting data:", err));
  }, []);

  function handleClick(meetingId) {
    navigate(`/execute/meeting/${meetingId}`);
  }
  function handleJoin(meeting) {
    if (meeting.meetingUrl) {
      joinMeetingBot({ meeting_id: meeting.meeting_id })
        .catch((err) => console.error("Bot join failed:", err));
      window.open(meeting.meetingUrl, "_blank");
    }
  }
  function handleReview(meetingId) {
    navigate(`/execute/postmeeting/${meetingId}`);
  }
  return (
    <div className="dv-page">
      <div className="main">
        <div className="quarter-bar">
          Q2 FY2024 · Week 10 of 12 ·{" "}
          <span className="closure-badge">Closure Phase</span>
        </div>

        <div className="page-header">
          <div className="page-header-row">
            <span className="topbar-title">My Calendar</span>
            <div className="topbar-actions">
              <button
                className="ex-add-icon-btn"
                onClick={() => navigate("/execute/add-meeting")}
              >
                <i className="ri-add-line"></i>
              </button>
            </div>
          </div>
        </div>

        <div className="content">
          <div className="col-main">
            {/* <div className="mr-stats-row">
                {filters.map((f) => (
                  <div
                    key={f.key}
                    className={`mr-stat-card ${activeFilter === f.key ? "active" : ""}`}
                    onClick={() => setActiveFilter(f.key)}
                    style={{ cursor: "pointer" }}
                  >
                    <span className="label">{f.label}</span>
                    <span className="value">
                      {activeFilter === f.key ? meetings.length : ""}
                    </span>
                  </div>
                ))}
              </div> */}

            <div className="sd-meetings-panel">
              <div className="sd-meetings-header">
                <h2 className="sd-meetings-title">
                  {new Date().toLocaleDateString("en-US", {
                    day: "numeric",
                    month: "long",
                  })}
                </h2>
              </div>
              {/* Post-Meeting Review items */}
              {postMeetings?.map((meeting) => {
                const platform = meeting.meeting_platform || "";
                const isGoogle = platform.toLowerCase().includes("google");
                const pIcon = isGoogle ? googleMeetLogo : teamsLogo;
                const platformLabel =
                  platform.replace(/^(Internal|External)\s*/i, "").trim() ||
                  "Teams";
                const typeLabel = platform.toLowerCase().includes("external")
                  ? "External"
                  : "Internal";
                const fmtTime = (dateStr) => {
                  const d = new Date(dateStr);
                  return d.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: true,
                  });
                };
                const isCompleted =
                  meeting.status &&
                  meeting.status.toLowerCase() === "completed";
                return (
                  <div
                    className="ex-meeting-card"
                    key={`post-${meeting.meeting_id}`}
                  >
                    <div className="ex-meeting-layout">
                      <div className="ex-meeting-time-block">
                        <div className="ex-start-time">
                          {fmtTime(meeting.meeting_start_time)}
                        </div>
                        <div className="ex-end-time">
                          {meeting.meeting_end_time
                            ? fmtTime(meeting.meeting_end_time)
                            : ""}
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 4,
                            marginTop: 4,
                          }}
                        >
                          <img
                            src={pIcon}
                            alt={platformLabel}
                            className="sd-platform-icon sd-platform-meet"
                            style={{ width: 14, height: 14 }}
                          />
                          <span style={{ fontSize: 11, color: "#64748b" }}>
                            {platformLabel}
                          </span>
                        </div>
                      </div>

                      <div className="ex-meeting-content">
                        <div className="ex-meeting-title-row">
                          <div className="sd-meeting-title">
                            {meeting.title}
                          </div>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            marginTop: 4,
                          }}
                        >
                          <span style={{ fontSize: 12, color: "#64748b" }}>
                            Organizer
                          </span>
                          <span
                            style={{
                              fontSize: 11,
                              fontWeight: 600,
                              padding: "2px 8px",
                              borderRadius: 12,
                              background: isCompleted ? "#ECFDF5" : "#FEF2F2",
                              color: isCompleted ? "#047857" : "#B91C1C",
                              border: `1px solid ${isCompleted ? "#A7F3D0" : "#FECACA"}`,
                            }}
                          >
                            {meeting.crm_updates} CRM Updates
                          </span>
                        </div>
                      </div>

                      <div className="ex-meeting-actions">
                        {isCompleted ? (
                          <button
                            className="sd-btn-prep"
                            onClick={() => handleReview(meeting.meeting_id)}
                          >
                            View Summary
                          </button>
                        ) : (
                          <button
                            className="ad-btn-gradient"
                            onClick={() => handleReview(meeting.meeting_id)}
                          >
                            Review &amp; Approve
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}

              {meetings.map((meeting, index) => (
                <div className="ex-meeting-card" key={index}>
                  <div className="ex-meeting-layout">
                    <div className="ex-meeting-time-block">
                      <div className="ex-start-time">{meeting.startTime}</div>
                      <div className="ex-end-time">{meeting.endTime}</div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 4,
                          marginTop: 4,
                        }}
                      >
                        <img
                          src={meeting.platformIcon}
                          alt={meeting.platform}
                          className="sd-platform-icon sd-platform-meet"
                          style={{ width: 14, height: 14 }}
                        />
                        <span style={{ fontSize: 11, color: "#64748b" }}>
                          {meeting.platform}
                        </span>
                      </div>
                    </div>

                    <div className="ex-meeting-content">
                      <div className="ex-meeting-title-row">
                        <div className="sd-meeting-title">
                          {meeting.accountName}
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginLeft: 12 }}>
                          <span style={{ fontSize: 12, color: "#475569", fontWeight: 500 }}>{meeting.deal}</span>
                          <span style={{ fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 12, background: "#EFF6FF", color: "#1D4ED8", border: "1px solid #BFDBFE" }}>{meeting.stage}</span>
                        </div>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 10,
                          marginTop: 4,
                        }}
                      >
                        {meeting.attendeesEmails && (
                          <div
                            className="sd-avatar-group"
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 4,
                            }}
                          >
                            {meeting.attendeesEmails.split(",").map((email, i) => {
                              const initials = email.trim().split("@")[0]
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
                        )}
                        <span
                          style={{
                            fontSize: 12,
                            fontWeight: 700,
                            color: "#0f172a",
                          }}
                        >
                          {meeting.title}
                        </span>
                      </div>
                    </div>

                    <div className="ex-meeting-actions">
                      <button
                        className="ad-btn-gradient"
                        onClick={() => handleClick(meeting.meeting_id)}
                      >
                        <img
                          src={prepIcon}
                          alt="Prep"
                          className="sd-prep-icon"
                        />
                        Prep
                      </button>

                      <button
                        className="sd-btn-join-ghost"
                        onClick={() => handleJoin(meeting)}
                      >
                        <img
                          src={videoicon}
                          alt="Video"
                          className="sd-prep-icon"
                        />{" "}
                        Join
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="col-side">
            {/* Card 1: Calendar */}
            <div className="card">
              <div className="health-header">
                <div className="card-title" style={{ marginBottom: 0 }}>
                  Calendar
                </div>
              </div>
              {(() => {
                const today = new Date();
                const year = today.getFullYear();
                const month = today.getMonth();
                const firstDay = new Date(year, month, 1).getDay();
                const daysInMonth = new Date(year, month + 1, 0).getDate();
                const dayNames = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
                const cells = [];
                for (let i = 0; i < firstDay; i++) cells.push(null);
                for (let d = 1; d <= daysInMonth; d++) cells.push(d);
                return (
                  <div style={{ padding: "8px 0" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: 8,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 13,
                          fontWeight: 700,
                          color: "#0f172a",
                        }}
                      >
                        {today.toLocaleDateString("en-US", {
                          month: "long",
                          year: "numeric",
                        })}
                      </span>
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(7, 1fr)",
                        textAlign: "center",
                        gap: 2,
                      }}
                    >
                      {dayNames.map((d) => (
                        <div
                          key={d}
                          style={{
                            fontSize: 10,
                            fontWeight: 600,
                            color: "#94a3b8",
                            padding: "4px 0",
                          }}
                        >
                          {d}
                        </div>
                      ))}
                      {cells.map((day, i) => (
                        <div
                          key={i}
                          style={{
                            fontSize: 12,
                            padding: "5px 0",
                            borderRadius: "50%",
                            fontWeight: day === today.getDate() ? 700 : 400,
                            background:
                              day === today.getDate()
                                ? "#1D4ED8"
                                : "transparent",
                            color:
                              day === today.getDate()
                                ? "#fff"
                                : day
                                  ? "#334155"
                                  : "transparent",
                            cursor: day ? "pointer" : "default",
                          }}
                        >
                          {day || ""}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Card 2: To Review */}
            <div className="card">
              <div className="health-header">
                <div className="card-title" style={{ marginBottom: 0 }}>
                  To Review: Completed Meetings
                </div>
              </div>
              <div className="health-row">
                <span>Meeting Pending Review</span>
                <span className="health-pct">
                  {
                    postMeetings.filter(
                      (m) => !m.status || m.status.toLowerCase() === "review",
                    ).length
                  }
                </span>
              </div>
              <div className="health-row">
                <span>Meetings Reviewed</span>
                <span className="health-pct">
                  {
                    postMeetings.filter(
                      (m) => m.status && m.status.toLowerCase() === "completed",
                    ).length
                  }
                </span>
              </div>
              <div className="health-row">
                <span>Pending CRM Approval</span>
                <span className="health-pct">
                  {postMeetings.filter((m) => m.crm_updates === 0).length}
                </span>
              </div>
            </div>

            {/* Card 3: To Prep */}
            <div className="card">
              <div className="health-header">
                <div className="card-title" style={{ marginBottom: 0 }}>
                  To Prep: Upcoming Meetings
                </div>
              </div>
              <div className="health-row">
                <span>Today</span>
                <span className="health-pct">{meetings.length} Meetings</span>
              </div>
              <div className="health-row">
                <span>Tomorrow</span>
                <span className="health-pct">0 Meetings</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Execute;
