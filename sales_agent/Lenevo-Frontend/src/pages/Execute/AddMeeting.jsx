import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  createMeeting,
  fetchMeetings,
  fetchPostMeetingReview,
} from "../../api/client";
import { Modal } from "react-bootstrap";
import "../../styles/execute.css";
import prepIcon from "../../assets/Group.png";
import webex from "../../assets/image 3.png";
import googleMeetLogo from "../../assets/logos_google-meet.png";
import videoicon from "../../assets/videoicon.png";
export default function AddMeeting() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    title: "",
    account: "",
    deal: "",
    date: "",
    time: "",
    duration: "30",
    attendees: "",
    meetingUrl: "",
    autoJoin: true,
    prepNotes: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [meetings, setMeetings] = useState([]);
  const [postMeetings, setPostMeetings] = useState([]);

  useEffect(() => {
    window.scrollTo(0, 0);
    fetchMeetings({ filter: "all_meetings" })
      .then((data) => {
        const mapped = (data?.meetings || data || []).map((m) => {
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
              "Webex",
            platformIcon: isGoogle ? googleMeetLogo : webex,
            title: m.title,
            company: m.organiser_name || m.account_name || "",
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
          };
        });
        setMeetings(mapped);
      })
      .catch(() => {});
    fetchPostMeetingReview()
      .then((data) => setPostMeetings(data?.meetings || data || []))
      .catch(() => {});
  }, []);

  const inferPlatform = (meetingUrl) => {
    const normalizedUrl = meetingUrl.toLowerCase();

    if (normalizedUrl.includes("teams.microsoft.com")) return "Microsoft Teams";
    if (normalizedUrl.includes("meet.google.com")) return "Google Meet";
    if (normalizedUrl.includes("webex.com")) return "Webex";
    if (normalizedUrl.includes("zoom.us")) return "Zoom";

    return meetingUrl ? "External" : "Internal";
  };

  const buildMeetingPayload = () => {
    const meetingStart = new Date(`${form.date}T${form.time}`);
    const meetingEnd = new Date(
      meetingStart.getTime() + Number(form.duration || 0) * 60000,
    );

    return {
      meeting_start_time: meetingStart.toISOString(),
      meeting_end_time: meetingEnd.toISOString(),
      platform: inferPlatform(form.meetingUrl),
      title: form.title,
      account_name: form.account,
      attendees_emails: form.attendees,
      opportunity: form.deal,
      meeting_url: form.meetingUrl,
      body: form.meetingUrl || null,
      prep_notes: form.prepNotes,
      seller_id: "055dafe7-9840-451d-8328-5f70a6326c03"
    };
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();
    setSubmitError("");
    setIsSubmitting(true);

    try {
      await createMeeting(buildMeetingPayload());
      setSuccessMessage("Meeting saved successfully");
      setShowSuccessModal(true);
    } catch (error) {
      console.error("Failed to create meeting:", error);
      setSubmitError(error.message || "Failed to create meeting.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCloseSuccess = () => {
    setShowSuccessModal(false);
    navigate("/execute");
  };

  function handleClick(meetingId) {
    navigate(`/execute/meeting/${meetingId}`);
  }
  function handleJoin(meetingId) {
    window.open(
      `https://teams.microsoft.com/l/meetup-join/${meetingId}`,
      "_blank",
    );
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
            <span className="topbar-title">Execute: Meeting Prep</span>
          </div>
        </div>

        <div className="page-sub-header" style={{ marginBottom: "4px" }}>
          <div className="page-header-row">
            <button className="at-back-btn" onClick={() => navigate(-1)}>
              ‹
            </button>
            <span className="topbar-title">New Meeting</span>
          </div>
        </div>
        <div className="content" style={{ flexDirection: "column" }}>
          <div className="am-panel am-panel-full" style={{marginBottom:"0px"}}>
            <h2 className="am-section-title">Create Meeting</h2>

            {submitError ? (
              <p
                className="am-description"
                style={{ color: "#b91c1c", marginTop: 0 }}
              >
                {submitError}
              </p>
            ) : null}

            <form className="am-form" onSubmit={handleSubmit}>
              {/* Row 1: Title | Account | Deal | Date */}
              <div className="am-grid am-grid-4">
                <div className="am-field">
                  <label className="am-label">Meeting Title</label>
                  <input
                    className="am-input"
                    type="text"
                    name="title"
                    value={form.title}
                    onChange={handleChange}
                    placeholder="e.g. QBR Alignment Call - Infosys"
                    required
                  />
                </div>
                <div className="am-field">
                  <label className="am-label">Account</label>
                  <input
                    className="am-input"
                    type="text"
                    name="account"
                    value={form.account}
                    onChange={handleChange}
                    placeholder="e.g. Infosys Limited"
                  />
                </div>
                <div className="am-field">
                  <label className="am-label">Deal / Opportunity</label>
                  <input
                    className="am-input"
                    type="text"
                    name="deal"
                    value={form.deal}
                    onChange={handleChange}
                    placeholder="e.g. Infosys Device Refresh 2025"
                  />
                </div>
                <div className="am-field">
                  <label className="am-label">Date</label>
                  <input
                    className="am-input"
                    type="date"
                    name="date"
                    value={form.date}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              {/* Row 2: Time | Duration | Attendees */}
              <div className="am-grid am-grid-3">
                <div className="am-field">
                  <label className="am-label">Time</label>
                  <input
                    className="am-input"
                    type="time"
                    name="time"
                    value={form.time}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="am-field">
                  <label className="am-label">Duration (minutes)</label>
                  <select
                    className="am-input"
                    name="duration"
                    value={form.duration}
                    onChange={handleChange}
                  >
                    <option value="15">15</option>
                    <option value="30">30</option>
                    <option value="45">45</option>
                    <option value="60">60</option>
                    <option value="90">90</option>
                    <option value="120">120</option>
                  </select>
                </div>
                <div className="am-field">
                  <label className="am-label">
                    Attendees (comma separated)
                  </label>
                  <input
                    className="am-input"
                    type="text"
                    name="attendees"
                    value={form.attendees}
                    onChange={handleChange}
                    placeholder="e.g. Arjun Shah, Priya Nair"
                  />
                </div>
              </div>

              {/* Meeting URL */}
              <div className="am-grid am-grid-url">
                <div className="am-field">
                  <label className="am-label">Meeting URL (optional)</label>
                  <input
                    className="am-input"
                    type="url"
                    name="meetingUrl"
                    value={form.meetingUrl}
                    onChange={handleChange}
                    placeholder="https://teams.microsoft.com/l/meetup-join/..."
                  />
                  <div className="am-info-note">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                      <rect
                        x="2"
                        y="5"
                        width="14"
                        height="14"
                        rx="2"
                        stroke="#4f46e5"
                        strokeWidth="1.8"
                        fill="none"
                      />
                      <path
                        d="M16 9.5l5-3v11l-5-3"
                        stroke="#4f46e5"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        fill="none"
                      />
                    </svg>
                    On-call agent auto-joins when URL is provided and toggle is
                    enabled.
                  </div>
                </div>
              </div>

              {/* Checkbox */}
              <div className="am-checkbox-row">
                <input
                  type="checkbox"
                  id="auto-join"
                  name="autoJoin"
                  checked={form.autoJoin}
                  onChange={handleChange}
                />
                <label htmlFor="auto-join">
                  Enable on-call agent to automatically join this meeting if URL
                  is provided
                </label>
              </div>

              {/* Prep notes */}
              <div className="am-field">
                <label className="am-label">Prep notes (optional)</label>
                <textarea
                  className="am-input am-textarea"
                  name="prepNotes"
                  value={form.prepNotes}
                  onChange={handleChange}
                  placeholder="Objective, context, risks, expected next step..."
                  rows={5}
                />
              </div>

              <div
                className="am-actions"
                style={{ display: "flex", gap: 8, marginTop: 16 }}
              >
                <button
                  type="button"
                  className="am-btn-cancel"
                  onClick={() => navigate(-1)}
                  disabled={isSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="am-btn-save"
                  disabled={isSubmitting}
                >
                  {isSubmitting ? "Creating..." : "Create"}
                </button>
              </div>
            </form>
          </div>

          <div className="" style={{ padding: "0", marginTop: 8 }}>
            <div className="content" style={{ padding: 0 }}>
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

                  {meetings.map((meeting, index) => (
                    <div className="ex-meeting-card" key={index}>
                      <div className="ex-meeting-layout">
                        <div className="ex-meeting-time-block">
                          <div className="ex-start-time">
                            {meeting.startTime}
                          </div>
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
                            <div
                              className="sd-avatar-group"
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 4,
                              }}
                            >
                              {meeting.avatars.map((avatar, i) => (
                                <div
                                  key={i}
                                  className="sd-avatar sd-avatar-initials-ra"
                                  title={meeting.attendees[i]}
                                >
                                  {avatar.text}
                                </div>
                              ))}
                            </div>
                            <span
                              style={{
                                fontSize: 12,
                                fontWeight: 700,
                                color: "#0f172a",
                              }}
                            >
                              {meeting.company}
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
                            onClick={() => handleJoin(meeting.meeting_id)}
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

                  {/* Post-Meeting Review items */}
                  {postMeetings.map((meeting) => {
                    const platform = meeting.meeting_platform || "";
                    const isGoogle = platform.toLowerCase().includes("google");
                    const pIcon = isGoogle ? googleMeetLogo : webex;
                    const platformLabel =
                      platform.replace(/^(Internal|External)\s*/i, "").trim() ||
                      "Webex";
                    const typeLabel = platform
                      .toLowerCase()
                      .includes("external")
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
                                  background: isCompleted
                                    ? "#ECFDF5"
                                    : "#FEF2F2",
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
                          (m) =>
                            !m.status || m.status.toLowerCase() === "review",
                        ).length
                      }
                    </span>
                  </div>
                  <div className="health-row">
                    <span>Meetings Reviewed</span>
                    <span className="health-pct">
                      {
                        postMeetings.filter(
                          (m) =>
                            m.status && m.status.toLowerCase() === "completed",
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
                    <span className="health-pct">
                      {meetings.length} Meetings
                    </span>
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
      </div>

      <Modal
        show={showSuccessModal}
        backdrop="static"
        keyboard={false}
        centered
      >
        <Modal.Body className="am-success-modal-body">
          <i className="ri-checkbox-circle-fill am-success-icon"></i>
          <span>{successMessage}</span>
        </Modal.Body>

        <Modal.Footer className="am-success-modal-footer">
          <button className="am-btn-save" onClick={handleCloseSuccess}>
            OK
          </button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}
