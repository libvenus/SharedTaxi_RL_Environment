import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import avatar1 from "../../assets/avatar1.jpg";
import webex from "../../assets/image 3.png";
import googleMeetLogo from "../../assets/logos_google-meet.png";
import { fetchPostMeetingReview } from "../../api/client";

function PostMeetingReview() {
  const navigate = useNavigate();
  const [meetings, setMeetings] = useState([]);

  useEffect(() => {
    fetchPostMeetingReview()
      .then((data) => setMeetings(data))
      .catch((err) => console.error("Failed to fetch post-meeting data:", err));
  }, []);

  const goToKeypointsPage = useCallback(
    (meetingId, e, viewOnly = false) => {
      if (e) {
        e.preventDefault();
        e.stopPropagation();
      }
      navigate(`/execute/postmeeting/${meetingId}${viewOnly ? "?viewOnly=true" : ""}`);
    },
    [navigate]
  );

  const handleCardKeyDown = useCallback(
    (meetingId, e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        navigate(`/execute/postmeeting/${meetingId}`);
      }
    },
    [navigate]
  );

  const formatTime = (dateStr) => {
    const d = new Date(dateStr);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: true });
  };

  const formatDate = (dateStr) => {
    const d = new Date(dateStr);
    return d.toLocaleDateString("en-GB", { day: "2-digit", month: "2-digit", year: "2-digit" });
  };

  const needsReview = meetings.filter((m) => !m.status || m.status.toLowerCase() === "review");
  const completed = meetings.filter((m) => m.status && m.status.toLowerCase() === "completed");

  return (
    <div className="col-main">
      <div className="mr-stats-row">
        <div className="mr-stat-card">
          <span className="label">Meeting Pending Review</span>
          <span className="value">{needsReview.length}</span>
        </div>
        <div className="mr-stat-card">
          <span className="label">Meetings Reviewed</span>
          <span className="value">{completed.length}</span>
        </div>
        <div className="mr-stat-card">
          <span className="label">Pending CRM Approval</span>
          <span className="value">{meetings.filter((m) => m.crm_updates === 0).length}</span>
        </div>
        <div className="mr-stat-card">
          <span className="label">Next Steps Logged</span>
          <span className="value">{meetings.reduce((sum, m) => sum + (m.next_steps_count || 0), 0)}</span>
        </div>
      </div>

      <div className="mr-panel">
        <div className="mr-panel-title">Needs Review</div>

        {needsReview.map((meeting) => {
          const platform = meeting.meeting_platform || "";
          const isGoogle = platform.toLowerCase().includes("google");
          const platformIcon = isGoogle ? googleMeetLogo : webex;
          const platformLabel = platform.replace(/^(Internal|External)\s*/i, "").trim() || "Webex";
          const typeLabel = platform.toLowerCase().includes("external") ? "External" : "Internal";

          return (
            <div
              key={meeting.meeting_id}
              className="mr-meeting-card"
              role="button"
              tabIndex={0}
              style={{ cursor: "pointer" }}
              onClick={(e) => goToKeypointsPage(meeting.meeting_id, e)}
              onKeyDown={(e) => handleCardKeyDown(meeting.meeting_id, e)}
            >
              <div className="mr-meeting-time">
                <div className="time">
                  {formatTime(meeting.meeting_start_time)}
                </div>
                <div className="date">{formatDate(meeting.meeting_start_time)}</div>
              </div>
              <div className="mr-meeting-info">
                <div className="mr-meeting-meta">
                  <img alt={platformLabel} className="sd-platform-icon sd-platform-meet" src={platformIcon} />
                  <span className="mr-meta-text">{typeLabel} · {platformLabel}</span>
                  <div className="sd-meeting-avatars mb-0">
                    <div className="sd-avatar-group">
                      {(meeting.attendees || []).map((a, i) => (
                        <div key={i} className="sd-avatar sd-avatar-initials-ra">
                          {a.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
                        </div>
                      ))}
                    </div>
                  </div>
                  {meeting.meeting_type && (
                    <span className={`mr-badge mr-badge-${meeting.meeting_type.toLowerCase()}`}>
                      {meeting.meeting_type}
                    </span>
                  )}
                </div>
                <div className="mr-title-row">
                  <span className="mr-meeting-name">{meeting.title}</span>
                  <span className="mr-company">{meeting.organiser_name || ""}</span>
                </div>
                <div className="mr-tags">
                  <span className="mr-tag mr-tag-crm-pending">{meeting.crm_updates} CRM Updates</span>
                  <span className="mr-tag mr-tag-keypoints">{meeting.key_points_count} Key Points</span>
                  <span className="mr-tag mr-tag-nextsteps">{meeting.next_steps_count} Next Steps</span>
                  {meeting.customer_sentiment && (
                    <span className={`mr-tag mr-tag-${meeting.customer_sentiment}`}>
                      {meeting.customer_sentiment.charAt(0).toUpperCase() + meeting.customer_sentiment.slice(1)}
                    </span>
                  )}
                </div>
              </div>
              <div className="mr-meeting-action">
                <button type="button" className="mr-btn-primary" onClick={(e) => goToKeypointsPage(meeting.meeting_id, e)}>
                  Review &amp; Approve
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {completed.length > 0 && (
        <div className="mr-panel">
          <div className="mr-panel-title">Completed</div>
          {completed.map((meeting) => {
            const platform = meeting.meeting_platform || "";
            const isGoogle = platform.toLowerCase().includes("google");
            const platformIcon = isGoogle ? googleMeetLogo : webex;
            const platformLabel = platform.replace(/^(Internal|External)\s*/i, "").trim() || "Webex";
            const typeLabel = platform.toLowerCase().includes("external") ? "External" : "Internal";

            return (
              <div
                key={meeting.meeting_id}
                className="mr-meeting-card"
                role="button"
                tabIndex={0}
                style={{ cursor: "pointer" }}
                onClick={(e) => goToKeypointsPage(meeting.meeting_id, e)}
                onKeyDown={(e) => handleCardKeyDown(meeting.meeting_id, e)}
              >
                <div className="mr-meeting-time">
                  <div className="time">
                    {formatTime(meeting.meeting_start_time)}
                  </div>
                  <div className="date">{formatDate(meeting.meeting_start_time)}</div>
                </div>
                <div className="mr-meeting-info">
                  <div className="mr-meeting-meta">
                    <img alt={platformLabel} className="sd-platform-icon sd-platform-meet" src={platformIcon} />
                    <span className="mr-meta-text">{typeLabel} · {platformLabel}</span>
                    <div className="sd-meeting-avatars mb-0">
                      <div className="sd-avatar-group">
                        {(meeting.attendees || []).map((a, i) => (
                          <div key={i} className="sd-avatar sd-avatar-initials-ra">
                            {a.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
                          </div>
                        ))}
                      </div>
                    </div>
                    {meeting.meeting_type && (
                      <span className={`mr-badge mr-badge-${meeting.meeting_type.toLowerCase()}`}>
                        {meeting.meeting_type}
                      </span>
                    )}
                  </div>
                  <div className="mr-title-row">
                    <span className="mr-meeting-name">{meeting.title}</span>
                    <span className="mr-company">{meeting.organiser_name || ""}</span>
                  </div>
                  <div className="mr-tags">
                    <span className="mr-tag mr-tag-crm-done">CRM Up to Date</span>
                    <span className="mr-tag mr-tag-keypoints">{meeting.key_points_count} Key Points</span>
                    <span className="mr-tag mr-tag-nextsteps">{meeting.next_steps_count} Next Steps</span>
                    {meeting.customer_sentiment && (
                      <span className={`mr-tag mr-tag-${meeting.customer_sentiment}`}>
                        {meeting.customer_sentiment.charAt(0).toUpperCase() + meeting.customer_sentiment.slice(1)}
                      </span>
                    )}
                  </div>
                </div>
                <div className="mr-meeting-action">
                  <button type="button" className="mr-btn-dark" onClick={(e) => goToKeypointsPage(meeting.meeting_id, e, true)}>
                    View Summary
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default PostMeetingReview;
