import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../../styles/execute.css";
import "../../styles/admin.css";

export default function AdminSettings() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("dead-letters");

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="dv-page">
      <div className="main">
        <div className="quarter-bar">
          Q2 FY2024 · Week 10 of 12 ·{" "}
          <span className="closure-badge">Closure Phase</span>
        </div>

        <div className="page-header">
          <div className="page-header-row">
            <span className="topbar-title">Admin Settings</span>
          </div>
          <div className="tabs-outer">
            <div className="tabs">
              <button
                className={`tab ${activeTab === "dead-letters" ? "active" : ""}`}
                onClick={() => setActiveTab("dead-letters")}
              >
                Dead Letters
              </button>
              <button
                className={`tab ${activeTab === "quarter-pulse" ? "active" : ""}`}
                onClick={() => navigate("/quarter-pulse")}
              >
                Quarter Pulse
              </button>
            </div>
          </div>
        </div>

        <div className="content">
          {activeTab === "dead-letters" && (
            <div className="admin-col-left" style={{ width: "100%" }}>
           

              <div className="row w-100 g-3">
                <div className="col-12">
                  <div className="admin-right-panel">
                    <div className="admin-right-header">
                      <h2 className="admin-right-title">Dead Letter Queue</h2>
                    </div>
                    <p className="admin-panel-subtitle">Failed delivery · requires manual triage</p>

                    <div className="row g-3">
                      <div className="col-12 col-lg-4">
                        <div className="admin-queue-item">
                          <div className="admin-queue-header">
                            <div className="admin-queue-title">
                              <i className="bi bi-exclamation-triangle-fill" style={{ color: "#C2410C" }}></i> quote.sent
                            </div>
                            <span className="admin-badge admin-badge-error1">3 retries</span>
                          </div>
                          <div className="admin-queue-desc">Subscriber timeout after 3 retries</div>
                          <div className="admin-queue-actions">
                            <button className="admin-btn-dismiss">Dismiss</button>
                            <button className="admin-btn-escalate">Escalate</button>
                            <button className="admin-btn-retry">Retry</button>
                          </div>
                        </div>
                      </div>

                      <div className="col-12 col-lg-4">
                        <div className="admin-queue-item">
                          <div className="admin-queue-header">
                            <div className="admin-queue-title">
                              <i className="bi bi-exclamation-triangle-fill" style={{ color: "#C2410C" }}></i> approval.requested
                            </div>
                            <span className="admin-badge admin-badge-error1">3 retries</span>
                          </div>
                          <div className="admin-queue-desc">D365 connector unreachable</div>
                          <div className="admin-queue-actions">
                            <button className="admin-btn-dismiss">Dismiss</button>
                            <button className="admin-btn-escalate">Escalate</button>
                            <button className="admin-btn-retry">Retry</button>
                          </div>
                        </div>
                      </div>

                      <div className="col-12 col-lg-4">
                        <div className="admin-queue-item">
                          <div className="admin-queue-header">
                            <div className="admin-queue-title">
                              <i className="bi bi-exclamation-triangle-fill" style={{ color: "#C2410C" }}></i> invoice.generated
                            </div>
                            <span className="admin-badge admin-badge-error1">5 retries</span>
                          </div>
                          <div className="admin-queue-desc">CRM write-back service could not be contacted</div>
                          <div className="admin-queue-actions">
                            <button className="admin-btn-dismiss">Dismiss</button>
                            <button className="admin-btn-escalate">Escalate</button>
                            <button className="admin-btn-retry">Retry</button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
