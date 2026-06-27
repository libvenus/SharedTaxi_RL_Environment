import "../../styles/settings.css";

export default function Settings() {
  return (
    <div className="sd-wrapper">
      <div className="row g-4">
        <div className="col-12 col-lg-8">
          <div className="sd-priority-panel">
            <div className="sd-priority-header">
              <h2 className="sd-priority-title">Dead Letter Queue (3)</h2>
            </div>

            <div className="sd-actions-grid">
              <div className="sd-action-card">
                <div className="sd-action-card-header">
                  <span className="sd-action-company">quote sent</span>
                  <span className="sd-badge-due-today">3 retries</span>
                </div>
                <p className="sd-action-desc">
                  Subscriber timeout after 3 retries
                </p>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    width: "100%",
                  }}
                >
                  <span
                    style={{
                      color: "#111827",
                      fontSize: "13px",
                      fontWeight: "600",
                      cursor: "pointer",
                    }}
                  >
                    Dismiss
                  </span>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button className="ex-btn">Escalate</button>
                    <button className="sd-action-btn">Retry</button>
                  </div>
                </div>
              </div>

              <div className="sd-action-card">
                <div className="sd-action-card-header">
                  <span className="sd-action-company">approval requested</span>
                  <span className="sd-badge-due-today">3 retries</span>
                </div>
                <p className="sd-action-desc">D365 connector unreachable</p>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    width: "100%",
                  }}
                >
                  <span
                    style={{
                      color: "#111827",
                      fontSize: "13px",
                      fontWeight: "600",
                      cursor: "pointer",
                    }}
                  >
                    Dismiss
                  </span>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button className="ex-btn">Escalate</button>
                    <button className="sd-action-btn">Retry</button>
                  </div>
                </div>
              </div>

              <div className="sd-action-card">
                <div className="sd-action-card-header">
                  <span className="sd-action-company">order.processed</span>
                  <span className="sd-badge-due-today">3 retries</span>
                </div>
                <p className="sd-action-desc">D365 connector unreachable</p>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    width: "100%",
                  }}
                >
                  <span
                    style={{
                      color: "#111827",
                      fontSize: "13px",
                      fontWeight: "600",
                      cursor: "pointer",
                    }}
                  >
                    Dismiss
                  </span>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button className="ex-btn">Escalate</button>
                    <button className="sd-action-btn">Retry</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-12 col-lg-4">
          <div className="sd-q3-panel">
            <div className="sd-q3-header">
              <h2 className="sd-q3-title">Q3 Pulse</h2>
            </div>
            <p className="sd-q3-days-left">
              Configure the metrics for the quarter
            </p>
            <div className="sd-q3-metric">
              <div className="settings-form-card">
                <div className="settings-form-group">
                  <label className="settings-form-label">
                    Quota for the Quarter
                  </label>
                  <input
                    type="text"
                    className="settings-form-input"
                    placeholder="Input"
                  />
                </div>
                <div className="settings-form-group">
                  <label className="settings-form-label">
                    Total Pipeline Value for the Quarter
                  </label>
                  <input
                    type="text"
                    className="settings-form-input"
                    placeholder="Input"
                  />
                </div>
                <div style={{ display: "flex", gap: "8px" }}>
                  <button className="kp-btn-cancel">Clear</button>
                  <button className="kp-btn-update">Save</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
