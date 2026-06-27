import { useState } from "react";

export default function Templates() {
  const [showForm, setShowForm] = useState(false);

  const templates = [
    {
      id: 1,
      name: "General Follow Up",
      description: "Sent to maintain engagement with prospects or customers",
      badges: ["Outreach", "SMB", "Prospecting"],
    },
    {
      id: 2,
      name: "Follow-Up After Meeting",
      description: "Post-meeting summary with next steps and recap.",
      badges: ["follow-up", "post-meeting"],
    },
    {
      id: 3,
      name: "Reengagemen",
      description: "Sent when a previously engaged prospect/customer has gone inactive or communication has stalled, to restart the conversation and encourage further engagement",
      badges: ["SMB", "commercial"],
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {showForm && (
        <div className="ct-new-form">
          <div className="ct-form-header">
            <div className="ct-form-title">Configure Template</div>
            <button className="ct-form-close" onClick={() => setShowForm(false)}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div className="ct-fields">
            <div className="ct-field">
              <label>Subject Line Style</label>
              <input type="text" placeholder="Enter subject line style" />
            </div>
            <div className="ct-field">
              <label>Tone (formal / consultative)</label>
              <select>
                <option value="">Select</option>
                <option value="formal">Formal</option>
                <option value="consultative">Consultative</option>
              </select>
            </div>
            <div className="ct-field">
              <label>CTA Type</label>
              <input type="text" placeholder="Enter CTA type" />
            </div>
            <div className="ct-field">
              <label>Personalization Depth</label>
              <select>
                <option value="">Select</option>
                <option value="light">Light</option>
                <option value="medium">Medium</option>
                <option value="deep">Deep</option>
              </select>
            </div>
            <div className="ct-field">
              <label>Compliance Footer</label>
              <input type="text" placeholder="Enter compliance footer" />
            </div>
          </div>
          <div className="ct-form-actions">
            <button className="ct-btn-cancel" onClick={() => setShowForm(false)}>Cancel</button>
            <button className="ct-btn-save">Save</button>
          </div>
        </div>
      )}

      <div className="library-grid">
        {templates.map((tpl) => (
          <div className="library-card" key={tpl.id} style={{ flexDirection: "column", alignItems: "stretch" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div className="library-card-icon">
                <i className="ri-file-copy-line"></i>
              </div>
              <div className="library-card-info" style={{ flex: 1 }}>
                <div className="library-card-name">{tpl.name}</div>
                <div className="library-card-desc">{tpl.description}</div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingLeft: 56, marginTop: 10 }}>
              {tpl.badges?.length > 0 && (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {tpl.badges.map((badge, i) => (
                    <span className="library-card-badge" key={i}>{badge}</span>
                  ))}
                </div>
              )}
              <button
                className="sd-action-btn"
                style={{ fontSize: 12, padding: "6px 16px", marginLeft: "auto" }}
                onClick={() => setShowForm(true)}
              >
                Configure
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
