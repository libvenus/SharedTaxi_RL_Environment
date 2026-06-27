export default function Summary({ id }) {
  return (
    <>
      <div className="col-main">

        <div className="card">
          <div className="card-title">Account Summarys</div>
          <div className="deal-inner">
            <div className="deal-inner-header">
              <div className="deal-inner-title">Infosys Limited</div>
              <div className="deal-close">Last Activity: 2 days ago</div>
            </div>
            <div className="deal-id">Account ID: {id}</div>
            <div className="deal-desc">Enterprise technology company with multiple ongoing opportunities across APAC region.</div>
            <div className="deal-meta">
              <span>Owner: Amit Ranjan</span><span>Region: APAC</span><span>Segment: Enterprise</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="actions-header">
            <div className="card-title" style={{ marginBottom: 0 }}>Recommended Actions</div>
            <div className="sd-priority-meta">27 total actions · <span className="sd-priority-due-today">3 due today</span></div>
          </div>
          <div className="action-item">
            <div className="action-item-header">
              <div className="action-item-title">Follow up on proposal</div>
              <div className="due-badge"><span className="due-dot"></span>Due Today</div>
            </div>
            <div className="action-item-desc">Re-engage stakeholders — 18 days without response</div>
            <button className="sd-action-btn">Action</button>
          </div>
        </div>
      </div>

      <div className="col-side">
        <div className="card account-card">
          <div className="account-header">
            <div className="account-logo">🏢</div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700 }}>Infosys Limited</div>
              <div className="account-meta">IT Services</div>
              <div className="account-meta-line">APAC · Enterprise</div>
            </div>
          </div>
          <div className="account-grid">
            <div className="account-field"><div className="field-label">Total Value</div><div className="field-value">$2.4M</div></div>
            <div className="account-field"><div className="field-label">Open Opps</div><div className="field-value">8</div></div>
            <div className="account-field full"><div className="field-label">Competitors</div><div className="field-value">Dell, HP, Samsung</div></div>
          </div>
        </div>
        <div className="card">
          <div className="health-header">
            <div className="card-title" style={{ marginBottom: 0 }}>Account Health</div>
            <div className="health-badge">✅ Healthy</div>
          </div>
          <div className="health-date">Last Activity: 2 days ago</div>
          <div className="health-main"><div className="health-main-label">Account KPI</div><div className="health-main-val">92%</div></div>
          <div className="health-row"><span>Engagement Score</span><span className="health-pct">85%</span></div>
          <div className="health-row"><span>Activity Freshness</span><span className="health-pct">90%</span></div>
        </div>
      </div>
    </>
  );
}
