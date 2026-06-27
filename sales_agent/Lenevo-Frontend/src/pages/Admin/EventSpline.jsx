import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../../styles/execute.css";
import "../../styles/admin.css";

export default function EventSpline() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("event-spline");

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
            <span className="topbar-title">Admin: Enterprise Event Spine Monitor</span>
          </div>
          <div className="tabs-outer">
            <div className="tabs">
              <button
                className={`tab ${activeTab === "event-spline" ? "active" : ""}`}
                onClick={() => setActiveTab("event-spline")}
              >
                Event Spline
              </button>
              <button
                className={`tab ${activeTab === "sales-operating-model" ? "active" : ""}`}
                onClick={() => navigate("/sales-operating-model")}
              >
                Sales Operating Model
              </button>
            </div>
          </div>
        </div>

        <div className="content">
          {activeTab === "event-spline" && (
            <div className="admin-col-left" style={{ width: "100%" }}>
              <div className="admin-summary-row">
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Sources Online</span>
                    <span className="admin-scard-value admin-scard-value-green">4/4</span>
                  </div>
                </div>
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Delivery SLA</span>
                    <span className="admin-scard-value admin-scard-value-green">99.2%</span>
                  </div>
                </div>
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Dead Letters</span>
                    <span className="admin-scard-value admin-scard-value-red">3</span>
                  </div>
                </div>
                <div className="admin-summary-col"></div>
              </div>

              <div className="admin-filter-bar">
                <div className="admin-search-box">
                  <svg className="admin-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="11" cy="11" r="7"></circle>
                    <line x1="16.5" y1="16.5" x2="21" y2="21"></line>
                  </svg>
                  <input placeholder="Search events..." />
                </div>
                <div className="admin-dd-wrap">
                  <div className="admin-dd-trigger">
                    <span>Event Type</span>
                    <svg className="admin-dd-arrow" viewBox="0 0 10 10">
                      <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path>
                    </svg>
                  </div>
                </div>
                <div className="admin-dd-wrap">
                  <div className="admin-dd-trigger">
                    <span>Source</span>
                    <svg className="admin-dd-arrow" viewBox="0 0 10 10">
                      <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path>
                    </svg>
                  </div>
                </div>
                <div className="admin-dd-wrap">
                  <div className="admin-dd-trigger">
                    <span>Attempts</span>
                    <svg className="admin-dd-arrow" viewBox="0 0 10 10">
                      <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path>
                    </svg>
                  </div>
                </div>
                <div className="admin-dd-wrap">
                  <div className="admin-dd-trigger">
                    <span>Domain</span>
                    <svg className="admin-dd-arrow" viewBox="0 0 10 10">
                      <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path>
                    </svg>
                  </div>
                </div>
              </div>

              <div className="row w-100 g-3">
                <div className="col-12 col-lg-8">
                  <div className="admin-table-card">
                    <div className="admin-table-outer">

                      {/* frozen left */}
                      <div className="admin-frozen-panel">
                        <table>
                          <colgroup><col style={{ width: 220 }} /></colgroup>
                          <thead style={{ height: 48 }}>
                            <tr>
                              <th className="admin-th"><div className="admin-th-inner">Event <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div></th>
                            </tr>
                            <tr className="admin-spacer-row"><td colSpan={1} /></tr>
                          </thead>
                          <tbody>
                            <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">email.received</div><div className="admin-event-id">EVT-2024-8821</div></td></tr>
                            <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">approval.rejected</div><div className="admin-event-id">EVT-2024-8820</div></td></tr>
                            <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">meeting.ended</div><div className="admin-event-id">EVT-2024-8819</div></td></tr>
                            <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">file.uploaded</div><div className="admin-event-id">EVT-2024-8818</div></td></tr>
                            <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">approval.approved</div><div className="admin-event-id">EVT-2024-8817</div></td></tr>
                            <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">invoice.generated</div><div className="admin-event-id">EVT-2024-8816</div></td></tr>
                            <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">email.received</div><div className="admin-event-id">EVT-2024-8815</div></td></tr>
                            <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">deal.stage_change</div><div className="admin-event-id">EVT-2024-8814</div></td></tr>
                            <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">contact.updated</div><div className="admin-event-id">EVT-2024-8813</div></td></tr>
                            <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">quote.sent</div><div className="admin-event-id">EVT-2024-8812</div></td></tr>
                          </tbody>
                        </table>
                      </div>

                      {/* scrollable right */}
                      <div className="admin-scroll-panel">
                        <table>
                          <thead>
                            <tr>
                              <th className="admin-th" style={{ minWidth: 120 }}>
                                <div className="admin-th-inner">Source <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                              </th>
                              <th className="admin-th" style={{ minWidth: 110 }}>
                                <div className="admin-th-inner">Account <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                              </th>
                              <th className="admin-th" style={{ minWidth: 110 }}>
                                <div className="admin-th-inner">Timestamp <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                              </th>
                              <th className="admin-th" style={{ minWidth: 110 }}>
                                <div className="admin-th-inner">Status <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                              </th>
                              <th className="admin-th" style={{ minWidth: 250 }}>
                                <div className="admin-th-inner">Error Details <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                              </th>
                              <th className="admin-th" style={{ minWidth: 80 }}>Size</th>
                            </tr>
                            <tr className="admin-spacer-row"><td colSpan={6} /></tr>
                          </thead>
                          <tbody>
                            <tr className="admin-row-even">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">12/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-success">Delivered</span></td>
                              <td className="admin-td"><div className="admin-error-text">Subscriber timeout after 3 retries</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-odd">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">12/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-success">Delivered</span></td>
                              <td className="admin-td"><div className="admin-error-text">Subscriber timeout after 3 retries</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-even">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">12/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-success">Delivered</span></td>
                              <td className="admin-td"><div className="admin-error-text">Subscriber timeout after 3 retries</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-odd">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">11/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-success">Delivered</span></td>
                              <td className="admin-td"><div className="admin-error-text">D365 connector unreachable</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-even">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">11/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-success">Delivered</span></td>
                              <td className="admin-td"><div className="admin-error-text">D365 connector unreachable</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-odd">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">11/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-warning">Retried</span></td>
                              <td className="admin-td"><div className="admin-error-text">CRM write-back service could not be contacted</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-even">
                              <td className="admin-td">M365 Graph</td>
                              <td className="admin-td">Infosys</td>
                              <td className="admin-td">10/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-error">Failed</span></td>
                              <td className="admin-td"><div className="admin-error-text">CRM write-back service could not be contacted</div></td>
                              <td className="admin-td admin-size-text">2.1 KB</td>
                            </tr>
                            <tr className="admin-row-odd">
                              <td className="admin-td">Salesforce</td>
                              <td className="admin-td">TCS</td>
                              <td className="admin-td">10/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-warning">Retried</span></td>
                              <td className="admin-td"><div className="admin-error-text">CRM write-back service could not be contacted</div></td>
                              <td className="admin-td admin-size-text">1.8 KB</td>
                            </tr>
                            <tr className="admin-row-even">
                              <td className="admin-td">HubSpot</td>
                              <td className="admin-td">Wipro</td>
                              <td className="admin-td">09/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-success">Delivered</span></td>
                              <td className="admin-td"><div className="admin-error-text">—</div></td>
                              <td className="admin-td admin-size-text">1.2 KB</td>
                            </tr>
                            <tr className="admin-row-odd">
                              <td className="admin-td">D365</td>
                              <td className="admin-td">Acme Corp</td>
                              <td className="admin-td">09/06/24</td>
                              <td className="admin-td"><span className="admin-badge admin-badge-error">Failed</span></td>
                              <td className="admin-td"><div className="admin-error-text">Subscriber timeout after 3 retries</div></td>
                              <td className="admin-td admin-size-text">3.4 KB</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* table footer */}
                    <div className="admin-table-footer">
                      <span className="admin-footer-info">Showing 10 of 128 events</span>
                      <div className="admin-pag-row">
                        <span className="admin-pag-label">Show</span>
                        <button className="admin-pag-btn" disabled>
                          <i className="bi bi-chevron-left" style={{ fontSize: 10 }}></i>
                        </button>
                        <button className="admin-pag-btn admin-pag-active">1</button>
                        <button className="admin-pag-btn">2</button>
                        <button className="admin-pag-btn">3</button>
                        <span className="admin-pag-ellipsis">…</span>
                        <button className="admin-pag-btn">13</button>
                        <button className="admin-pag-btn">
                          <i className="bi bi-chevron-right" style={{ fontSize: 10 }}></i>
                        </button>
                        <span className="admin-pag-label">13 pages</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="col-12 col-lg-4">
                  <div className="admin-col-right">
                    {/* Source Summary */}
                    <div className="admin-right-panel">
                   
                      <div className="admin-stat-row">
                        <span className="admin-stat-label">Sources Online</span>
                        <span className="admin-stat-count">4/4 <span className="admin-stat-dot admin-stat-dot-green"></span></span>
                      </div>
                      <div className="admin-stat-row">
                        <span className="admin-stat-label">Delivery SLA</span>
                        <span className="admin-stat-count">99.2% <span className="admin-stat-dot admin-stat-dot-red"></span></span>
                      </div>
                      <div className="admin-stat-row">
                        <span className="admin-stat-label">Dead Letters</span>
                        <span className="admin-stat-count">3 <span className="admin-stat-dot admin-stat-dot-red"></span></span>
                      </div>
                    </div>

                    {/* Dead Letter Queue */}
                    <div className="admin-right-panel">
                      <div className="admin-right-header">
                        <h2 className="admin-right-title">Dead Letter Queue (3)</h2>
                      </div>
                      <p className="admin-panel-subtitle">Failed delivery · requires manual triage</p>

                      <div className="admin-queue-item">
                        <div className="admin-queue-header">
                          <div className="admin-queue-title">
                            <i className="bi bi-exclamation-triangle-fill" style={{ color: "#C2410C" }}></i> quote sent
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

                 
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "sales-operating-model" && (
            <div className="col-main">
              <h2 className="admin-section-title">Sales Operating Model</h2>
              <p className="admin-section-desc">Sales operating model content goes here.</p>
              <div className="admin-summary-row">
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Total Models</span>
                    <span className="admin-scard-value">6</span>
                  </div>
                </div>
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Active</span>
                    <span className="admin-scard-value">3</span>
                  </div>
                </div>
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Draft</span>
                    <span className="admin-scard-value">2</span>
                  </div>
                </div>
                <div className="admin-summary-col">
                  <div className="admin-scard">
                    <span className="admin-scard-label">Archived</span>
                    <span className="admin-scard-value">1</span>
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
