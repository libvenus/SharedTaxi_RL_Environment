import { useEffect } from "react";
import "../../styles/execute.css";
import "../../styles/admin.css";

export default function Commercial() {
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
            <span className="topbar-title">Commercial</span>
          </div>
        </div>

        <div className="content">
          <div className="admin-col-left" style={{ width: "100%" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "#DCFCE7", borderRadius: 8, padding: "10px 16px", marginBottom: 16 }}>
              <span style={{ fontSize: 13, color: "#15803D", fontWeight: 500 }}>Last synced 2 min ago – Product Name</span>
              <span className="admin-badge admin-badge-success">Active</span>
            </div>

            <div className="admin-filter-bar">
              <div className="admin-search-box" style={{ flex: 2 }}>
                <svg className="admin-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="11" cy="11" r="7"></circle>
                  <line x1="16.5" y1="16.5" x2="21" y2="21"></line>
                </svg>
                <input placeholder="Search Number..." />
              </div>
              <div className="admin-dd-wrap" style={{ minWidth: 100 }}>
                <div className="admin-dd-trigger">
                  <span>Product</span>
                  <svg className="admin-dd-arrow" viewBox="0 0 10 10">
                    <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path>
                  </svg>
                </div>
              </div>
              <div className="admin-dd-wrap" style={{ minWidth: 100 }}>
                <div className="admin-dd-trigger">
                  <span>Status</span>
                  <svg className="admin-dd-arrow" viewBox="0 0 10 10">
                    <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path>
                  </svg>
                </div>
              </div>
            </div>

            <div className="row w-100 g-3">
              <div className="col-12">
                <div className="admin-table-card">
                  <div className="admin-table-outer">

                    {/* frozen left */}
                    <div className="admin-frozen-panel">
                      <table>
                        <colgroup><col style={{ width: 220 }} /></colgroup>
                        <thead style={{ height: 48 }}>
                          <tr>
                            <th className="admin-th"><div className="admin-th-inner">Product Name <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div></th>
                          </tr>
                          <tr className="admin-spacer-row"><td colSpan={1} /></tr>
                        </thead>
                        <tbody>
                          <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkPad X1 Carbon</div></td></tr>
                          <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkStation P360</div></td></tr>
                          <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkVision T27h</div></td></tr>
                          <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">IdeaPad Flex 5</div></td></tr>
                          <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkPad L14 Gen 4</div></td></tr>
                          <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkCentre M90q</div></td></tr>
                          <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">Legion Pro 7i</div></td></tr>
                          <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkPad T16 Gen 2</div></td></tr>
                          <tr className="admin-row-even"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkBook 16p Gen 4</div></td></tr>
                          <tr className="admin-row-odd"><td className="admin-td admin-name-cell"><div className="admin-event-name">ThinkEdge SE50</div></td></tr>
                        </tbody>
                      </table>
                    </div>

                    {/* scrollable right */}
                    <div className="admin-scroll-panel">
                      <table>
                        <thead>
                          <tr>
                            <th className="admin-th" style={{ minWidth: 120 }}>
                              <div className="admin-th-inner">Category <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                            </th>
                            <th className="admin-th" style={{ minWidth: 110 }}>
                              <div className="admin-th-inner">Base Price <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                            </th>
                            <th className="admin-th" style={{ minWidth: 100 }}>
                              <div className="admin-th-inner">Discount <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                            </th>
                            <th className="admin-th" style={{ minWidth: 100 }}>
                              <div className="admin-th-inner">Margin <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                            </th>
                            <th className="admin-th" style={{ minWidth: 130 }}>
                              <div className="admin-th-inner">D365 Status <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                            </th>
                            <th className="admin-th" style={{ minWidth: 150 }}>
                              <div className="admin-th-inner">Qlik Validated <svg className="admin-sort-arrow" viewBox="0 0 10 10"><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"></path></svg></div>
                            </th>
                          </tr>
                          <tr className="admin-spacer-row"><td colSpan={6} /></tr>
                        </thead>
                        <tbody>
                          <tr className="admin-row-even">
                            <td className="admin-td">Laptops</td>
                            <td className="admin-td">$1,849</td>
                            <td className="admin-td">12%</td>
                            <td className="admin-td">34%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Validated</span></td>
                          </tr>
                          <tr className="admin-row-odd">
                            <td className="admin-td">Workstations</td>
                            <td className="admin-td">$2,499</td>
                            <td className="admin-td">8%</td>
                            <td className="admin-td">28%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Validated</span></td>
                          </tr>
                          <tr className="admin-row-even">
                            <td className="admin-td">Monitors</td>
                            <td className="admin-td">$549</td>
                            <td className="admin-td">15%</td>
                            <td className="admin-td">42%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-error">Pending</span></td>
                          </tr>
                          <tr className="admin-row-odd">
                            <td className="admin-td">Laptops</td>
                            <td className="admin-td">$799</td>
                            <td className="admin-td">20%</td>
                            <td className="admin-td">25%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-error">Inactive</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Validated</span></td>
                          </tr>
                          <tr className="admin-row-even">
                            <td className="admin-td">Laptops</td>
                            <td className="admin-td">$999</td>
                            <td className="admin-td">10%</td>
                            <td className="admin-td">30%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Validated</span></td>
                          </tr>
                          <tr className="admin-row-odd">
                            <td className="admin-td">Desktops</td>
                            <td className="admin-td">$1,199</td>
                            <td className="admin-td">5%</td>
                            <td className="admin-td">38%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-error">Pending</span></td>
                          </tr>
                          <tr className="admin-row-even">
                            <td className="admin-td">Gaming</td>
                            <td className="admin-td">$2,899</td>
                            <td className="admin-td">6%</td>
                            <td className="admin-td">22%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-error">Inactive</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-error">Pending</span></td>
                          </tr>
                          <tr className="admin-row-odd">
                            <td className="admin-td">Laptops</td>
                            <td className="admin-td">$1,299</td>
                            <td className="admin-td">14%</td>
                            <td className="admin-td">31%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Validated</span></td>
                          </tr>
                          <tr className="admin-row-even">
                            <td className="admin-td">Laptops</td>
                            <td className="admin-td">$1,599</td>
                            <td className="admin-td">11%</td>
                            <td className="admin-td">29%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Validated</span></td>
                          </tr>
                          <tr className="admin-row-odd">
                            <td className="admin-td">Edge/IoT</td>
                            <td className="admin-td">$699</td>
                            <td className="admin-td">3%</td>
                            <td className="admin-td">45%</td>
                            <td className="admin-td"><span className="admin-badge admin-badge-success">Active</span></td>
                            <td className="admin-td"><span className="admin-badge admin-badge-error">Pending</span></td>
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
            </div>

            {/* Recently Validated */}
            <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10, padding: 20, marginTop: 16 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 14 }}>Recently Validated</h3>
              <div style={{ display: "flex", gap: 12 }}>
                <div style={{ flex: 1, background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, padding: "12px 16px" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>ThinkPad T16 Gen 3 (Core Ultra 7, 32GB)</div>
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>2 min ago</div>
                </div>
                <div style={{ flex: 1, background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, padding: "12px 16px" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>ThinkStation P3 Ultra</div>
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>2 min ago</div>
                </div>
                <div style={{ flex: 1, background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, padding: "12px 16px" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>ThinkSystem ST650 V3</div>
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>2 min ago</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
