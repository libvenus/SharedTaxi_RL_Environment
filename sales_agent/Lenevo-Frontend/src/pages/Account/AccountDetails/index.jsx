import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Opportunities from "./Opportunities";
import { fetchAccountDetails } from "../../../api/client";
import CustomerInformation from "./CustomerInformation";
import Contacts from "./Contacts";
import Notes from "./Notes";
// import Summary from "./Summary";
import accounticon from "../../../assets/icons/account_icon.png";
import buildingicon from "../../../assets/icons/building_icon.png";
import {
  formatCurrencyShort,
  formatDateMMDDYY,
} from "../../../utils/format";
import "../../../styles/account.css";
import "../../../styles/opportunity.css";
export default function AccountDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTopTab, setActiveTopTab] = useState("summary");
  const [accountDetails, setAccountDetails] = useState({});

  const renderTabContent = () => {
    switch (activeTopTab) {
      case "opportunities":
        return <Opportunities />;
      case "customer-info":
        return <CustomerInformation />;
      case "contacts":
        return <Contacts />;
      case "notes":
        return <Notes />;
      default:
        return renderSummary();
    }
  };
  useEffect(() => {
    fetchAccountDetails(id).then((response) => {
      setAccountDetails(response);
    });
  }, [id]);
  const formatNumber = (value) => {
    const num = Number(value);

    if (num >= 1000) {
      return `${(num / 1000).toFixed(1).replace(/\.0$/, "")}K`;
    }

    return num.toString();
  };

  formatNumber(177000); // 177K
  formatNumber(177500); // 177.5K
  const renderSummary = () => (
    <>
      <div className="col-main">
        <div className="card">
          <div className="ad-summary-header">
            <span>Account Summary</span>
            <span className="ad-summary-amount">$544k</span>
          </div>

          <div className="deal-inner">
            <div className="ad-summary-content">
              <img src={accounticon} alt="Account" className="deal-image" />

              <div className="deal-desc">
                Infosys is Lenovo's 3rd largest enterprise account in South Asia
                with 340,000+ employees. Key decision maker Kiran Rao has
                completed two prior Lenovo purchases. The account has 3 open
                opportunities and an annual revenue potential of $850K. Strong
                executive relationship — Priya met the CTO at Lenovo Transform
                last quarter.
              </div>
            </div>
          </div>

          <div className="deal-meta text-center">
            <p>
  AI disclaimer information goes here with <a href="#" style={{ textDecoration: "none" }}>links</a> and resources
              available.
            </p>
          </div>
        </div>

        <div className="card">
          <div className="actions-header">
            <img src={accounticon} alt="Account" className="deal-image" />
            <div className="card-title" style={{ marginBottom: 0 }}>
              Recommended Actions
            </div>
            <div className="sd-priority-meta">
              4 total actions ·{" "}
              <span className="sd-priority-due-today">2 due today</span>
            </div>
          </div>

          <div className="deal-inner mb-4">
            <div className="actions-headers">
              <div className="card-title" style={{ marginBottom: 0 }}>

                Send Remainder
              </div>
              <div className="sd-priority-meta">
                <span className="sd-priority-due-today">Due today</span>
              </div>
            </div>
            <div className="ad-action-info">
              <div className="ad-action-title">Think Pad Fleet Refresh <span className="ad-action-amount">$420k</span></div>
            </div>
            <div className="deal-content">
              <div className="deal-desc">
                Kiran Rao was sent a proposal email 5 days ago but has not opened it. There's been no read receipt, no reply, and no follow-up from their side. The deal is sitting idle, and the window to act is narrowing.
              </div>
            </div>
            <div className="ad-action-buttons">
              <button className="ad-btn-secondary">Complete Task</button>
              <button className="ad-btn-gradient">
                <img src={accounticon} alt="Account" className="deal-image-button" />
                Send Remainder</button>
            </div>
          </div>
          <div className="deal-inner">
            <div className="actions-headers">
              <div className="card-title" style={{ marginBottom: 0 }}>
                Confirmation Needed
              </div>
              <div className="sd-priority-meta">
                <span className="sd-priority-due-today">Due today</span>
              </div>
            </div>
            <div className="ad-action-info">
              <div className="ad-action-title">Think Pad Fleet Refresh <span className="ad-action-amount">$420k</span></div>
            </div>
            <div className="deal-content">
              <div className="deal-desc">
                Kiran Rao was sent a proposal email 5 days ago but has not opened it. There's been no read receipt, no reply, and no follow-up from their side. The deal is sitting idle, and the window to act is narrowing.
              </div>
            </div>
            <div className="ad-action-buttons">
              <button className="ad-btn-secondary">Complete Task</button>
              <button className="ad-btn-gradient">
                <img src={accounticon} alt="Account" className="deal-image-button" />
                Send Remainder</button>
            </div>
          </div>
        </div>
      </div>

      <div className="col-side">
        <div className="card account-card">
          <div style={{ fontSize: 15, fontWeight: 700 }} className="mb-2">
            {accountDetails?.name || "-"}
          </div>
          <div className="account-header">
            <img src={buildingicon} alt="Account" className="account-image" />
            <div>
              <div className="account-meta">{accountDetails?.industry || "-"}</div>
              <div className="account-meta-line">{accountDetails?.region || "-"}</div>
            </div>
          </div>
          <div className="account-grid">
            <div >
              <div className="field-label">Revenue Potential</div>
              <div className="field-value">{formatCurrencyShort(accountDetails?.revenue || "-")} year</div>
            </div>
            <div >
              <div className="field-label">Open Opps</div>
              <div className="field-value">{accountDetails?.activeOpportunitiesCount || "-"}</div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="health-header">
            <div className="card-title" style={{ marginBottom: 0 }}>
              Basic Information
            </div>
          </div>

          <div className="health-row">
            <span>Region</span>
            <span className="health-pct">{accountDetails?.region || "-"}</span>
          </div>

          <div className="health-row">
            <span>Industry</span>
            <span className="health-pct">
              {accountDetails?.industry || "-"}
            </span>
          </div>

          <div className="health-row">
            <span>Employee Count</span>
            <span className="health-pct">
              {formatNumber(accountDetails?.employeeCount || "-")}
            </span>
          </div>

          <div className="health-row">
            <span>Annual Revenue</span>
            <span className="health-pct">{formatCurrencyShort(accountDetails?.revenue || "-")}</span>
          </div>

          <div className="health-row">
            <span>Segment</span>
            <span className="health-pct">{accountDetails?.segment || "-"}</span>
          </div>

          <div className="health-row">
            <span>Sub-Segment</span>
            <span className="health-pct">{"-"}</span>
          </div>

          <div className="health-row">
            <span>Lead Origin</span>
            <span className="health-pct">{"-"}</span>
          </div>

          <div className="health-row">
            <span>Account Type</span>
            <span className="health-pct">
              {accountDetails?.accountType || "-"}
            </span>
          </div>
          <div className="ad-full-info-button-container">
            <button className="ad-btn-full-info">
              Full Information
              <span className="ad-arrow-icon">→</span>
            </button>
          </div>
        </div>
      </div>
    </>
  );

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .dv-page {
          font-family:  'Segoe UI', sans-serif;
          font-size: 13px;
          background: #f5f6f8;
          color: #1a1a2e;
          height: 100vh;
          overflow: hidden;
        }

        body {
    padding: 0px;
}

        .main { height: 100%; display: flex; flex-direction: column; overflow: hidden; }

        .quarter-bar {
          background: #f5f6f8; border-bottom: 1px solid #e8eaed; padding: 4px 24px;
          font-size: 12px; color: #555; display: flex; align-items: center; gap: 6px;
        }
        .closure-badge { color: #e2001a; font-weight: 600; }

        .page-header { background: #f5f6f8; border-bottom: 1px solid #e8eaed; padding: 10px 0px 0; margin-top: 38px; }
        .page-header-row { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; padding: 0px 24px; }
        .back-btn { font-size: 18px; cursor: pointer; color: #444; background: none; border: none; }
        .topbar-title { font-size: 28px; font-weight: 700; color: #111; }
        .topbar-account { font-size: 12px; text-decoration: underline; cursor: pointer;padding: 4px 10px;
    background: #ECFDF5;
    border-radius: 4px;text-decoration: none;  }
        .topbar-actions { margin-left: auto; display: flex; align-items: center; gap: 10px; }
        .sd-btn-prep { 
   display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #E2E8F0;
    color: #374151;
    border: none;
    font-size: 12.5px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    padding: 7px 16px;
    border-radius: 50px;
    cursor: pointer;
    transition: background .15s, border-color .15s;         }
      .btn-delete { width: 32px; height: 32px; border-radius: 50px; border: none; background: #e2001a; color: #fff; cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center; }

        .tabs-outer { height: 64px; background: #fff; display: flex; align-items: center; border-top: 1px solid #e8eaed; }
        .tabs { height: 40px; display: flex; gap: 0; background: #fff; justify-content: center; width: 100%; }
        .tab { padding: 8px 16px; font-size: 14px; cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap; background: #fff; border-top: 0; border-left: 0; border-right: 0; height: 100%; display: flex; align-items: center; }
        .tab.active { color: #1a73e8; border-bottom-color: #1a73e8; font-weight: 500; background: #EFF6FF; }

        .content { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; gap: 20px; }
        .col-main { flex: 1; display: flex; flex-direction: column; gap: 16px; }
        .col-side { width: 280px; min-width: 280px; display: flex; flex-direction: column; gap: 16px; }

        .pipeline { background: #fff; border: 1px solid #e8eaed; border-radius: 8px; padding: 20px 24px; display: flex; align-items: center; justify-content: space-between; position: relative; }
        .pipeline::before { content: ''; position: absolute; top: 35px; left: 38px; right: 38px; height: 1px; background: #ddd; z-index: 0; transform: translateY(-50%); }
        .stage { display: flex; flex-direction: column; align-items: center; gap: 6px; position: relative; z-index: 1; }
        .stage-dot { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; border: 2px solid #ddd; background: #fff; }
        .stage-dot.done { background: #34a853; border-color: #34a853; color: #fff; }
        .stage-dot.active { border-color: #1a73e8; border-width: 3px; }
        .stage-dot.inactive { border-color: #ccc; }
        .stage-label { font-size: 11px; color: #555; white-space: nowrap; }

        .card { background: #fff; border: 1px solid #e8eaed; border-radius: 8px; padding: 20px; }
        .card-title { font-size: 18px; font-weight: 700; margin-bottom: 14px; color: #111; }

        .metrics { display: flex; gap: 12px; margin-bottom: 16px; }
        .metric { flex: 1; border: 1px solid #e8eaed; border-radius: 6px; padding: 10px 14px; }
        .metric-label { font-size: 11px; color: #777; margin-bottom: 4px; }
        .metric-value { font-size: 17px; font-weight: 700; }

        .deal-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.summary-image {
  width: 40px;
  height: 40px;
  object-fit: contain;
  flex-shrink: 0;
}
  .deal-content {
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  gap: 12px;
}

.deal-image {
  width: 40px;
  height: 40px;
  flex-shrink: 0;
}
      .deal-inner {
  position: relative;
  padding: 16px;
  border-radius: 8px;
  background: #fff;
}

.deal-inner::before {
  content: "";
  position: absolute;
  inset: 0;
  padding: 2px; /* border thickness */
  border-radius: 8px;
  background: linear-gradient(to bottom, #AD1A98, #3768C7);

  -webkit-mask:
    linear-gradient(#fff 0 0) content-box,
    linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;

  pointer-events: none;
}
        .deal-inner-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px; }
        .deal-inner-title { font-size: 14px; font-weight: 700; }
        .deal-close { font-size: 12px; color: #555; }
        .deal-id { font-size: 11px; color: #888; margin-bottom: 10px;margin: 0; }
        .deal-desc { font-size: 14px;
    color: #333;
    line-height: 20px;
    margin-bottom: 12px;
    font-weight: 400; }
        .deal-meta { display: flex; justify-content: center; align-items-center; gap: 20px; font-size: 11px; color: #555; border-top: 1px solid #eee; margin-bottom:0;  }

        .actions-header { display: flex; gap:20px; align-items: center; margin-bottom: 14px; }
        .action-item { border: 1px solid #e8eaed; border-radius: 8px; padding: 14px; }
        .action-item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .action-item-title { font-size: 13px; font-weight: 600; }
        .due-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c; font-size: 11px; font-weight: 600; }
        .due-dot { width: 7px; height: 7px; border-radius: 50%; background: #dc2626; }
        .action-item-desc { font-size: 12px; color: #555; margin-bottom: 12px; }

        .sd-action-btn { display: inline-flex; align-items: center; justify-content: center; background: #1D4ED8; color: #fff; font-size: 13px; font-weight: 600; padding: 8px 22px; border-radius: 50px; border: none; cursor: pointer; }
        .sd-action-btn:hover { background: #1F3A8A; }

        .account-header { display: flex; align-items: center; margin-bottom: 14px; }
        .account-logo { width: 40px; height: 40px; background: #2c3e50; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 18px; margin-right: 12px; }
        .account-meta { font-size: 12px; color: #666; }
        .account-meta-line { font-size: 11px; color: #888; margin-top: 2px; }
        .account-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .account-field { background: #f9f9fb; border-radius: 6px; padding: 10px 12px; }
        .field-label { font-size: 10px; color: #888; margin-bottom: 3px; }
        .field-value { font-size: 13px; font-weight: 600; }
        .account-field.full { grid-column: 1 / -1; }

        .health-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .health-badge { font-size: 12px; color: #34a853; font-weight: 600; }
        .health-date { font-size: 11px; color: #888; margin-bottom: 14px; }
        .health-main { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; margin-bottom: 10px; }
        .health-main-label { font-size: 13px; font-weight: 600; }
        .health-main-val { font-size: 14px; font-weight: 700; color: #34a853; }
        .health-row { display: flex; justify-content: space-between; font-size: 12px; color: #555; padding: 5px 0; border-bottom: 1px solid #f0f0f0; }
        .health-row:last-child { border-bottom: none; }
        .health-pct { font-weight: 600; color: #333; }

        .sd-priority-meta { font-size: 12px; color: #6b7280; }
        .sd-priority-due-today { color: #E1251B; font-weight: 600; }
      `}</style>

      <div className="dv-page">
        <div className="main">
          <div className="quarter-bar">
            Q2 FY2024 · Week 10 of 12 ·{" "}
            <span className="closure-badge">Closure Phase</span>
          </div>

          <div className="page-header">
            <div className="page-header-row">
              <button
                className="back-btn"
                onClick={() => navigate("/accounts")}
              >
                  <i className="bi bi-chevron-left" style={{ fontSize: 16, WebkitTextStroke: "1px", color: "#0F172A" }}></i>


              </button>
              <span className="topbar-title">{accountDetails?.accountNumber} - {accountDetails?.name || "-"}</span>
              <span className="topbar-account">{accountDetails?.accountType || "-"}</span>
              {/* <div className="topbar-actions">
                <button className="sd-btn-prep">Edit</button>
                <button className="btn-delete" aria-label="Delete">
                  <i className="bi bi-trash"></i>
                </button>
              </div> */}
            </div>
            <div className="tabs-outer">
              <div className="tabs">
                <button
                  className={`tab ${activeTopTab === "summary" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("summary")}
                >
                  Summary
                </button>
                <button
                  className={`tab ${activeTopTab === "opportunities" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("opportunities")}
                >
                  Opportunities
                </button>
                <button
                  className={`tab ${activeTopTab === "customer-info" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("customer-info")}
                >
                  Customer Information
                </button>
                <button
                  className={`tab ${activeTopTab === "contacts" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("contacts")}
                >
                  Contacts
                </button>
                {/* <button
                  className={`tab ${activeTopTab === "notes" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("notes")}
                >
                  Notes
                </button> */}
              </div>
            </div>
          </div>

          <div className="content">
            {activeTopTab === "summary" ? (
              renderSummary()
            ) : (
              <div className="col-main">{renderTabContent()}</div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
