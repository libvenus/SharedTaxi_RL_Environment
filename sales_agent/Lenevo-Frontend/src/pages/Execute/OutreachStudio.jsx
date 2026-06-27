import { useEffect, useState } from "react";
import avatar1 from "../../assets/avatar1.jpg";
import accountIcon from "../../assets/icons/account_icon.png";
import { fetchOutreach } from "../../api/client";
import "../../styles/execute.css";

function formatDealAmount(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "$0";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(amount);
}

function mapStatusClasses(priorityBadge = "") {
  const badge = String(priorityBadge).toUpperCase();
  if (badge.includes("HIGH") || badge.includes("RISK")) {
    return { statusClass: "sd-status-at-risk", dotClass: "sd-status-dot-red" };
  }
  if (badge.includes("ACCOUNT")) {
    return { statusClass: "sd-status-needs-attention", dotClass: "sd-status-dot-amber" };
  }
  return { statusClass: "sd-status-opportunity", dotClass: "sd-status-dot-blue" };
}

export default function OutreachStudio() {
  const [accounts, setAccounts] = useState([]);
  const [tabCounts, setTabCounts] = useState({
    HIGH_PRIORITY: 0,
    MEETING_FOLLOW_UP: 0,
    ACCOUNT: 0,
    SILENT_AT_RISK: 0,
  });
  const [showDraftComposer, setShowDraftComposer] = useState(false);
  const [draftAccount, setDraftAccount] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState("HIGH_PRIORITY");
  const defaultToValue = "stakeholder@acmeglobal.com";
  const defaultCcValue = "salesmanager@lenovo.com";
  const defaultSubjectValue = "Proposal follow-up and next steps";
  const draftHelperText = 'Click "Generate Draft" to create an AI-personalized email using the selected template and account context.';
  const [toValue, setToValue] = useState(defaultToValue);
  const [ccValue, setCcValue] = useState(defaultCcValue);
  const [subjectValue, setSubjectValue] = useState(defaultSubjectValue);
  const [bodyValue, setBodyValue] = useState(draftHelperText);

  const categoryTabs = [
    { id: "HIGH_PRIORITY", label: "High Priority" },
    { id: "MEETING_FOLLOW_UP", label: "Meeting Follow-Ups" },
    { id: "ACCOUNT", label: "Account Outreach" },
    { id: "SILENT_AT_RISK", label: "Silent / At-Risk" },
  ];

  const selectedTabLabel = categoryTabs.find((tab) => tab.id === selectedCategory)?.label || "Accounts";

  useEffect(() => {
    const controller = new AbortController();
    const loadOutreach = async () => {
      try {
        const data = await fetchOutreach({ category: selectedCategory }, controller.signal);
        if (data?.counts) {
          setTabCounts({
            HIGH_PRIORITY: data.counts.high_priority ?? 0,
            MEETING_FOLLOW_UP: data.counts.meeting_follow_ups ?? 0,
            ACCOUNT: data.counts.account_outreach ?? 0,
            SILENT_AT_RISK: data.counts.silent_at_risk ?? 0,
          });
        }
        if (Array.isArray(data?.records)) {
          setAccounts(data.records);
          return;
        }
        setAccounts([]);
      } catch (error) {
        if (error?.name !== "AbortError") {
          setAccounts([]);
        }
      }
    };

    loadOutreach();
    return () => controller.abort();
  }, [selectedCategory]);

  const handleResetDraftFields = () => {
    setToValue(defaultToValue);
    setCcValue(defaultCcValue);
    setSubjectValue(defaultSubjectValue);
  };

  const handleOpenDraftComposer = (acct) => {
    const recipientEmail = acct?.decision_maker_email || defaultToValue;
    const accountOrCompany = acct?.account_name || acct?.company_name || "Account";
    const dealName = acct?.deal_name || "Opportunity";
    const nextReason = acct?.why_now_reason || "";
    const activitySummary = acct?.last_activity_summary || "";

    setDraftAccount(acct);
    setToValue(recipientEmail);
    setCcValue(defaultCcValue);
    setSubjectValue(`Follow-up: ${dealName} (${accountOrCompany})`);
    setBodyValue(
      [
        `Hi ${acct?.decision_maker_name || "there"},`,
        "",
        activitySummary,
        nextReason,
        "",
        "Please let me know a good time to connect and discuss next steps.",
      ]
        .filter(Boolean)
        .join("\n"),
    );
    setShowDraftComposer(true);
  };

  return (
    <div className="opd-root">
      {!showDraftComposer && (
        <>
          <div className="opd-summary-row">
            {categoryTabs.map((tab) => {
              const isActive = selectedCategory === tab.id;
              return (
                <div key={tab.id} className="opd-summary-col">
                  <div
                    className="opd-scard-simple"
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedCategory(tab.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelectedCategory(tab.id);
                      }
                    }}
                    aria-pressed={isActive}
                    style={isActive ? { borderColor: "#1a56db", background: "#eff4ff" } : undefined}
                  >
                    <span className="opd-label">{tab.label}</span>
                    <span className="opd-value">{tabCounts[tab.id] ?? 0}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="sd-accounts-panel">
            <div className="sd-accounts-header">
              <h2 className="sd-accounts-title">{selectedTabLabel}</h2>
            </div>

            {accounts.map((acct) => {
              const { statusClass, dotClass } = mapStatusClasses(acct?.priority_badge);
              const rowKey = acct?.id || `${acct?.account_name || "account"}-${Math.random()}`;
              const companyName =  acct?.company_name || "Unknown Account";
              const accountName =  acct?.account_name || "Unknown Account";
              const statusLabel = acct?.priority_badge || acct?.outreach_type || "--";
              const dealText = acct?.deal_name || "--";
              const dealAmount = formatDealAmount(acct?.deal_value);
              const dealStage = acct?.deal_stage || "--";
              const leftActivityType = acct?.last_activity_type || "--";
              const leftActivityDateTime = acct?.last_activity_datetime || "--";
              const leftActivitySummary = acct?.last_activity_summary || "--";
              const leftCardText = `${leftActivityType} .  ${leftActivityDateTime} .  ${leftActivitySummary}`;
              const signalText = acct?.why_now_reason || "--";

              return (
              <div key={rowKey} className="sd-account-row">
                <div className="sd-account-row-header">
                  <div className="sd-account-heading-block">
                    <div className="sd-account-name-group">
                      <span className="sd-account-name">{companyName}</span>
                      <span className={statusClass}>
                        <span className={dotClass}></span>
                        {statusLabel}
                      </span>
                    </div>
                    <div className="sd-account-avatar-strip">
                      <div className="sd-avatar-group">
                        <div className="sd-avatar sd-avatar-initials-ra">RA</div>
                        <img src={avatar1} alt="Stakeholder" className="sd-avatar-img" />
                        <div className="sd-avatar sd-avatar-initials-ks">KS</div>
                      </div>
                      <div className="sd-account-avatar-info">
                        <span className="sd-account-deal-text">
                          <span>{accountName} - {dealText}</span>
                          <span className="sd-account-deal-amount">{dealAmount}</span>
                        </span>
                        <span className="sd-account-stage-pill">{dealStage}</span>
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="sd-draft-btn"
                    onClick={() => handleOpenDraftComposer(acct)}
                  >
                    <img src={accountIcon} alt="" className="sd-draft-btn-icon" aria-hidden="true" />
                    <span>Draft Email</span>
                  </button>
                </div>

                {dealStage?.toUpperCase() !== "DISCOVER" && (
                  <div className="sd-account-task-row">
                    <div className="sd-account-insights">
                      <div className="sd-mini-card sd-mini-card-avatars">
                        <div className="sd-mini-card-copy">
                          {/* <div className="sd-mini-card-title">Outreach Context</div> */}
                          <p className="sd-mini-card-subtitle">{leftCardText}</p>
                        </div>
                      </div>
                      <div className="sd-mini-card sd-mini-card-highlight">
                        <img src={accountIcon} alt="Account signal" className="sd-mini-card-icon" />
                        <div className="sd-mini-card-copy">
                          <div className="sd-mini-card-title">Account Signal</div>
                          <p className="sd-mini-card-subtitle">{signalText}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );})}
          </div>
        </>
      )}

      {showDraftComposer && (
        <div className="sd-draft-shell">
          <div className="sd-draft-layout">
            <div className="sd-draft-left">
              <div className="sd-draft-card sd-draft-card-sm">
                <h3 className="sd-draft-card-title">Select Template - Generate initial email draft</h3>
                <div className="sd-template-row">
                  {["General Follow-up", "Meeting Follow-up", "Proposal/Quote", "Re-Engagement"].map((template) => (
                    <div key={template} className="sd-template-chip" role="button" tabIndex={0}>
                      {template}
                    </div>
                  ))}
                </div>
              </div>

              <div className="sd-draft-card sd-draft-card-lg">
                <div className="sd-compose-toolbar">
                  <button type="button" className="sd-compose-reset" onClick={handleResetDraftFields}>
                    <i className="bi bi-arrow-clockwise" aria-hidden="true"></i>
                  </button>
                </div>
                <div className="sd-compose-row">
                  <span className="sd-compose-label">TO:</span>
                  <div className="sd-compose-field-wrap">
                    <input className="sd-compose-value sd-compose-input" value={toValue} onChange={(e) => setToValue(e.target.value)} aria-label="To" />
                    <div className="sd-compose-divider" />
                  </div>
                </div>
                <div className="sd-compose-row">
                  <span className="sd-compose-label">CC:</span>
                  <div className="sd-compose-field-wrap">
                    <input className="sd-compose-value sd-compose-input" value={ccValue} onChange={(e) => setCcValue(e.target.value)} aria-label="Cc" />
                    <div className="sd-compose-divider" />
                  </div>
                </div>
                <div className="sd-compose-row">
                  <span className="sd-compose-label">Subject:</span>
                  <div className="sd-compose-field-wrap">
                    <input className="sd-compose-value sd-compose-input" value={subjectValue} onChange={(e) => setSubjectValue(e.target.value)} aria-label="Subject" />
                    <div className="sd-compose-divider" />
                  </div>
                </div>

                <div className="sd-compose-editor-wrap">
                  <div className="sd-compose-header">
                    <img src={accountIcon} alt="" className="sd-compose-header-icon" aria-hidden="true" />
                  </div>
                  <textarea className="sd-compose-editor sd-compose-textarea" value={bodyValue} onChange={(e) => setBodyValue(e.target.value)} aria-label="Draft body" />
                  <div className="sd-compose-footer">
                    <div className="sd-compose-count">{(bodyValue || "").length}/1000</div>
                    <button type="button" className="sd-send-btn">
                      <img src={accountIcon} alt="" className="sd-send-btn-icon" aria-hidden="true" />
                      <span>Send Email</span>
                    </button>
                  </div>
                </div>
              </div>

              <div className="sd-draft-card sd-draft-card-sm">
                <h3 className="sd-draft-card-title">Additional content (optional)</h3>
                <div className="sd-additional-row">
                  <div className="sd-attach-note">
                    <span className="sd-attach-inline-icon">📎</span>
                    <span>Attach e-mails, documents from desktop, folder, mailbox</span>
                  </div>
                  <button type="button" className="sd-next-btn">→</button>
                </div>
              </div>
            </div>

            <div className="sd-draft-right">
              <div className="sd-draft-card sd-latest-card">
                <h3 className="sd-draft-card-title">Latest Activity</h3>
                <div className="sd-activity-head">
                  <span className="sd-mail-icon">✉</span>
                  <span className="sd-activity-name">Outreach Mail</span>
                  <span className="sd-high-badge">{draftAccount?.priority_badge}</span>
                </div>
                <div className="sd-activity-datetime">12 May 2026 <span>6:32pm</span></div>
                <p className="sd-activity-extra">Kiran has reviewed the submitted proposal in full and is moving forward with an updated scope.</p>
                <ul className="sd-activity-list">
                  <li>Stakeholder replied and requested revised proposal timeline.</li>
                  <li>Follow-up call window shared for this week.</li>
                </ul>
              </div>

              <div className="sd-draft-card sd-context-card">
                <h3 className="sd-draft-card-title">Outreach Context</h3>
                <div className="sd-context-line">
                  <span>Deal Stage</span>
                  <span className="sd-proposal-badge">{draftAccount?.deal_stage}</span>
                </div>
                <div className="sd-context-line">
                  <span>Engagement Type</span>
                  <strong>Inbound email</strong>
                </div>
                <div className="sd-context-line">
                  <span>Deal Value</span>
                  <strong>{formatDealAmount(draftAccount?.deal_value) || "$142k"}</strong>
                </div>
                <div className="sd-context-line">
                  <span>Primary Contact</span>
                  <strong>{draftAccount?.account_name || draftAccount?.company_name || "Acme Corp Global"}</strong>
                </div>

                <div className="sd-why-card">
                  <div className="sd-why-title-row">
                    <img src={accountIcon} alt="Account signal" className="sd-why-icon-img" />
                    <span className="sd-why-title">Why Now?</span>
                  </div>
                  <p className="sd-why-copy">
                    {draftAccount?.why_now_reason }
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
