import { useState, useEffect } from "react";
import accounticon from "../../../assets/icons/account_icon.png";
import { updateCrmField, updateMeetingSummary } from "../../../api/client";

const fallbackCrmUpdates = [
  { label: "Stage", value: "Negotiation → Closure", priority: "high" },
  { label: "Date Close", value: "Jun 15th → Jun 13th", priority: "high" },
  { label: "Deal Value", value: "$180K → $195K", priority: "high" },
  { label: "Next Milestone", value: "Signed PO by June 12th", priority: "low" },
  {
    label: "Products",
    value: "ThinkStation P360 → P360 + ProSupport+ + Monitor",
    priority: "medium",
  },
];

const fallbackSyncFields = [
  {
    label: "Stage",
    priority: "high",
    priorityBadge: "priority",
    current: "Negotiation",
    recommended: "Closure",
    why: "Suresh Kumar verbally confirmed pricing alignment and legal review is complete. Deal is ready to advance to Closure.",
  },
  {
    label: "Date Close",
    priority: "high",
    priorityBadge: "priority",
    current: "Apr 25, 2026",
    recommended: "May 25, 2026",
    why: "Suresh Kumar verbally confirmed pricing alignment and legal review is complete. Deal is ready to advance to Closure.",
  },
  {
    label: "Next Milestone",
    priority: "low",
    priorityBadge: "badge-low-warn",
    current: "Confirm final pricing sign-off",
    recommended: "PO signed by June 12th",
    why: "Suresh Kumar verbally confirmed pricing alignment and legal review is complete. Deal is ready to advance to Closure.",
  },
];

function getUpdatesFromApi(crmData) {
  // if (!crmData?.updates)
  //   return { crmList: fallbackCrmUpdates, syncList: fallbackSyncFields };

  // const allUpdates = Object.values(crmData.updates).flat();

  // const crmList = allUpdates.map((item) => ({
  //   label: item.Field?.trim(),
  //   value: `${item.current} → ${item.suggested}`,
  //   priority: item.confidence?.toLowerCase() || "low",
  // }));

  // const syncList = allUpdates.map((item) => {
  //   const priority = item.confidence?.toLowerCase() || "low";
  //   return {
  //     id: item.id,
  //     label: item.Field?.trim(),
  //     priority,
  //     priorityBadge: priority === "high" ? "priority" : "badge-low-warn",
  //     current: item.current,
  //     recommended: item.suggested,
  //     why: "",
  //   };
  // });

  const crmList = [
    { label: "Quantity", value: "1200 → 1400", priority: "high" },
    { label: "Target Close Date", value: "30th July → 15th July", priority: "high" },
  ];

  const syncList = [
    {
      id: 1,
      label: "Quantity",
      priority: "high",
      priorityBadge: "priority",
      current: "1200",
      recommended: "1400",
      why: "Scope confirmed at 1,400 units during the final technical review with Ford manufacturing team.",
    },
    {
      id: 2,
      label: "Target Close Date",
      priority: "high",
      priorityBadge: "priority",
      current: "30th July",
      recommended: "15th July",
      why: "Ford's production line launches August 1 — delivery must begin in July to meet the timeline.",
    },
  ];

  return { crmList, syncList };
}

export default function CrmUpdates({
  viewOnly = false,
  crmUpdates: crmData,
  meetingId,
  summary: apiSummary,
}) {
  const { crmList, syncList } = getUpdatesFromApi(crmData);
  const [editingIdx, setEditingIdx] = useState(null);
  const [editValues, setEditValues] = useState({
    current: "",
    recommended: "",
  });
  const [editingSummary, setEditingSummary] = useState(false);
  const [summaryValue, setSummaryValue] = useState(
    apiSummary || "Pricing review call with Suresh Kumar and Divya Nair. Pricing confirmed at $195K. Legal cleared contract. Delivery by June 20th required. ProSupport+ and monitor bundle confirmed. Next: collect signed PO by June 12th."
  );

  useEffect(() => {
    if (apiSummary) setSummaryValue(apiSummary);
  }, [apiSummary]);

  const handleEdit = (idx, field) => {
    setEditingIdx(idx);
    setEditValues({ current: field.current, recommended: field.recommended });
  };

  const handleCancelEdit = () => {
    setEditingIdx(null);
    setEditValues({ current: "", recommended: "" });
  };

  const [updateStatus, setUpdateStatus] = useState({});
  const [summaryUpdated, setSummaryUpdated] = useState(false);

  const callUpdateApi = async (field, status, current, suggested) => {
    if (!meetingId || field.id == null) {
      console.warn("Cannot update: missing meetingId or field.id", { meetingId, fieldId: field.id });
      return false;
    }
    try {
      await updateCrmField(meetingId, field.id, {
        field: field.label,
        current_value: current,
        suggested_value: suggested,
        status,
      });
      return true;
    } catch (error) {
      console.error("CRM update failed:", error);
      return false;
    }
  };

  const handleSkip = async (idx, field) => {
    const current = editingIdx === idx ? editValues.current : field.current;
    const suggested =
      editingIdx === idx ? editValues.recommended : field.recommended;
    const success = await callUpdateApi(field, "skip", current, suggested);
    if (success) {
      setUpdateStatus((prev) => ({ ...prev, [idx]: "skipped" }));
    }
    handleCancelEdit();
  };

  const handleUpdate = async (idx, field) => {
    const current = editingIdx === idx ? editValues.current : field.current;
    const suggested =
      editingIdx === idx ? editValues.recommended : field.recommended;
    const success = await callUpdateApi(field, "update", current, suggested);
    if (success) {
      setUpdateStatus((prev) => ({ ...prev, [idx]: "updated" }));
    }
    handleCancelEdit();
  };

  const handleSkipSummary = async () => {
    if (!meetingId) {
      console.warn("Cannot update summary: missing meetingId");
      return;
    }
    try {
      await updateMeetingSummary(meetingId, {
        summary: summaryValue,
        status: "skip",
      });
      setEditingSummary(false);
      setSummaryUpdated(true);
    } catch (error) {
      console.error("Summary update failed:", error);
    }
  };

  const handleEditSummary = () => {
    setEditingSummary(true);
    setTimeout(() => {
      const field = document.querySelector('[data-summary-field]');
      if (field) {
        field.focus();
        const range = document.createRange();
        range.selectNodeContents(field);
        range.collapse(false);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }, 0);
  };

  const handleUpdateSummary = async () => {
    if (!meetingId) {
      console.warn("Cannot update summary: missing meetingId");
      return;
    }
    try {
      await updateMeetingSummary(meetingId, {
        summary: summaryValue,
        status: "update",
      });
      setEditingSummary(false);
      setSummaryUpdated(true);
    } catch (error) {
      console.error("Summary update failed:", error);
    }
  };
  return (
    <>
      {/* <div className="kp-panel">
        <div className="kp-panel-title">CRM Updates From The Call</div>
        <div className="crm-card-list">
          {crmList.map((item, idx) => (
            <div className="crm-row" key={idx}>
              <div className="crm-icon"><img src={accounticon} alt="CRM" /></div>
              <div className="crm-body">
                <span className="crm-label">{item.label}</span>
                <span className="crm-value">{item.value}</span>
              </div>
              <span className={`priority priority-${item.priority}`}>{item.priority.charAt(0).toUpperCase() + item.priority.slice(1)}</span>
            </div>
          ))}
        </div>
      </div> */}
      <div className="kp-panel-title">Sync with Dynamics 365</div>
      <div className="kp-panel-title">ID2384503 ThinkStation Pricing Close</div>
        <div className="sync-card">
          {syncList.map((field, idx) => (
            <div key={idx}>
              {idx > 0 && <hr className="crm-divider" />}
              <div className="crm-section">
                <div className="section-header">
                  <span className="section-label">{field.label}</span>
                  {field.priorityBadge === "priority" ? (
                    <span className={`priority priority-${field.priority}`}>
                      {field.priority.charAt(0).toUpperCase() +
                        field.priority.slice(1)}
                    </span>
                  ) : (
                    <span className="badge-low-warn">
                      <span className="warn-dot">!</span>
                      {field.priority.charAt(0).toUpperCase() +
                        field.priority.slice(1)}
                    </span>
                  )}
                </div>
                <div className="fields-row">
                  <div className="field-box">
                    <div className="field-meta">Current</div>
                    {editingIdx === idx ? (
                      <input
                        className="field-value field-edit-input"
                        value={editValues.current}
                        onChange={(e) =>
                          setEditValues((v) => ({
                            ...v,
                            current: e.target.value,
                          }))
                        }
                      />
                    ) : (
                      <div className="field-value">{field.current}</div>
                    )}
                  </div>
                  <div className="field-box recommended">
                    <div className="field-meta">
                      <img
                        src={accounticon}
                        alt="AI"
                        style={{ width: 14, height: 14 }}
                      />
                      Recommended
                    </div>
                    {editingIdx === idx ? (
                      <input
                        className="field-value field-edit-input"
                        value={editValues.recommended}
                        onChange={(e) =>
                          setEditValues((v) => ({
                            ...v,
                            recommended: e.target.value,
                          }))
                        }
                      />
                    ) : (
                      <div className="field-value">{field.recommended}</div>
                    )}
                  </div>
                  {!viewOnly && !updateStatus[idx] && (
                    <div className="field-actions">
                      {editingIdx === idx ? (
                        <i
                          className="ri-close-line"
                          style={{ cursor: "pointer" }}
                          onClick={handleCancelEdit}
                        ></i>
                      ) : (
                        <i
                          className="ri-pencil-line"
                          style={{ cursor: "pointer" }}
                          onClick={() => handleEdit(idx, field)}
                        ></i>
                      )}
                      <button
                        className="btn-skip"
                        onClick={() => handleSkip(idx, field)}
                      >
                        Skip
                      </button>
                      <button
                        className="btn-update"
                        onClick={() => handleUpdate(idx, field)}
                      >
                        Update
                      </button>
                    </div>
                  )}
                </div>
                <p className="why-text">
                  <strong>Why?</strong> {field.why}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="sync-card">
          <div className="kp-panel-title">Meeting Notes</div>
          <div className="crm-section">
            <div className="section-header">
              <span className="section-label">AI-Generated Meeting Notes</span>
            </div>
            <div className="fields-row" style={{ flexWrap: "wrap" }}>
              <div className="field-box recommended full-width">
                <div
                  data-summary-field
                  className="field-value"
                  contentEditable={editingSummary}
                  suppressContentEditableWarning
                  onBlur={(e) => setSummaryValue(e.currentTarget.textContent)}
                  style={{
                    minHeight: "60px",
                    outline: "none",
                    border: "none",
                    boxShadow: "none",
                  }}
                >
                  {summaryValue}
                </div>
              </div>
              {!viewOnly && !summaryUpdated && (
                <div className="field-actions">
                  {editingSummary ? (
                    <i
                      className="ri-close-line"
                      style={{ cursor: "pointer" }}
                      onClick={() => setEditingSummary(false)}
                    ></i>
                  ) : (
                    <i
                      className="ri-pencil-line"
                      style={{ cursor: "pointer" }}
                      onClick={handleEditSummary}
                    ></i>
                  )}
                  <button className="btn-skip" onClick={handleSkipSummary}>
                    Skip
                  </button>
                  <button className="btn-update" onClick={handleUpdateSummary}>
                    Update
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
    </>
  );
}
