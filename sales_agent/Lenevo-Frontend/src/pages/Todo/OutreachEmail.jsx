import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import accountIcon from "../../assets/icons/account_icon.png";
import "../../styles/execute.css";
import {
  sendEmail,
  generateEmailDraftV1,
  fetchTodoOutreach,
} from "../../api/client";
import { formatCurrencyShort } from "../../utils/format";
import { Modal } from "react-bootstrap";

const EMPTY_OUTREACH_CONTEXT = {
  priority_badge: "",
  deal_stage: "",
  deal_value: "",
  primary_contact: "",
  why_now_reason: "",
};

const EMPTY_EMAIL_DRAFT = {
  to: "",
  cc: "",
  subject: "",
  body: "",
};

function pickFirst(obj, keys, fallback = "") {
  if (!obj) return fallback;
  for (const key of keys) {
    const value = obj[key];
    if (value !== undefined && value !== null && value !== "") {
      return value;
    }
  }
  return fallback;
}

const OutreachEmail = () => {
  const location = useLocation();
  const { id } = useParams();
  const selectedTask = location.state?.selectedTask || null;

  // Full task detail fetched from API (richer than navigation state)
  const [todoDetail, setTodoDetail] = useState(null);

  // Merge API data (preferred) with navigation state fallback
  const taskContext = useMemo(() => {
    const src = todoDetail || selectedTask || {};
    return {
      task_title: src.task_title || src.title || "",
      priority: src.priority || src.priority_badge || "",
      stage_name: src.stage_name || src.deal_stage || src.stage || "",
      deal_value: src.deal_value ?? null,
      account_name: src.account_name || "",
      opportunity_name: src.opportunity_name || "",
      attendees_email: src.attendees_email || src.attendees_emails || "",
      notes: src.notes || src.why_now_reason || "",
      source_label: src.source_label || "",
      due_date: src.due_date || "",
      type_tag: src.type_tag || "",
    };
  }, [todoDetail, selectedTask]);

  const mappedContext = useMemo(() => ({
    priority_badge: taskContext.priority,
    deal_stage: taskContext.stage_name,
    deal_value: taskContext.deal_value,
    primary_contact: taskContext.account_name,
    why_now_reason: taskContext.notes,
  }), [taskContext]);

  const initialDraft = useMemo(() => ({
    to: taskContext.attendees_email,
    cc: "",
    subject: taskContext.task_title,
    body: "",
  }), [taskContext]);

  const [toValue, setToValue] = useState(initialDraft.to);
  const [ccValue, setCcValue] = useState(initialDraft.cc);
  const [subjectValue, setSubjectValue] = useState(initialDraft.subject);
  const [bodyValue, setBodyValue] = useState(initialDraft.body);
  const bodyTextareaRef = useRef(null);

  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [additionalContext, setAdditionalContext] = useState("");
  const [isLoadingDraft, setIsLoadingDraft] = useState(false);
  const [emailTemplates, setEmailTemplates] = useState([]);

  // Fetch task detail + templates from API
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    fetchTodoOutreach(id, controller.signal)
      .then((data) => {
        setTodoDetail(data);
        const templates = data?.email_template || [];
        setEmailTemplates(templates);
        const defaultTemplate = templates.find((t) => t.default === true) || templates[0];
        if (defaultTemplate) {
          setSelectedTemplate(defaultTemplate.template_name);
        }
      })
      .catch((err) => {
        if (err.name !== "AbortError")
          console.error("Failed to fetch todo outreach:", err);
      });
    return () => controller.abort();
  }, [id]);

  useEffect(() => {
    setToValue(initialDraft.to);
    setCcValue(initialDraft.cc);
    setSubjectValue(initialDraft.subject);
    setBodyValue(initialDraft.body);
  }, [initialDraft]);

  useEffect(() => {
    if (!bodyTextareaRef.current) return;
    bodyTextareaRef.current.style.height = "auto";
    bodyTextareaRef.current.style.height = `${bodyTextareaRef.current.scrollHeight}px`;
  }, [bodyValue]);

  const [sending, setSending] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [modalMessage, setModalMessage] = useState("");

  const handleTemplateSelect = (template) => {
    setSelectedTemplate(template.template_name);
    setSubjectValue(template.template_name);
  };

  const handleGenerateDraft = async () => {
    const template = emailTemplates.find(
      (t) => t.template_name === selectedTemplate,
    );
    if (!template) {
      setModalMessage("Please select a template first.");
      setShowErrorModal(true);
      return;
    }

    setIsLoadingDraft(true);
    try {
      const taskData = {
        task_title: taskContext.task_title,
        notes: taskContext.notes,
        account_name: taskContext.account_name,
        opportunity_name: taskContext.opportunity_name,
        priority: taskContext.priority,
        due_date: taskContext.due_date,
        stage: taskContext.stage_name,
        deal_value: taskContext.deal_value,
        contact_email: toValue,
      };
      const draftResponse = await generateEmailDraftV1({
        context: [
          taskData.task_title,
          taskData.notes,
          taskData.account_name ? `Account: ${taskData.account_name}` : "",
          taskData.opportunity_name ? `Opportunity: ${taskData.opportunity_name}` : "",
          taskData.stage ? `Stage: ${taskData.stage}` : "",
          taskData.due_date ? `Due: ${taskData.due_date}` : "",
        ].filter(Boolean).join(". "),
        written_context: [template.context_used, additionalContext].filter(Boolean).join(" "),
        template: template.template_name,
        data: taskData,
        placeholders: {},
      });
      if (draftResponse) {
        const draft = draftResponse?.draft || draftResponse;
        if (draft.error) {
          setModalMessage("Could not generate draft: " + draft.error);
          setShowErrorModal(true);
        } else {
          setSubjectValue(draft.subject || subjectValue);
          setBodyValue(draft.body || bodyValue);
          setToValue(draft.to || toValue);
          setCcValue(draft.cc || ccValue);
        }
      }
    } catch (err) {
      console.error("Failed to generate email draft:", err);
      setModalMessage("Failed to generate draft. Please try again.");
      setShowErrorModal(true);
    } finally {
      setIsLoadingDraft(false);
    }
  };

  const handleResetDraftFields = () => {
    setToValue(initialDraft.to);
    setCcValue(initialDraft.cc);
    setSubjectValue(initialDraft.subject);
    setBodyValue(initialDraft.body);
  };

  const handleSendEmail = async () => {
    setSending(true);
    try {
      const res = await sendEmail({
        to: ["ranjeet.mathew@sutherlandglobal.com"],
        cc: ["battula.tanusha@sutherlandglobal.com"],
        subject: subjectValue,
        body: bodyValue,
      });
      // const res = await sendEmail({
      //   to: toValue
      //     ? toValue
      //         .split(",")
      //         .map((s) => s.trim())
      //         .filter(Boolean)
      //     : [],
      //   cc: ccValue
      //     ? ccValue
      //         .split(",")
      //         .map((s) => s.trim())
      //         .filter(Boolean)
      //     : [],
      //   subject: subjectValue,
      //   body: bodyValue,
      // });
      setModalMessage(res?.message || "Email sent successfully");
      setShowSuccessModal(true);
    } catch (err) {
      setModalMessage("Failed to send email. Please try again.");
      setShowErrorModal(true);
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <div className="opd-root ">
        <div className="page-header">
          <div className="page-header-row">
            <span className="topbar-title">Todo List : Outreach Studio</span>
          </div>
        </div>
        <div className="sd-draft-shell">
          <div className="sd-draft-layout">
            <div className="sd-draft-left">
              <div className="sd-draft-card sd-draft-card-sm">
                <h3 className="sd-draft-card-title">
                  Select Template - Generate initial email draft
                </h3>
                <div className="sd-template-row">
                  {emailTemplates.length > 0
                    ? emailTemplates.map((template) => (
                        <div
                          key={template.template_name}
                          className={`sd-template-chip ${selectedTemplate === template.template_name ? "active" : ""}`}
                          role="button"
                          tabIndex={0}
                          onClick={() => handleTemplateSelect(template)}
                          onKeyDown={(e) =>
                            e.key === "Enter" && handleTemplateSelect(template)
                          }
                          style={{
                            cursor: "pointer",
                            backgroundColor:
                              selectedTemplate === template.template_name
                                ? "#1d4ed8"
                                : "#f0f0f0",
                            color:
                              selectedTemplate === template.template_name
                                ? "#fff"
                                : "#000",
                            transition: "all 0.2s",
                          }}
                        >
                          {template.template_name}
                        </div>
                      ))
                    : null}
                </div>
              </div>

              <div className="sd-draft-card sd-draft-card-lg">
                <div className="sd-compose-toolbar">
                  <button
                    type="button"
                    className="sd-compose-reset"
                    onClick={handleResetDraftFields}
                  >
                    <i className="bi bi-arrow-clockwise" aria-hidden="true"></i>
                  </button>
                </div>
                <div className="sd-compose-row">
                  <span className="sd-compose-label">TO:</span>
                  <div className="sd-compose-field-wrap">
                    <input
                      className="sd-compose-value sd-compose-input"
                      value={toValue}
                      onChange={(e) => setToValue(e.target.value)}
                      aria-label="To"
                    />
                    <div className="sd-compose-divider" />
                  </div>
                </div>
                <div className="sd-compose-row">
                  <span className="sd-compose-label">CC:</span>
                  <div className="sd-compose-field-wrap">
                    <input
                      className="sd-compose-value sd-compose-input"
                      value={ccValue}
                      onChange={(e) => setCcValue(e.target.value)}
                      aria-label="Cc"
                    />
                    <div className="sd-compose-divider" />
                  </div>
                </div>
                <div className="sd-compose-row">
                  <span className="sd-compose-label">Subject:</span>
                  <div className="sd-compose-field-wrap">
                    <input
                      className="sd-compose-value sd-compose-input"
                      value={subjectValue}
                      onChange={(e) => setSubjectValue(e.target.value)}
                      aria-label="Subject"
                    />
                    <div className="sd-compose-divider" />
                  </div>
                </div>

                <div className="sd-compose-editor-wrap">
                  <div className="sd-compose-header">
                    <img
                      src={accountIcon}
                      alt=""
                      className="sd-compose-header-icon"
                      aria-hidden="true"
                    />
                  </div>
                  <textarea
                    ref={bodyTextareaRef}
                    className="sd-compose-editor sd-compose-textarea"
                    value={bodyValue}
                    onChange={(e) => setBodyValue(e.target.value)}
                    aria-label="Draft body"
                    style={{ backgroundColor: "white" }}
                  />
                  <div className="sd-compose-footer">
                    <div className="sd-compose-count">
                      {(bodyValue || "").length}/1000
                    </div>
                    <button
                      type="button"
                      className="sd-send-btn"
                      onClick={handleSendEmail}
                      disabled={sending}
                    >
                      <img
                        src={accountIcon}
                        alt=""
                        className="sd-send-btn-icon"
                        aria-hidden="true"
                      />
                      <span>{sending ? "Sending..." : "Send Email"}</span>
                    </button>
                  </div>
                </div>
              </div>

              <div className="sd-draft-card sd-draft-card-sm">
                <h3 className="sd-draft-card-title">
                  Additional content (optional)
                </h3>
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 12 }}
                >
                  <div>
                    {/* <label style={{ fontSize: 13, color: "#0f172a", fontWeight: 600, marginBottom: 6, display: "block" }}>
                                        External Writing Changes
                                    </label> */}
                    <textarea
                      value={additionalContext}
                      onChange={(e) => setAdditionalContext(e.target.value)}
                      placeholder="Add any specific modifications or context for the email"
                      style={{
                        width: "100%",
                        minHeight: 80,
                        padding: 10,
                        borderRadius: 4,
                        border: "1px solid #e2e8f0",
                        fontSize: 13,
                        fontFamily: "inherit",
                        resize: "vertical",
                        backgroundColor: "white",
                        color: "#1a1a1a",
                      }}
                    />
                  </div>
                  <div
                    className="sd-additional-row"
                    style={{ justifyContent: "flex-end" }}
                  >
                    <button
                      type="button"
                      className="sd-next-btn"
                      onClick={handleGenerateDraft}
                      disabled={isLoadingDraft}
                      aria-label="Generate draft"
                    >
                      {isLoadingDraft ? (
                        <span
                          className="spinner-border spinner-border-sm"
                          role="status"
                          aria-hidden="true"
                        />
                      ) : (
                        "→"
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="sd-draft-right">
              <div className="sd-draft-card sd-latest-card">
                <h3 className="sd-draft-card-title">Latest Activity</h3>
                <div className="sd-activity-head">
                  <span className="sd-mail-icon">✉</span>
                  <span className="sd-activity-name">{taskContext.type_tag ? taskContext.type_tag.charAt(0).toUpperCase() + taskContext.type_tag.slice(1) : "Outreach"}</span>
                  {taskContext.priority && (
                    <span className="sd-high-badge">{taskContext.priority}</span>
                  )}
                </div>
                {taskContext.due_date && (
                  <div className="sd-activity-datetime">
                    Due {new Date(taskContext.due_date + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                  </div>
                )}
                <p className="sd-activity-extra">{taskContext.task_title}</p>
                {taskContext.source_label && (
                  <ul className="sd-activity-list">
                    <li>{taskContext.source_label}</li>
                  </ul>
                )}
              </div>

              <div className="sd-draft-card sd-context-card">
                <h3 className="sd-draft-card-title">Outreach Context</h3>
                {taskContext.account_name && (
                  <div className="sd-context-line">
                    <span>Account</span>
                    <strong>{taskContext.account_name}</strong>
                  </div>
                )}
                {taskContext.opportunity_name && (
                  <div className="sd-context-line">
                    <span>Opportunity</span>
                    <strong>{taskContext.opportunity_name}</strong>
                  </div>
                )}
                <div className="sd-context-line">
                  <span>Deal Stage</span>
                  <span className="sd-proposal-badge">
                    {taskContext.stage_name || "-"}
                  </span>
                </div>
                <div className="sd-context-line">
                  <span>Deal Value</span>
                  <strong>{formatCurrencyShort(taskContext.deal_value)}</strong>
                </div>
                {taskContext.priority && (
                  <div className="sd-context-line">
                    <span>Priority</span>
                    <strong style={{ textTransform: "capitalize" }}>{taskContext.priority}</strong>
                  </div>
                )}
                {taskContext.due_date && (
                  <div className="sd-context-line">
                    <span>Due Date</span>
                    <strong>{new Date(taskContext.due_date + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</strong>
                  </div>
                )}

                <div className="sd-why-card">
                  <div className="sd-why-title-row">
                    <img
                      src={accountIcon}
                      alt="Account signal"
                      className="sd-why-icon-img"
                    />
                    <span className="sd-why-title">Why Now?</span>
                  </div>
                  <p className="sd-why-copy">{taskContext.notes || "-"}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Modal
        show={showSuccessModal}
        backdrop="static"
        keyboard={false}
        centered
      >
        <Modal.Body style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <i
            className="ri-checkbox-circle-fill"
            style={{ color: "#047857", fontSize: 20 }}
          ></i>
          <span>{modalMessage}</span>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button
            className="kp-btn-update"
            onClick={() => setShowSuccessModal(false)}
          >
            OK
          </button>
        </Modal.Footer>
      </Modal>

      <Modal show={showErrorModal} backdrop="static" keyboard={false} centered>
        <Modal.Body style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <i
            className="ri-error-warning-fill"
            style={{ color: "#dc2626", fontSize: 20 }}
          ></i>
          <span>{modalMessage}</span>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button
            className="kp-btn-update"
            onClick={() => setShowErrorModal(false)}
          >
            OK
          </button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default OutreachEmail;
