import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
// import "../../styles/execute.css";
import "../../styles/admin.css";
import {
  createInterviewQuestion,
  deleteOrganizationalIntentCard,
  deleteOrganizationalIntentCardField,
  deleteInterviewQuestion,
  fetchInterviewSetup,
  fetchInterviewQuestions,
  fetchOrganizationalIntentCard,
  saveInterviewSetup,
  saveInterviewSetupDraft,
  updateInterviewQuestion,
  saveOrganizationalIntentCard,
} from "../../api/client";

const ROLE_API_MAP = {
  national: "national_manager",
  regional: "regional_manager",
  seller: "seller_manager",
};

const ROLE_SCOPE_LABELS = {
  national: "Scope: Org-level intent and global guardrails",
  regional: "Scope: Region-specific behavior and constraints",
  seller: "Scope: Team-level execution rules and behaviors",
};

const DEFAULT_QUESTIONS = [
  { questionId: null, sortOrder: 1, text: "What does a successful quarter mean beyond quota?" },
  { questionId: null, sortOrder: 2, text: "Should we optimize for growth, stability, or defense this quarter?" },
  { questionId: null, sortOrder: 3, text: "Which motion is primary this quarter?" },
  { questionId: null, sortOrder: 4, text: "What constraints are non-negotiable this quarter?" },
  { questionId: null, sortOrder: 5, text: "When goals conflict, what wins first?" },
];

// Default field definitions for organizational intent cards
const INTENT_CARD_FIELDS = {
  outcome: {
    revenueAndQuality: "Success = 70% revenue + 30% gross margin; services attach target 42%",
    predictability: "Forecast accuracy within 5%; stable commit pipeline",
    progressionExpectation: "Advance 2 stages by week 8 of quarter",
    riskPosture: "Discount tolerance low: max 12% without executive approval",
    additionalContext: "Multi-quarter terms preferred over one-off discount wins",
  },
  motion: {
    primaryGrowthLever: "60% expansion/upsell, 25% net-new logos, 15% renewal protection",
    sellingMotionMix: "Solution-led with ThinkPad entry where needed",
    routeToMarket: "Direct-led in enterprise; partner-led mandatory for public sector",
    salesCyclePolicy: "Fast < 90d; Strategic up to 730d with exec sponsor",
    attachExpectation: "DaaS attach on deals > $500K ACV",
  },
  focus: {
    quarterType: "Harvest quarter through Week 10; pipeline-build mini-window Weeks 11-12",
    priorityFocus: "Strategic accounts: Infosys, Alphatec, Meridian Health",
    temporaryDeprioritisation: "SMB segment paused until next quarter",
    expiryDate: "2026-09-30",
    additionalContext: "Renew focus overrides before expiry",
  },
  behavioral: {
    multithreadingNorm: "Minimum 2 stakeholders by Qualification, 3 by Proposal for enterprise",
    followUpCadence: "High-priority: every 3 business days; medium: every 7 days",
    walkAwayRule: "No technical buyer and no budget signal after 45 days — exit deal",
    coachingLens: "Use for reinforcement and explanation, not policy enforcement",
  },
  constraint: {
    marginFloors: [
      { dealType: "hardware-only", minPercent: 18 },
      { dealType: "bundled", minPercent: 24 },
    ],
    complianceGates: "Healthcare/security deals require CISO checklist before proposal closure",
    dealDeskTriggers: "Discounts > 10%, non-standard terms, or multi-country contracts",
    pricingAuthority: "Seller up to 8%; manager up to 12%; director up to 15%; above needs deal council",
  },
  tradeoff: {
    priorityRank: [
      "commit_accuracy",
      "margin_protection",
      "revenue_upside",
      "new_logo_count",
    ],
    revenueVsMargin: "Margin wins when projected GM < target floor by > 3 points",
    newLogoVsExpansion: "Expansion wins in Q2 unless strategic logo is board-tagged",
    commitVsUpside: "Committed deals protected first in Weeks 9-12 of quarter",
  },
};

export default function SalesOperatingModel() {
  const navigate = useNavigate();
  const [activeRole, setActiveRole] = useState("national");

  const [questions, setQuestions] = useState(DEFAULT_QUESTIONS);
  const [newQuestionTitle, setNewQuestionTitle] = useState("");
  const [newQuestionInput, setNewQuestionInput] = useState("");
  const [editingQuestionTitle, setEditingQuestionTitle] = useState(false);
  const [editingQuestionIndex, setEditingQuestionIndex] = useState(null);
  const [editingQuestionText, setEditingQuestionText] = useState("");
  const [questionResponsesByRole, setQuestionResponsesByRole] = useState({
    national: {},
    regional: {},
    seller: {},
  });
  const [savedResponsesByRole, setSavedResponsesByRole] = useState({
    national: {},
    regional: {},
    seller: {},
  });
  const [scopeLabelsByRole, setScopeLabelsByRole] = useState({ ...ROLE_SCOPE_LABELS });
  const [showVerificationStep, setShowVerificationStep] = useState(false);
  const [isCommittingResponses, setIsCommittingResponses] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const draftTimerRef = useRef(null);

  const [outcomeMetrics, setOutcomeMetrics] = useState([
    { title: "Revenue & Quality", desc: "Success = 15% revenue + 30% gross margin; services attach target 42%; multi-quarter terms preferred over one-off discount wins" },
    { title: "Predictability", desc: "Forecast accuracy target +/-5%; pipeline coverage stability (1.25x-1.45x) prioritized over speculative upside this quarter." },
    { title: "Progressive Expectation", desc: "30% of Discovery deals should reach Qualification by Week 8 and 22% should reach Proposal by Week 9" },
    { title: "Risk Posture", desc: "Leadership mode = balanced lean defense for current quarter; discount tolerance low (max 12% without executive approval)." },
  ]);
  const [newOutcomeTitle, setNewOutcomeTitle] = useState("");
  const [newOutcomeDesc, setNewOutcomeDesc] = useState("");
  const [editingOutcomeTitle, setEditingOutcomeTitle] = useState(false);

  const [motionMetrics, setMotionMetrics] = useState([
    { title: "Primary growth leve", desc: "60% expansion/upsell, 25% net new logos, 15% renewal protection." },
    { title: "Selling motion mix", desc: "Solution-led first, with hardware as entry wedge and services as margin amplifier." },
    { title: "Route-to-market", desc: "Direct-led in enterprise; partner-led mandatory for public sector and tier 2 geo accounts." },
    { title: "Sales cycle policy", desc: "Long cycle allowed if ACV > ₹1.5C; sub-₹25L opportunities should target ~45-day cycle." },
    { title: "Attach expectation", desc: "DaaS/financing attach mandatory for enterprise refreshes; opportunistic for SMB transactional deals." },
  ]);
  const [newMotionTitle, setNewMotionTitle] = useState("");
  const [newMotionDesc, setNewMotionDesc] = useState("");
  const [editingMotionTitle, setEditingMotionTitle] = useState(false);

  const [focusMetrics, setFocusMetrics] = useState([
    { title: "Quarter type", desc: "Harvest quarter through Week 10; pipeline build into window zappers in Weeks 11-13." },
    { title: "Priority focus", desc: "BFG enterprise refresh + Healthcare standardization campaign; strategic account override active for Infosys and HDFC group entities." },
    { title: "Temporary deprioritization", desc: "Education SMB and non-intent rated bundles paused until next quarter." },
    { title: "Equity controls", desc: "Current focus override auto-expire 30-Jun-2025 unless explicitly renewed." },
  ]);
  const [newFocusTitle, setNewFocusTitle] = useState("");
  const [newFocusDesc, setNewFocusDesc] = useState("");
  const [editingFocusTitle, setEditingFocusTitle] = useState(false);

  const [behavioralMetrics, setBehavioralMetrics] = useState([
    { title: "Multithreading norm", desc: "Minimum 3 stakeholders by Qualification and 5 by Proposal for all enterprise opportunities." },
    { title: "Follow up cadence", desc: "High-priority opportunities require touch every 3 business days; medium every 7 days." },
    { title: "Walk away rule", desc: "Reps are encouraged to exit deals with no technical buyer + no budget signal after 45 days." },
    { title: "Coaching lens", desc: "Metric used for enrichment and explanation, not policy enforcement." },
  ]);
  const [newBehavioralTitle, setNewBehavioralTitle] = useState("");
  const [newBehavioralDesc, setNewBehavioralDesc] = useState("");
  const [editingBehavioralTitle, setEditingBehavioralTitle] = useState(false);

  const [constraintMetrics, setConstraintMetrics] = useState([
    { title: "Margin floors", desc: "10% absolute floor for hardware only; 24% for bundled offers; non-overridable by agents." },
    { title: "Compliance gates", desc: "Healthcare/public deals require CSO-checklist completion before proposal closure." },
    { title: "Deal desk triggers", desc: "Mandatory review for discounts >35%; non-standard terms; or multi-country contracts." },
    { title: "Pricing authority", desc: "Seller up to 8%; manager up to 12%; director up to 15%; anything higher needs deal council." },
  ]);
  const [newConstraintTitle, setNewConstraintTitle] = useState("");
  const [newConstraintDesc, setNewConstraintDesc] = useState("");
  const [editingConstraintTitle, setEditingConstraintTitle] = useState(false);

  const [tradeoffMetrics, setTradeoffMetrics] = useState([
    { title: "Trade-off priority rank", desc: "1) Commit accuracy, 2) Margin protection, 3) Revenue upside, 4) New-logo count." },
    { title: "Revenue vs margin", desc: "Margin wins when projected GM < target floor by >3 points" },
    { title: "New logo vs expansion", desc: "Expansion wins in Q2 unless strategic logo is board-tagged" },
    { title: "Commit vs upside", desc: "Committed deals always protected first in Weeks 9-12 of quarter." },
  ]);
  const [newTradeoffTitle, setNewTradeoffTitle] = useState("");
  const [newTradeoffDesc, setNewTradeoffDesc] = useState("");
  const [editingTradeoffTitle, setEditingTradeoffTitle] = useState(false);

  // Organizational Intent Card State
  const [cardDataByType, setCardDataByType] = useState({});
  const [editingCard, setEditingCard] = useState(null);
  const [editingField, setEditingField] = useState(null);
  const [editingValue, setEditingValue] = useState("");
  const [savingCard, setSavingCard] = useState(null);
  const [saveError, setSaveError] = useState("");
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [newFieldTitleByCard, setNewFieldTitleByCard] = useState({});
  const [newFieldDescByCard, setNewFieldDescByCard] = useState({});
  const [editingNewFieldTitleByCard, setEditingNewFieldTitleByCard] = useState({});

  const loadInterviewQuestionsByRole = async (role, signal) => {
    const roleKey = Object.keys(ROLE_API_MAP).find((key) => ROLE_API_MAP[key] === role) || activeRole;
    let data;

    try {
      data = await fetchInterviewSetup(role, signal);
    } catch {
      data = await fetchInterviewQuestions(role, signal);
    }

    const rawQuestions = Array.isArray(data)
      ? data
      : Array.isArray(data?.questions)
        ? data.questions
        : [];

    const list = rawQuestions
      .filter((q) => q && (typeof q === "string" || typeof q?.text === "string" || typeof q?.questionText === "string"))
      .map((q, idx) => {
        if (typeof q === "string") {
          return {
            questionId: null,
            sortOrder: idx + 1,
            text: q,
            role,
          };
        }

        return {
          questionId: q.questionId || q.question_id || null,
          sortOrder: q.sortOrder || q.sort_order || idx + 1,
          text: q.text || q.questionText || "",
          role: q.role || role,
        };
      })
      .filter((q) => q.text?.trim())
      .sort((a, b) => (a.sortOrder || 0) - (b.sortOrder || 0));

    setQuestions(list.length ? list : DEFAULT_QUESTIONS);

    const draftResponses = data?.responses && typeof data.responses === "object" ? data.responses : {};
    const savedResponses = data?.savedResponses && typeof data.savedResponses === "object" ? data.savedResponses : {};

    setQuestionResponsesByRole((prev) => ({
      ...prev,
      [roleKey]: {
        ...(prev[roleKey] || {}),
        ...draftResponses,
      },
    }));

    setSavedResponsesByRole((prev) => ({
      ...prev,
      [roleKey]: savedResponses,
    }));

    if (data?.scopeLabel) {
      setScopeLabelsByRole((prev) => ({
        ...prev,
        [roleKey]: data.scopeLabel,
      }));
    }
  };

  const handleSaveQuestion = async () => {
    const questionText = newQuestionInput.trim();
    if (!questionText) return;

    const role = ROLE_API_MAP[activeRole] || ROLE_API_MAP.national;

    try {
      await createInterviewQuestion({
        role,
        questionText,
        sortOrder: questions.length + 1,
      });

      await loadInterviewQuestionsByRole(role);
      setNewQuestionInput("");
      setNewQuestionTitle("");
      setEditingQuestionTitle(false);
      setShowVerificationStep(false);
    } catch (err) {
      console.error("Failed to save interview question:", err);
    }
  };

  const handleEditInterviewQuestion = (question, idx) => {
    setEditingQuestionIndex(idx);
    setEditingQuestionText(question?.text || "");
  };

  const handleCancelEditInterviewQuestion = () => {
    setEditingQuestionIndex(null);
    setEditingQuestionText("");
  };

  const handleSaveEditedInterviewQuestion = async (question, idx) => {
    const questionText = editingQuestionText.trim();
    if (!questionText) return;

    const role = ROLE_API_MAP[activeRole] || ROLE_API_MAP.national;

    try {
      if (question?.questionId) {
        await updateInterviewQuestion(question.questionId, {
          questionText,
          sortOrder: String(question.sortOrder || idx + 1),
        });
        await loadInterviewQuestionsByRole(role);
      } else {
        setQuestions((prev) =>
          prev.map((q, i) => (i === idx ? { ...q, text: questionText } : q)),
        );
      }

      setEditingQuestionIndex(null);
      setEditingQuestionText("");
      setShowVerificationStep(false);
    } catch (err) {
      console.error("Failed to edit interview question:", err);
    }
  };

  const handleDeleteInterviewQuestion = async (question, idx) => {
    const role = ROLE_API_MAP[activeRole] || ROLE_API_MAP.national;

    try {
      if (question?.questionId) {
        await deleteInterviewQuestion(question.questionId);
        await loadInterviewQuestionsByRole(role);
      } else {
        setQuestions((prev) => prev.filter((_, i) => i !== idx));
      }
    } catch (err) {
      console.error("Failed to delete interview question:", err);
    }
  };

  const getResponseKey = (question, idx) => question?.questionId || `idx_${idx}`;

  const getRoleResponsesPayload = (roleKey, roleResponses) => {
    const source = roleResponses || {};
    return questions
      .map((q, idx) => ({
        questionId: q?.questionId,
        text: (source[getResponseKey(q, idx)] || "").trim(),
      }))
      .filter((item) => item.questionId && item.text);
  };

  const saveDraftResponses = async (roleKey, roleResponses) => {
    const apiRole = ROLE_API_MAP[roleKey] || ROLE_API_MAP.national;
    const responses = getRoleResponsesPayload(roleKey, roleResponses);
    if (!responses.length) return;

    try {
      await saveInterviewSetupDraft(apiRole, { responses });
    } catch (err) {
      console.error("Failed to save interview draft:", err);
    }
  };

  const handleResponseChange = (question, idx, value) => {
    const roleKey = activeRole;
    const responseKey = getResponseKey(question, idx);

    setQuestionResponsesByRole((prev) => {
      const nextRoleResponses = {
        ...(prev[roleKey] || {}),
        [responseKey]: value,
      };

      if (draftTimerRef.current) {
        clearTimeout(draftTimerRef.current);
      }

      draftTimerRef.current = setTimeout(() => {
        saveDraftResponses(roleKey, nextRoleResponses);
      }, 700);

      return {
        ...prev,
        [roleKey]: nextRoleResponses,
      };
    });
  };

  const getResponseValue = (question, idx) => {
    const roleResponses = questionResponsesByRole[activeRole] || {};
    return roleResponses[getResponseKey(question, idx)] || "";
  };

  const getSavedResponseValue = (question) => {
    const savedResponses = savedResponsesByRole[activeRole] || {};
    return savedResponses[question?.questionId] || "";
  };

  const capturedResponseCount = questions.reduce((count, q, idx) => {
    const value = getResponseValue(q, idx);
    return value.trim() ? count + 1 : count;
  }, 0);

  const openReviewPanel = () => {
    setReviewError("");
    setShowVerificationStep(true);
  };

  const handleCommitInterviewResponses = async () => {
    const apiRole = ROLE_API_MAP[activeRole] || ROLE_API_MAP.national;
    const roleResponses = questionResponsesByRole[activeRole] || {};
    const responses = questions
      .map((q, idx) => ({
        questionId: q?.questionId,
        text: (roleResponses[getResponseKey(q, idx)] || "").trim(),
      }))
      .map((item) => ({
        questionId: item.questionId,
        text: item.text.trim(),
      }))
      .filter((item) => item.questionId && item.text);

    if (!responses.length) return;

    try {
      setIsCommittingResponses(true);
      setReviewError("");
      await saveInterviewSetup(apiRole, { responses });

      setQuestionResponsesByRole((prev) => ({
        ...prev,
        [activeRole]: {},
      }));
      setSavedResponsesByRole((prev) => ({
        ...prev,
        [activeRole]: responses.reduce((acc, item) => {
          acc[item.questionId] = item.text;
          return acc;
        }, {}),
      }));
      setShowVerificationStep(false);
    } catch (err) {
      setReviewError("ERR_MSG_0021: Configuration could not be saved. Your responses are preserved - please try again.");
      console.error("Failed to commit interview responses:", err);
    } finally {
      setIsCommittingResponses(false);
    }
  };

  const handleClearResponses = () => {
    if (draftTimerRef.current) {
      clearTimeout(draftTimerRef.current);
    }

    setQuestionResponsesByRole((prev) => ({
      ...prev,
      [activeRole]: {},
    }));
    setShowVerificationStep(false);
  };

  const addOutcomeMetric = () => {
    if (!newOutcomeTitle.trim() && !newOutcomeDesc.trim()) return;
    setOutcomeMetrics([...outcomeMetrics, { title: newOutcomeTitle.trim() || "Untitled Metric", desc: newOutcomeDesc.trim() }]);
    setNewOutcomeTitle("");
    setNewOutcomeDesc("");
    setEditingOutcomeTitle(false);
  };

  const addMotionMetric = () => {
    if (!newMotionTitle.trim() && !newMotionDesc.trim()) return;
    setMotionMetrics([...motionMetrics, { title: newMotionTitle.trim() || "Untitled Metric", desc: newMotionDesc.trim() }]);
    setNewMotionTitle("");
    setNewMotionDesc("");
    setEditingMotionTitle(false);
  };

  const addFocusMetric = () => {
    if (!newFocusTitle.trim() && !newFocusDesc.trim()) return;
    setFocusMetrics([...focusMetrics, { title: newFocusTitle.trim() || "Untitled Metric", desc: newFocusDesc.trim() }]);
    setNewFocusTitle("");
    setNewFocusDesc("");
    setEditingFocusTitle(false);
  };

  const addBehavioralMetric = () => {
    if (!newBehavioralTitle.trim() && !newBehavioralDesc.trim()) return;
    setBehavioralMetrics([...behavioralMetrics, { title: newBehavioralTitle.trim() || "Untitled Metric", desc: newBehavioralDesc.trim() }]);
    setNewBehavioralTitle("");
    setNewBehavioralDesc("");
    setEditingBehavioralTitle(false);
  };

  const addConstraintMetric = () => {
    if (!newConstraintTitle.trim() && !newConstraintDesc.trim()) return;
    setConstraintMetrics([...constraintMetrics, { title: newConstraintTitle.trim() || "Untitled Metric", desc: newConstraintDesc.trim() }]);
    setNewConstraintTitle("");
    setNewConstraintDesc("");
    setEditingConstraintTitle(false);
  };

  const addTradeoffMetric = () => {
    if (!newTradeoffTitle.trim() && !newTradeoffDesc.trim()) return;
    setTradeoffMetrics([...tradeoffMetrics, { title: newTradeoffTitle.trim() || "Untitled Metric", desc: newTradeoffDesc.trim() }]);
    setNewTradeoffTitle("");
    setNewTradeoffDesc("");
    setEditingTradeoffTitle(false);
  };

  const handleDeleteIntentCard = async (intentType) => {
    if (!intentType) return;

    try {
      await deleteOrganizationalIntentCard(intentType);
      try {
        await fetchCardData(intentType);
      } catch {
        setCardDataByType((prev) => ({
          ...prev,
          [intentType]: {
            intentType,
            displayName: intentType.charAt(0).toUpperCase() + intentType.slice(1),
            status: "NOT_CONFIGURED",
            fields: {},
            fieldLabels: {},
          },
        }));
      }
    } catch (err) {
      console.error(`Failed to delete ${intentType} intent card:`, err);
    }
  };

  // Fetch individual card data with fields
  const fetchCardData = async (cardType) => {
    try {
      const data = await fetchOrganizationalIntentCard(cardType);
      setCardDataByType((prev) => ({
        ...prev,
        [cardType]: data,
      }));
      return data;
    } catch (err) {
      console.error(`Failed to fetch card data for ${cardType}:`, err);
      return null;
    }
  };

  // Start editing a field
  const startEditField = (cardType, fieldKey) => {
    const cardData = cardDataByType[cardType];
    const rawValue = (cardData?.fields && cardData.fields[fieldKey]) ?? INTENT_CARD_FIELDS[cardType]?.[fieldKey] ?? "";
    const currentValue = Array.isArray(rawValue)
      ? (typeof rawValue[0] === "object" && rawValue[0] !== null
        ? JSON.stringify(rawValue, null, 2)
        : rawValue.join("\n"))
      : String(rawValue);
    setEditingCard(cardType);
    setEditingField(fieldKey);
    setEditingValue(currentValue);
    setSaveError("");
    setSaveSuccess(false);
  };

  // Cancel editing
  const cancelEdit = () => {
    setEditingCard(null);
    setEditingField(null);
    setEditingValue("");
    setSaveError("");
  };

  // Save card field via API
  const saveCardField = async (cardType) => {
    if (!editingCard || !editingField) return;

    setSavingCard(cardType);
    setSaveError("");
    setSaveSuccess(false);

    try {
      // Start with default fields or existing card data
      const defaultFields = INTENT_CARD_FIELDS[cardType] || {};
      const cardData = cardDataByType[cardType];
      const existingFields = cardData?.fields || {};

      // Merge: existing data + default fields + updated value
      const updatedFields = {
        ...defaultFields,
        ...existingFields,
        [editingField]: editingValue,
      };

      await saveOrganizationalIntentCard(cardType, updatedFields);

      // Update local state with new card data
      setCardDataByType((prev) => ({
        ...prev,
        [cardType]: {
          ...prev[cardType],
          fields: updatedFields,
        },
      }));

      setSaveSuccess(true);
      cancelEdit();

      // Refresh this card data to update CONFIGURED status without page reload
      try {
        await fetchCardData(cardType);
      } catch (refreshErr) {
        console.warn(`Failed to refresh ${cardType} card after save:`, refreshErr);
      }

      // Clear success message after 3 seconds
      setTimeout(() => {
        setSaveSuccess(false);
      }, 3000);
    } catch (err) {
      console.error(`Failed to save card field for ${cardType}:`, err);
      setSaveError(err.message || "Failed to save. Please try again.");
    } finally {
      setSavingCard(null);
    }
  };

  // Add a new field to a card locally
  const addFieldToCard = (cardType) => {
    const title = (newFieldTitleByCard[cardType] || "").trim();
    const desc = (newFieldDescByCard[cardType] || "").trim();
    if (!title && !desc) return;
    const fieldKey = title
      ? title.charAt(0).toLowerCase() + title.slice(1).replace(/\s+(.)/g, (_, c) => c.toUpperCase()).replace(/\s+/g, "")
      : `field_${Date.now()}`;
    setCardDataByType((prev) => ({
      ...prev,
      [cardType]: {
        ...(prev[cardType] || {}),
        fields: {
          ...(INTENT_CARD_FIELDS[cardType] || {}),
          ...(prev[cardType]?.fields || {}),
          [fieldKey]: desc,
        },
        fieldLabels: {
          ...(prev[cardType]?.fieldLabels || {}),
          [fieldKey]: title || fieldKey,
        },
      },
    }));
    setNewFieldTitleByCard((prev) => ({ ...prev, [cardType]: "" }));
    setNewFieldDescByCard((prev) => ({ ...prev, [cardType]: "" }));
    setEditingNewFieldTitleByCard((prev) => ({ ...prev, [cardType]: false }));
  };

  // Delete a field from a card via API
  const deleteFieldFromCard = async (cardType, fieldKey) => {
    try {
      await deleteOrganizationalIntentCardField(cardType, fieldKey);
      // Remove from local state after successful API call
      setCardDataByType((prev) => {
        const prevFields = { ...(prev[cardType]?.fields || {}) };
        delete prevFields[fieldKey];
        return {
          ...prev,
          [cardType]: {
            ...prev[cardType],
            fields: prevFields,
          },
        };
      });
    } catch (err) {
      console.error(`Failed to delete field ${fieldKey} from ${cardType}:`, err);
      setSaveError(`Failed to delete field: ${err.message || "Unknown error"}`);
      setTimeout(() => setSaveError(""), 3000);
    }
  };

  useEffect(() => {
    window.scrollTo(0, 0);

    // Fetch all card types directly
    const cardTypes = ["outcome", "motion", "focus", "behavioral", "constraint", "tradeoff"];
    Promise.all(cardTypes.map((cardType) => fetchCardData(cardType)))
      .catch((err) => console.error("Failed to fetch organizational intent cards:", err));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const role = ROLE_API_MAP[activeRole] || ROLE_API_MAP.national;

    setEditingQuestionIndex(null);
    setEditingQuestionText("");
    setNewQuestionInput("");
    setNewQuestionTitle("");
    setEditingQuestionTitle(false);
    setShowVerificationStep(false);

    loadInterviewQuestionsByRole(role, controller.signal)
      .catch((err) => {
        if (err?.name === "AbortError") return;
        console.error("Failed to fetch interview setup:", err);
        setQuestions(DEFAULT_QUESTIONS);
      });

    return () => controller.abort();
  }, [activeRole]);

  useEffect(() => {
    return () => {
      if (draftTimerRef.current) {
        clearTimeout(draftTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="dv-page">
      <div className="main">
        <div className="quarter-bar">
          Q2 FY2024 · Week 10 of 12 ·{" "}
          <span className="closure-badge">Closure Phase</span>
        </div>

        <div className="page-header">
          <div className="page-header-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}>
            <span className="topbar-title">Admin: Sales Operating Model</span>
            <button className="som-add-btn" title="Add new section"><i className="bi bi-plus"></i></button>
          </div>
          <div className="tabs-outer">
            <div className="tabs">
              <button
                className="tab"
                onClick={() => navigate("/admin")}
              >
                Event Spline
              </button>
              <button className="tab active">
                Sales Operating Model
              </button>
            </div>
          </div>
        </div>

        <div className="content">
          <div className="admin-col-left" style={{ width: "100%" }}>

            {/* Interview-first setup */}
            <div className="som-card">
              <div className="som-card-header">
                <div className="som-card-title">Interview-first setup</div>
                <div className="som-tab-group">
                  <button className={`som-tab ${activeRole === "national" ? "active" : ""}`} onClick={() => setActiveRole("national")}>National Manager</button>
                  <button className={`som-tab ${activeRole === "regional" ? "active" : ""}`} onClick={() => setActiveRole("regional")}>Regional Manager</button>
                  <button className={`som-tab ${activeRole === "seller" ? "active" : ""}`} onClick={() => setActiveRole("seller")}>Seller Manager</button>
                </div>
              </div>
              <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 10 }}>
                {scopeLabelsByRole[activeRole] || ROLE_SCOPE_LABELS[activeRole]}
              </div>

              {!showVerificationStep && (
                <>
                  {questions.map((q, idx) => (
                    <div className="som-q-block" key={idx}>
                      <div className="som-q-header">
                        <div className="som-q-title" style={{ flex: 1 }}>
                          {editingQuestionIndex === idx ? (
                            <input
                              type="text"
                              value={editingQuestionText}
                              onChange={(e) => setEditingQuestionText(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                  e.preventDefault();
                                  handleSaveEditedInterviewQuestion(q, idx);
                                }
                                if (e.key === "Escape") {
                                  e.preventDefault();
                                  handleCancelEditInterviewQuestion();
                                }
                              }}
                              autoFocus
                              style={{ width: "100%", fontWeight: 600, fontSize: 14, color: "#111827", border: "1px solid #d1d5db", borderRadius: 6, padding: "6px 10px", outline: "none", background: "#fff" }}
                            />
                          ) : (
                            <>
                              {idx + 1}. &nbsp;{q.text}
                              <i className="bi bi-question-circle-fill" style={{ color: "#111827", fontSize: 13, marginLeft: 6 }}></i>
                            </>
                          )}
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          {editingQuestionIndex === idx ? (
                            <>
                              <button className="som-icon-btn som-edit" onClick={() => handleSaveEditedInterviewQuestion(q, idx)}><i className="bi bi-check-lg"></i></button>
                              <button className="som-icon-btn som-del" onClick={handleCancelEditInterviewQuestion}><i className="bi bi-x-lg"></i></button>
                            </>
                          ) : (
                            <button className="som-icon-btn som-edit" onClick={() => handleEditInterviewQuestion(q, idx)}><i className="bi bi-pencil"></i></button>
                          )}
                          <button className="som-icon-btn som-del" onClick={() => handleDeleteInterviewQuestion(q, idx)}><i className="bi bi-trash3"></i></button>
                        </div>
                      </div>
                      <textarea
                        className="som-note-input"
                        placeholder={getSavedResponseValue(q) || "Record response..."}
                        value={getResponseValue(q, idx)}
                        onChange={(e) => handleResponseChange(q, idx, e.target.value)}
                      />
                    </div>
                  ))}

                  {/* Add New Question */}
                  <div className="som-q-block">
                    <div className="som-q-header">
                      <div className="som-q-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        {editingQuestionTitle ? (
                          <input
                            type="text"
                            value={newQuestionTitle}
                            onChange={(e) => setNewQuestionTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && newQuestionTitle.trim()) {
                                setEditingQuestionTitle(false);
                              }
                            }}
                            onBlur={() => setEditingQuestionTitle(false)}
                            autoFocus
                            placeholder="Enter new question"
                            style={{ fontWeight: 600, fontSize: 14, color: "#111827", border: "none", outline: "none", background: "transparent", width: "100%" }}
                          />
                        ) : (
                          <span style={{ color: newQuestionTitle ? "#111827" : "#9ca3b8" }}>
                            {newQuestionTitle || "New Question"}
                          </span>
                        )}
                        <button className="som-icon-btn som-edit" onClick={() => setEditingQuestionTitle(true)}><i className="bi bi-pencil"></i></button>
                      </div>
                    </div>
                    <div style={{ position: "relative" }}>
                      <textarea
                        className="som-note-input"
                        placeholder="Type a new question and press add"
                        value={newQuestionInput}
                        onChange={(e) => setNewQuestionInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey && newQuestionInput.trim()) {
                            e.preventDefault();
                            handleSaveQuestion();
                          }
                        }}
                        style={{ paddingRight: 40 }}
                      />
                      <button
                        className="som-icon-btn"
                        style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "#9ca3af" }}
                        onClick={handleSaveQuestion}
                      >
                        <i className="bi bi-plus-circle"></i>
                      </button>
                    </div>
                  </div>

                  <div className="som-form-actions">
                    <button className="som-btn-clear" onClick={handleClearResponses}>Clear All</button>
                    <button className="admin-btn-save" disabled={capturedResponseCount === 0} onClick={openReviewPanel}>
                      Verify & Edit Before Save
                    </button>
                  </div>
                </>
              )}

              {showVerificationStep && (
                <div style={{ marginTop: 14, border: "1px solid #e5e7eb", borderRadius: 10, padding: 14, background: "#f8fafc" }}>
                  <div style={{ fontWeight: 700, color: "#111827", marginBottom: 10 }}>Verification step</div>

                  {questions.map((q, idx) => {
                    const responseText = getResponseValue(q, idx).trim();
                    return (
                      <div key={`${q.questionId || "q"}_${idx}`} style={{ marginBottom: 10 }}>
                        <div style={{ fontWeight: 600, color: "#111827", marginBottom: 4 }}>{idx + 1}. {q.text}</div>
                        <div style={{ color: responseText ? "#111827" : "#6b7280", fontSize: 13 }}>
                          {responseText || "No response captured yet."}
                        </div>
                      </div>
                    );
                  })}

                  {reviewError && (
                    <div style={{ marginBottom: 10, color: "#b91c1c", fontSize: 13 }}>{reviewError}</div>
                  )}

                  <div style={{ display: "flex", gap: 8 }}>
                    <button className="som-btn-clear" onClick={() => setShowVerificationStep(false)} disabled={isCommittingResponses}>
                      Continue Editing
                    </button>
                    <button className="admin-btn-save" onClick={handleCommitInterviewResponses} disabled={isCommittingResponses}>
                      {isCommittingResponses ? "Saving..." : "Save Verified Responses"}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* ══════════ ORGANIZATIONAL INTENT SETUP ══════════ */}
            <div className="som-card">
              <div className="som-card-header">
                <div className="som-card-title">Organizational Intent Setup</div>
                <button className="som-add-btn" title="Add new section"><i className="bi bi-plus"></i></button>
              </div>

              {/* Render each intent card dynamically from API */}
              {["outcome", "motion", "focus", "behavioral", "constraint", "tradeoff"].map((cardType) => {
                const cardData = cardDataByType[cardType];

                const displayName = cardData?.displayName || (cardType.charAt(0).toUpperCase() + cardType.slice(1));
                const status = cardData?.status || "NOT_CONFIGURED";
                
                // Use default fields or API fields (merged)
                const defaultFields = INTENT_CARD_FIELDS[cardType] || {};
                const apiFields = cardData?.fields || {};
                const fields = { ...defaultFields, ...apiFields };
                
                const fieldLabels = cardData?.fieldLabels || {};
                const fieldKeys = Object.keys(fields);

                return (
                  <div key={cardType}>
                    {/* Card Header */}
                    <div className="som-outcome-block" style={{ border: "none", borderRadius: 0, marginTop: 16, paddingBottom: 0 }}>
                      <div className="som-outcome-header">
                        <div className="som-outcome-title">
                          {displayName}
                          <i className="bi bi-question-circle-fill" style={{ color: "#111827", fontSize: 13, marginLeft: 6 }}></i>
                          <span style={{ marginLeft: 8, fontSize: 11, fontWeight: 700, color: status === "CONFIGURED" ? "#166534" : "#6b7280" }}>
                            {status}
                          </span>
                        </div>
                        <div className="som-outcome-actions">
                          <button className="som-icon-btn som-del" onClick={() => handleDeleteIntentCard(cardType)} title="Delete this card">
                            <i className="bi bi-trash3"></i>
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Card Fields */}
                    <div className="som-outcome-block">
                      {fieldKeys.length > 0 ? (
                        fieldKeys.map((fieldKey) => {
                          const fieldValue = fields[fieldKey] || "";
                          const fieldLabel = fieldLabels[fieldKey] || fieldKey.replace(/([A-Z])/g, " $1").trim();
                          const isEditing = editingCard === cardType && editingField === fieldKey;
                          
                          // Format field value display (handle arrays)
                          let displayValue;
                          if (Array.isArray(fieldValue)) {
                            if (fieldValue.length === 0) {
                              displayValue = "—";
                            } else if (typeof fieldValue[0] === "object" && fieldValue[0] !== null) {
                              displayValue = fieldValue.map((item) =>
                                Object.entries(item).map(([k, v]) => `${k}: ${v}`).join(", ")
                              ).join("\n");
                            } else {
                              displayValue = fieldValue.map((v, i) => `${i + 1}. ${v}`).join("\n");
                            }
                          } else {
                            displayValue = fieldValue || "—";
                          }

                          return (
                            <div className="som-outcome-row" key={fieldKey}>
                              <div style={{ flex: 1 }}>
                                <div style={{ fontWeight: 700, fontSize: 13, color: "#111827", marginBottom: 2 }}>
                                  {fieldLabel}
                                </div>
                                {isEditing ? (
                                  <textarea
                                    value={editingValue}
                                    onChange={(e) => setEditingValue(e.target.value)}
                                    style={{
                                      flex: 1,
                                      fontSize: 12.5,
                                      color: "#374151",
                                      lineHeight: 1.5,
                                      border: "1px solid #d1d5db",
                                      borderRadius: 4,
                                      padding: "8px 10px",
                                      outline: "none",
                                      background: "#fff",
                                      fontFamily: "inherit",
                                      minHeight: 60,
                                      width: "100%",
                                      resize: "vertical",
                                    }}
                                    autoFocus
                                  />
                                ) : (
                                  <div style={{ fontSize: 12.5, color: "#6b7280", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
                                    {displayValue || "—"}
                                  </div>
                                )}
                              </div>
                              <div className="som-outcome-actions">
                                {isEditing ? (
                                  <>
                                    <button
                                      className="som-icon-btn som-edit"
                                      onClick={() => saveCardField(cardType)}
                                      disabled={savingCard === cardType}
                                      title="Save field"
                                    >
                                      <i className="bi bi-check-lg"></i>
                                    </button>
                                    <button
                                      className="som-icon-btn som-del"
                                      onClick={cancelEdit}
                                      disabled={savingCard === cardType}
                                      title="Cancel"
                                    >
                                      <i className="bi bi-x-lg"></i>
                                    </button>
                                  </>
                                ) : (
                                  <>
                                    <button
                                      className="som-icon-btn som-edit"
                                      onClick={() => startEditField(cardType, fieldKey)}
                                      title="Edit field"
                                    >
                                      <i className="bi bi-pencil"></i>
                                    </button>
                                    <button
                                      className="som-icon-btn som-del"
                                      onClick={() => deleteFieldFromCard(cardType, fieldKey)}
                                      title="Delete field"
                                    >
                                      <i className="bi bi-trash3"></i>
                                    </button>
                                  </>
                                )}
                              </div>
                            </div>
                          );
                        })
                      ) : (
                        <div style={{ fontSize: 13, color: "#6b7280", padding: "16px" }}>
                          No fields configured.
                        </div>
                      )}
                    </div>

                    {/* New Field Input */}
                    <div className="som-outcome-block">
                      <div className="som-new-row" style={{ flexDirection: "column", alignItems: "stretch", gap: 8 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          {editingNewFieldTitleByCard[cardType] ? (
                            <input
                              type="text"
                              value={newFieldTitleByCard[cardType] || ""}
                              onChange={(e) => setNewFieldTitleByCard((prev) => ({ ...prev, [cardType]: e.target.value }))}
                              onKeyDown={(e) => e.key === "Enter" && setEditingNewFieldTitleByCard((prev) => ({ ...prev, [cardType]: false }))}
                              onBlur={() => setEditingNewFieldTitleByCard((prev) => ({ ...prev, [cardType]: false }))}
                              autoFocus
                              placeholder="Enter metric title"
                              style={{ fontWeight: 700, fontSize: 13, color: "#111827", border: "1px solid #e2e8f0", borderRadius: 4, padding: "4px 8px", outline: "none", background: "#f8fafc", width: 220 }}
                            />
                          ) : (
                            <span style={{ fontWeight: 700, fontSize: 13, color: "#111827" }}>
                              {newFieldTitleByCard[cardType] || "New Metric"}
                            </span>
                          )}
                          <button
                            className="som-icon-btn som-edit"
                            onClick={() => setEditingNewFieldTitleByCard((prev) => ({ ...prev, [cardType]: true }))}
                          >
                            <i className="bi bi-pencil"></i>
                          </button>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 4, padding: "0 12px", height: 40 }}>
                          <input
                            type="text"
                            value={newFieldDescByCard[cardType] || ""}
                            onChange={(e) => setNewFieldDescByCard((prev) => ({ ...prev, [cardType]: e.target.value }))}
                            onKeyDown={(e) => e.key === "Enter" && addFieldToCard(cardType)}
                            placeholder="New recommendation"
                            style={{ flex: 1, fontSize: 13, color: "#374151", border: "none", outline: "none", background: "transparent" }}
                          />
                          <button className="som-icon-btn" style={{ color: "#9ca3af" }} onClick={() => addFieldToCard(cardType)}>
                            <i className="bi bi-plus-circle"></i>
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Card Actions - Cancel & Save Buttons */}
                    <div className="som-form-actions">
                      <button
                        className="som-btn-clear"
                        onClick={cancelEdit}
                        disabled={savingCard === cardType}
                      >
                        Cancel
                      </button>
                      <button
                        className="admin-btn-save"
                        onClick={() => saveCardField(cardType)}
                        disabled={savingCard !== null || editingCard !== cardType}
                      >
                        {savingCard === cardType ? "Saving..." : `Save `}
                      </button>
                    </div>
                  </div>
                );
              })}

              {/* Error and Success Messages */}
              {saveError && (
                <div
                  style={{
                    marginTop: 12,
                    padding: 12,
                    background: "#fee2e2",
                    border: "1px solid #fecaca",
                    borderRadius: 6,
                    color: "#b91c1c",
                    fontSize: 13,
                  }}
                >
                  ⚠️ {saveError}
                </div>
              )}

              {saveSuccess && (
                <div
                  style={{
                    marginTop: 12,
                    padding: 12,
                    background: "#dcfce7",
                    border: "1px solid #bbf7d0",
                    borderRadius: 6,
                    color: "#166534",
                    fontSize: 13,
                  }}
                >
                  ✓ Configuration saved successfully!
                </div>
              )}


            </div>

            {/* ══════════ SALES CYCLE TRACKER ══════════ */}
            <div className="som-card">
              <div className="som-card-header">
                <div className="som-card-title">Sales Cycle Tracker</div>
                <button className="som-add-btn" title="Add new section"><i className="bi bi-plus"></i></button>
              </div>

              <div className="som-outcome-block">
                {[
                  { title: "1. Tempo classes", items: [
                    "Deals defined: Transactional (no resource; SMB below 4Wks); Quarterly/Enterprise (mid size 4Wks); Strategic (Multiple quarters; partner framework team)",
                    "Default → Quarterly/Enterprise escalation → Strategic/Multiple (partner framework team)",
                  ]},
                  { title: "2. Predictability", items: [
                    "Each anchor — Tam customer seeking: Enterprise anchor = reference/standards; Programmable anchor = RR squad",
                    "No anchor: proceed with manager deal (manager reviews cycle)",
                    "3+/4 score events: legal freeze; procurement blackout; government election code.",
                  ]},
                  { title: "3. Signal Expectations by Time Band", items: [
                    "Week 1-4: expected solution validation and pricing; discount: Flat annually is acceptable up to $50",
                    "W5-W8: expected commercial/legal movement; no flag for missing technical deep dive in strategic class.",
                    "Minimum area solutions during phase = stakeholder maps vs collaborative risk meeting",
                  ]},
                  { title: "4. General and Default Activations", items: [
                    "Activation condition: Q2 evaluation procurement; Q2 enterprise within-budget design class.",
                    "Activity normally in final 2 weeks of fiscal year close.",
                    "Per activation activity flagged at week/key activation (list only).",
                    "(If activation never occurs within 21 days, send to multi-deal queue)",
                  ]},
                  { title: "5. Acceleration and Decay", items: [
                    "Acceleration markers: new economic buyer engaged; legal review started; budget codes confirmed.",
                    "Decay signals: >14 one-sided activity; reconfirmed, scope regression; missing risk escalation.",
                    "Decay zones threshold: 30d (fast); 7Q (enterprise); 7Q (strategic)",
                    "Decay note: account notes, legal coverage before risk escalation.",
                  ]},
                  { title: "6. Multi-Year and Programmable Deals", items: [
                    "Lifecycle until programmatic = first-year strategy + multi-year program milestone",
                    "Formal checkpoints: annual budget approved; board signs cycle; partner framework renewals.",
                    "Strategic deal enabler: sponsor continuity + roadmap checkpoint.",
                    "Programmatic vs demand (label): is created or defined by design.",
                  ]},
                  { title: "7. Commercial Quarter Timelines", items: [
                    "Week 1-4 = pipeline creation dominant target 40% of top opportunities",
                    "Week 5-9 = qualification + continuing dominant target 75% (with progression)",
                    "Week 8-11 = negotiation = closure dominant target 75% (control projection)",
                    "Agents coordinate approval signals differently from current 12th.",
                  ]},
                  { title: "8. Multi-Year Timeline", items: [
                    "Q1 Sprint",
                    "Q2 Sprint",
                    "Q3 Sprint",
                    "Q4 Sprint",
                  ]},
                ].map((section, si) => (
                  <div key={si}>
                    <div className="som-outcome-row" style={{ borderBottom: "none" }}>
                      <div style={{ flex: 1, fontWeight: 700, fontSize: 13, color: "#111827" }}>{section.title}</div>
                    </div>
                    {section.items.map((item, ii) => (
                      <div className="som-outcome-row" key={ii}>
                        <div style={{ flex: 1, fontSize: 12.5, color: "#374151", lineHeight: 1.55 }}>{item}</div>
                        <div className="som-outcome-actions">
                          <button className="som-icon-btn som-edit"><i className="bi bi-pencil"></i></button>
                          <button className="som-icon-btn som-del"><i className="bi bi-trash3"></i></button>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>

              <div className="som-form-actions">
                <button className="som-btn-clear">Cancel</button>
                <button className="admin-btn-save">Save</button>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
