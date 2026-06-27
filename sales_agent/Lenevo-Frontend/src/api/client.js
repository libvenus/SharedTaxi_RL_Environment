/**
 * API client for the Lenovo Opportunities FastAPI back-end.
 *
 * In dev, requests go through the Vite proxy (vite.config.js → /api).
 * In prod, set VITE_API_BASE_URL to point at the deployed back-end.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

/** Build a URL with non-empty query params. Arrays become comma-separated. */
function buildUrl(path, params = {}) {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  console.log("url", url);
  for (const [key, value] of Object.entries(params)) {
    if (
      value === undefined ||
      value === null ||
      value === "" ||
      value === "all"
    ) {
      continue;
    }
    if (Array.isArray(value)) {
      if (value.length === 0) continue;
      url.searchParams.set(key, value.join(","));
    } else {
      url.searchParams.set(key, String(value));
    }
  }
  return url.pathname + url.search;
}

async function request(
  path,
  { method = "GET", params, body, headers = {}, signal } = {},
) {
  const url = buildUrl(path, params);

  const res = await fetch(url, {
    method,
    signal,
    headers: {
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let bodyText = "";
    try {
      bodyText = await res.text();
    } catch {}

    throw new Error(
      `Request to ${url} failed (${res.status} ${res.statusText}). ${bodyText}`,
    );
  }

  return res.json();
}
// export function fetchMeetings({ filter = "all_meetings" } = {}, signal) {
//   return request("/meeting-prep/meetings", {
//     signal,
//     params: { filter },
//   });
// }
/* ---------- Meeting Prep -------------------------------------------- */
export function fetchMeetings({ filter = "all_meetings" } = {}, signal) {
  return request("ai-api/meeting-prep/meetings", {
    signal,
    params: { filter },
  });
}
export function createMeeting(body) {
  return request("/ai-api/meeting-prep", { method: "POST", body });
}
/* ---------- Outreach Studio ----------------------------------------- */
export function fetchOutreach({ category = "HIGH_PRIORITY" } = {}, signal) {
  return request("ai-api/outreach", {
    signal,
    params: { category },
  });
}
export function sendEmail(body) {
  return request("/ai-api/outreach/send-email", { method: "POST", body });
}
export function fetchTodoOutreach(todoId, signal) {
  return request(`ai-api/todos/to_do_outreach/${todoId}`, { signal });
}

/* AI generate draft */
export function generateEmailDraft(body) {
  return request("/ai-api/outreach/email-draft", { method: "POST", body });
}

/* ---------- Post Meeting Preview ----------------------------------- */
export function fetchPostMeetingReview(signal) {
  return request("ai-api/post-meeting-preview/meeting", {
    signal,
    params: { seller_id: "055dafe7-9840-451d-8328-5f70a6326c03" },
  });
}

export function fetchPostMeetingDetail(meetingId, signal) {
  return request(`ai-api/post-meeting-preview/${meetingId}`, {
    signal,
    params: { seller_id: "055dafe7-9840-451d-8328-5f70a6326c03" },
  });
}

export function addKeyPoint(meetingId, keyPoint) {
  return request(`ai-api/post-meeting-preview/${meetingId}/key-points`, {
    method: "POST",
    body: keyPoint,
  });
}

export function updateKeyPoint(meetingId, keyPointId, keyPoint) {
  return request(
    `ai-api/post-meeting-preview/${meetingId}/key-points/${keyPointId}`,
    {
      method: "PUT",
      body: keyPoint,
    },
  );
}

export function deleteKeyPoint(meetingId, keyPointId) {
  return request(
    `ai-api/post-meeting-preview/${meetingId}/key-points/${keyPointId}`,
    {
      method: "DELETE",
    },
  );
}

export function updateNextStep(meetingId, nextStepId, nextStep) {
  return request(
    `ai-api/post-meeting-preview/${meetingId}/nextSteps/${nextStepId}`,
    {
      method: "PUT",
      body: nextStep,
    },
  );
}

export function deleteNextStep(meetingId, nextStepId) {
  return request(
    `ai-api/post-meeting-preview/${meetingId}/nextSteps/${nextStepId}`,
    {
      method: "DELETE",
    },
  );
}

export function completeKeyPoints(meetingId) {
  return request(
    `ai-api/post-meeting-preview/${meetingId}/keypoints/complete`,
    {
      method: "PATCH",
    },
  );
}

export function completeNextSteps(meetingId) {
  return request(`ai-api/post-meeting-preview/${meetingId}/nextstep/complete`, {
    method: "PATCH",
  });
}

export function updateCrmField(meetingId, seqId, body) {
  return request(
    `ai-api/post-meeting-preview/${meetingId}/update_crm/${seqId}`,
    {
      method: "PUT",
      body,
    },
  );
}
export function updateMeetingSummary(meetingId, body) {
  return request(`/ai-api/post-meeting-preview/summary/${meetingId}`, {
    method: "PUT",
    body,
  });
}
/* ---------- Todos --------------------------------------------------- */
export function fetchTodos({ filter_type = "all", show_completed = false } = {}, signal) {
  const url = `ai-api/todos/?filter_type=${encodeURIComponent(filter_type)}&showCompleted=${show_completed}`;
  return request(url, { signal });
}

export function createTodo(body) {
  return request("ai-api/todos/create", { method: "POST", body });
}

export function updateTodo(todoId, body) {
  return request(`ai-api/todos/${todoId}`, { method: "PUT", body });
}

export function updateTodoStatus(todoId, status) {
  return request(`ai-api/todos/${todoId}/status`, {
    method: "PATCH",
    body: { status },
  });
}

/* ---------- #1  /api/opportunities/kpi-summary ----------------------- */
export function fetchKpiSummary(
  { comparePeriod, search, regions, industries, stages, products } = {},
  signal,
) {
  return request("/api/opportunities/kpi-summary", {
    signal,
    params: {
      comparePeriod,
      search,
      regions,
      industries,
      stages,
      products,
    },
  });
}
export function fetchAccountKpiSummary(
  {
    comparePeriod,
    search,
    regions,
    industries,
    accountTypes,
    accountStatuses,
    segments,
  } = {},
  signal,
) {
  return request("/api/accounts/kpi-summary", {
    signal,
    params: {
      comparePeriod,
      search,
      regions,
      industries,
      accountTypes,
      accountStatuses,
      segments,
    },
  });
}
/* ---------- #2  /api/opportunities ----------------------------------- */
export function fetchOpportunities(
  {
    page = 1,
    pageSize = 10,
    sortBy = "closeDate",
    sortOrder = "desc",
    search,
    regions,
    industries,
    stages,
    products,
    bucket,
    ownerId,
  } = {},
  signal,
) {
  return request("/api/opportunities", {
    signal,
    params: {
      page,
      pageSize,
      sortBy,
      sortOrder,
      search,
      regions,
      industries,
      stages,
      products,
      bucket,
      ownerId,
    },
  });
}

/* ---------- #5  /api/opportunities/{id}/competitors ------------------ */
export function fetchCompetitors(opportunityId, signal) {
  return request(
    `/api/opportunities/${encodeURIComponent(opportunityId)}/competitors`,
    { signal },
  );
}

export function updateCompetitor(opportunityId, body) {
  return request(
    `/api/opportunities/${encodeURIComponent(opportunityId)}/competitors`,
    { method: "POST", body },
  );
}

/* ---------- #13 /api/opportunities/{id}/sale-motion ------------------ */
export function fetchSaleMotion(opportunityId, signal) {
  return request(
    `/api/opportunities/${encodeURIComponent(opportunityId)}/sale-motion`,
    { signal },
  );
}
export function deleteOpportunity(opportunityId) {
  return request(`/api/opportunities/${encodeURIComponent(opportunityId)}`, {
    method: "DELETE",
  });
}
/* ---------- #8-#11  /api/filters/* ----------------------------------- */
export function fetchFilterOptions(signal) {
  return Promise.all([
    request("/api/filters/regions", { signal }),
    request("/api/filters/industries", { signal }),
    request("/api/filters/stages", { signal }),
    request("/api/filters/products", { signal }),
  ]).then(([regions, industries, stages, products]) => ({
    regions,
    industries,
    stages,
    products,
  }));
}

export function fetchAccounts(
  {
    page = 1,
    pageSize = 10,
    sortBy = "closeDate",
    sortOrder = "desc",
    bucket,
  } = {},
  signal,
) {
  return request("/api/accounts", {
    signal,
    params: {
      page,
      pageSize,
      sortBy,
      sortOrder,
      bucket,
    },
  });
}

/* ---------- Lookup: Accounts (for autocomplete) ---------------------- */
export function lookupAccounts(search = "", signal) {
  return request("/api/accounts", {
    signal,
    params: { search, page: 1, pageSize: 50 },
  });
}

/* ---------- Lookup: Opportunities (for autocomplete) ----------------- */
export function lookupOpportunities(search = "", signal) {
  return request("/api/opportunities", {
    signal,
    params: { search, page: 1, pageSize: 50 },
  });
}

export function fetchAccountDetails(accountId) {
  return request(`/api/accounts/${accountId}`);
}
export function fetchCustomerDetails(accountId) {
  return request(`/api/accounts/${accountId}/customer-information`);
}
export function fetchOpportunityDetails(Id) {
  return request(`/api/opportunities/${Id}`);
}
export function fetchAccountContactDetails(accountId) {
  return request(`/api/accounts/${accountId}/contacts`);
}
export function fetchOpportunityContactDetails(accountId) {
  return request(`/api/opportunities/${accountId}/contacts`);
}
export function addAccountContact(accountId, payload) {
  return request(`/api/accounts/${accountId}/contacts`, {
    method: "POST",
    body: payload, // ✅ FIXED
  });
}
export function updateAccountContact(accountId, contactLinkId, payload) {
  return request(`/api/accounts/${accountId}/contacts/${contactLinkId}`, {
    method: "PATCH",
    body: payload,
  });
}
export function editAccountContact(id, contactLinkId, payload) {
  return request(`/api/opportunities/${id}/contacts/${contactLinkId}`, {
    method: "PATCH",
    body: payload,
  });
}
export function updateDeal(Id, payload) {
  return request(`/api/opportunities/${Id}`, {
    method: "PATCH",
    body: payload,
  });
}

export function deleteAccountContact(accountId, contactLinkId) {
  return request(`/api/accounts/${accountId}/contacts/${contactLinkId}`, {
    method: "DELETE",
  });
}
export function deleteopportunityContact(id, contactLinkId) {
  return request(`/api/opportunities/${id}/contacts/${contactLinkId}`, {
    method: "DELETE",
  });
}
export function fetchAccountOpportunities(
  accountId,
  {
    page = 1,
    pageSize = 10,
    sortBy = "closeDate",
    sortOrder = "desc",
    search,
    regions,
    industries,
    stages,
    accountStatuses,
    accountTypes,
    segments,
  } = {},
  signal,
) {
  return request(`/api/accounts/${accountId}/opportunities`, {
    signal,
    params: {
      page,
      pageSize,
      sortBy,
      sortOrder,
      search,
      regions,
      industries,
      accountStatuses,
      accountTypes,
      segments,
    },
  });
}

export function fetchActivities(opportunityId, signal) {
  return request(
    `/api/opportunities/${encodeURIComponent(opportunityId)}/timeline?pageSize=50`,
    { signal },
  );
}
export function fetchAccountFilters(signal) {
  return request("/api/accounts/filters", {
    signal,
  });
}

export function fetchFilteredAccounts(params = {}, signal) {
  return request("/api/accounts", {
    signal,
    params,
  });
}

export function createOpportunityNote(opportunityId, notes) {
  return request("/api/opportunities/notes", {
    method: "POST",
    body: { opportunity_id: opportunityId, notes },
  });
}

export function fetchOpportunityNotes(opportunityId) {
  return request(
    `/api/opportunities/opportunity/${encodeURIComponent(opportunityId)}/notes`,
  );
}

export function updateOpportunity(opportunityId, payload) {
  return request(
    `/api/opportunities/opportunity_update/${encodeURIComponent(opportunityId)}`,
    {
      method: "PATCH",
      body: payload,
    },
  );
}

export function fetchQuarterPulse(sellerId, signal) {
  return request("/api/quarter-pulse", {
    signal,
    params: { sellerId },
  });
}

export function fetchQuarterPulseQuota(sellerId, signal) {
  return request("/api/quarter-pulse/quota", {
    signal,
    params: { sellerId },
  });
}

export function updateQuarterPulseQuota(sellerId, payload) {
  return request(
    `/api/quarter-pulse/quota?sellerId=${encodeURIComponent(sellerId)}`,
    {
      method: "PUT",
      body: payload,
    },
  );
}

export function fetchNotifications(sellerId, limit = 6, signal) {
  return request("/api/notifications", {
    signal,
    params: { sellerId, limit },
  });
}

export function fetchActivityTimeline(
  sellerId,
  { page = 1, pageSize = 25 } = {},
  signal,
) {
  return request("/api/activity-timeline", {
    signal,
    params: { sellerId, page, pageSize },
  });
}
export function generateEmailDraftV1({
  context,
  written_context,
  template,
  data = {},
  placeholders = {},
}) {
  return request("/v2/email/draft", {
    method: "POST",
    body: { context, written_context, data, template, placeholders },
  });
}

/* ---------- Bot Join Meeting ---------------------------------------- */
// The bot join is now proxied through the AIBackend so that credentials
// stay server-side and the backend can store the Vexa bot_id for webhook
// correlation and status tracking.
export async function joinMeetingBot({ meeting_id }) {
  const res = await fetch(`/ai-api/meeting-prep/${meeting_id}/join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`Bot join failed (${res.status}): ${detail}`);
  }
  return res.json();
}
// sales operating model interview questions apis
export function fetchInterviewQuestions(role, signal) {
  return request("/api/sales-operating-model/interview-questions", {
    signal,
    params: { role },
  });
}

export function fetchInterviewSetup(role, signal) {
  return request("/api/sales-operating-model/interview-setup", {
    signal,
    params: { role },
  });
}

export function createInterviewQuestion(body) {
  return request("/api/sales-operating-model/interview-questions", {
    method: "POST",
    body,
  });
}

export function updateInterviewQuestion(questionId, body) {
  const encodedId = encodeURIComponent(questionId);
  const normalizedBody = {
    questionText: body?.questionText,
    sortOrder: body?.sortOrder,
  };

  return request(
    `/api/sales-operating-model/interview-questions/${encodedId}/`,
    {
      method: "PUT",
      body: normalizedBody,
    },
  ).catch((err) => {
    if (!String(err?.message || "").includes("(405")) throw err;

    return request(
      `/api/sales-operating-model/interview-questions/${encodedId}/`,
      {
        method: "PATCH",
        body: normalizedBody,
      },
    );
  });
}

export function deleteInterviewQuestion(questionId) {
  return request(
    `/api/sales-operating-model/interview-questions/${encodeURIComponent(questionId)}`,
    {
      method: "DELETE",
    },
  );
}

export function saveInterviewSetupDraft(role, body) {
  return request(
    `/api/sales-operating-model/interview-setup/${encodeURIComponent(role)}/draft`,
    {
      method: "PUT",
      body,
    },
  );
}

export function saveInterviewSetup(role, body) {
  return request(
    `/api/sales-operating-model/interview-setup/${encodeURIComponent(role)}/save`,
    {
      method: "POST",
      body,
    },
  );
}
export function fetchOrganizationalIntentCards(signal) {
  return request("/api/sales-operating-model/organizational-intent-cards", {
    signal,
  });
}

export function fetchOrganizationalIntentCard(intentType, signal) {
  return request(
    `/api/sales-operating-model/organizational-intent-cards/${encodeURIComponent(intentType)}`,
    { signal },
  );
}

export function saveOrganizationalIntentCard(intentType, fields) {
  return request(
    `/api/sales-operating-model/organizational-intent-cards/${encodeURIComponent(intentType)}`,
    {
      method: "PUT",
      body: { fields },
    },
  );
}

export function deleteOrganizationalIntentCard(intentType) {
  return request(
    `/api/sales-operating-model/organizational-intent-cards/${encodeURIComponent(intentType)}`,
    {
      method: "DELETE",
    },
  );
}

export function deleteOrganizationalIntentCardField(intentType, metricId) {
  return request(
    `/api/sales-operating-model/organizational-intent-cards/${encodeURIComponent(intentType)}/fields/${encodeURIComponent(metricId)}`,
    { method: "DELETE" },
  );
}

/* ---------- Live Transcript ---------------------------------------- */
export function fetchTranscript(meetingId, signal) {
  return request(`/ai-api/transcripts/${meetingId}`, { signal });
}

/* ---------- Fiscal Period ------------------------------------------ */
export function fetchFiscalPeriod() {
  return request("/api/quarter-pulse/period");
}
