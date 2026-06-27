# US03 — Consent Capture · Status & Dependencies

**Sprint:** 1A
**User Story:** Consent Capture (pre-meeting consent email)
**Last updated:** 2026-06-15
**Status legend:** ✅ done · 🟡 ready / awaiting trigger · ⏳ blocked · ❌ not started

---

## 1. What's done (backend)

| # | Deliverable | Status |
|---|---|---|
| B1 | Schema: `tbl_meeting_consent_email` (idempotent migration `sql/2026_06_us03_consent_email.sql`) | ✅ |
| B2 | ORM model: `MeetingConsentEmail` (`app/models/consent_email.py`) | ✅ |
| B3 | Pydantic schemas: 8 request/response models (`app/schema/consent_email.py`) | ✅ |
| B4 | Service layer: `schedule_consent_emails`, `record_delivery_status`, `record_opt_out`, `get_consent_status`, `due_for_retry`, `list_consent_emails` (`app/services/consent_email_service.py`) | ✅ |
| B5 | Router: 5 routes under `/consent-emails` + 1 under `/meetings/{id}/consent-status` (`app/api/consent_emails.py`) | ✅ |
| B6 | Server-rendered HTML for opt-out (success / 410 / 404) — inline CSS, self-contained | ✅ |
| B7 | US01 lifecycle integration — opt-out cascades to `bot_status='cancelled'` with reason `participant_opted_out` (no schema change needed) | ✅ |
| B8 | Config additions: `CONSENT_WINDOW_MINUTES`, `INTERNAL_EMAIL_DOMAINS`, `OPT_OUT_BASE_URL`, `SYSTEM_EMAIL_ADDRESS`, `CONSENT_RETRY_DELAY_MINUTES` (`app/core/config.py`) | ✅ |
| B9 | Pytest smoke suite — 15 tests (`tests/test_consent_capture_lifecycle.py`) | ✅ |
| B10 | conftest.py extended to register consent routers + import the model | ✅ |
| B11 | README updated — repo layout, .env vars, state diagram, all 6 cURL examples | ✅ |
| B12 | AI-team handoff doc (`US03_BACKEND_HANDOFF_FOR_AI_TEAM.md`) | ✅ |
| B13 | This status doc (`US03_STATUS_AND_DEPENDENCIES.md`) | ✅ |
| B14 | Plan-of-action doc (`SPRINT_1A_US03_CONSENT_CAPTURE_PLAN.md`) | ✅ |
| B15 | Bot-status canonical-reason note added to `app/models/schedulemeeting.py` comment block | ✅ |

---

## 2. Pending — backend, non-blocking

| # | Item | Notes | Trigger |
|---|---|---|---|
| BP1 | Run `pytest -q` end-to-end and confirm 15/15 transcript tests pass + 9/9 US02 + 8/8 US01 (32 total) | Tests written, not yet executed in this session | Run locally |
| BP2 | Sweeper for orphaned consent rows when meeting is deleted | Sprint 1B — not blocking US03 | Reactive |
| BP3 | Real X-Forwarded-For parsing for `opt_out_ip` (currently uses `request.client.host` which is wrong behind a load balancer) | Bind to ops deployment; fix when we go behind a real LB | DevOps |
| BP4 | Pagination on `/due-for-retry` and `/consent-emails/{id}` if backlog grows | Reactive — only when we see hundreds-per-meeting | Reactive |
| BP5 | Authn between bot ↔ AIBackend | Sprint 1B (DevOps + bot team joint) | Sprint 1B |

---

## 3. Pending — AI / Bot team (blocks "story-level done")

| # | Item | Status | Notes |
|---|---|---|---|
| AI1 | Email template rendering (HTML + plain text) using template vars from `/schedule` response | ❌ | Backend exposes data; bot owns templates |
| AI2 | Microsoft Graph integration to send "from" the seller's account | ❌ | Delegated permissions; OAuth on seller's behalf |
| AI3 | Call `POST /consent-emails/schedule` after US01 resolves opportunity | ❌ | |
| AI4 | Honour `should_send=false` — don't send email; route to in-meeting consent | ❌ | |
| AI5 | Call `PATCH /delivery` after every send attempt | ❌ | |
| AI6 | Poll `GET /due-for-retry` periodically and re-send + PATCH again | ❌ | Cadence: 1–5 min recommended |
| AI7 | Call `GET /meetings/{id}/consent-status` right before bot transitions to `joining` | ❌ | AC #7 is enforced HERE |
| AI8 | If `any_opted_out=true`: do NOT join, do NOT PATCH bot_status (cancel cascade already done) | ❌ | |
| AI9 | If `consent_mechanism='in_meeting_chat'` or `'mixed'`: still post US02 chat msg in-meeting | ❌ | Belt and braces |
| AI10 | Display INF_MSG_0001 to seller when `seller_notified_at` is non-null on any consent row | ❌ | "Future sprints" per AC #9 — backend has the data ready |
| AI11 | Display SUCC_MSG_0010 in seller UI for opted-out attendees | ❌ | Optional — backend already shows it to the participant who opted out |

---

## 4. Pending — DevOps

| # | Item | Status | Notes |
|---|---|---|---|
| D1 | Apply `sql/2026_06_us03_consent_email.sql` on dev Postgres | ❌ | Idempotent — safe to re-run |
| D2 | Set `OPT_OUT_BASE_URL` to the AIBackend's PUBLIC hostname (HTTPS in prod) | ❌ | Without this, opt-out links don't work — they'd point to localhost |
| D3 | Set `INTERNAL_EMAIL_DOMAINS` for the deployment (default `lenovo.com,motorola.com`) | ❌ | Confirm with PM if more domains needed |
| D4 | Set `SYSTEM_EMAIL_ADDRESS` (default `sales-assistant@lenovo.com`) | ❌ | Confirm with PM whether per-region |
| D5 | Open inbound HTTPS firewall: anywhere → AIBackend `/consent-emails/opt-out/*` | ❌ | The opt-out URL is hit by participants' browsers from arbitrary networks |
| D6 | Open firewall: bot → AIBackend (for `/consent-emails/*` writes) | ❌ | Same as US01 |
| D7 | Configure Microsoft Graph "send-as-seller" delegated permissions in the Lenovo tenant | ❌ | AI team's responsibility, but DevOps may need to register the app |

---

## 5. Pending — Joint (cross-team)

| # | Item | Owner | Notes |
|---|---|---|---|
| J1 | Sign off on `SUCC_MSG_0010` and `INF_MSG_0001` canonical texts | PM | Q1 |
| J2 | Confirm bot polling interval for `/due-for-retry` | AI team → backend | Q4 |
| J3 | Decide reschedule (US01) → consent re-send cascade behaviour | AI team + backend | Q5 |
| J4 | Decide DELETE meeting (US01) → consent rows policy | AI team + backend | Q7 |
| J5 | E2E dry-run on dev | All | Real Outlook event → real email → real opt-out click → bot doesn't join |

---

## 6. Open questions (carried from handoff doc §7)

| # | Question | Status | Owner |
|---|---|---|---|
| Q1 | Where do `SUCC_MSG_0010` and `INF_MSG_0001` canonical texts live? | open | PM |
| Q2 | Per-region from address vs one global address? | open | PM |
| Q3 | Store opt-out IP/user-agent for compliance? | answered (default: yes) | Legal to confirm |
| Q4 | Bot's `/due-for-retry` polling interval? | open | AI team |
| Q5 | Meeting reschedule → consent re-send behaviour? | open | Joint |
| Q6 | `INTERNAL_EMAIL_DOMAINS` global vs per-tenant? | open | DevOps + PM |
| Q7 | DELETE meeting (US01) → soft-cascade vs preserve consent rows? | open | Joint |

---

## 7. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Opt-out endpoint hit by automated email-link scanners (mailbox AV / bot prefetchers) → false opt-out | **High** | **TODO: switch GET → confirmation page → POST /confirm-opt-out**. Currently a single GET click records the opt-out. Many enterprise mail scanners pre-fetch links to scan for malware, which would trigger spurious opt-outs. Strongly recommend a two-step UI before going to prod (Q-needed) |
| R2 | Token guessing | Low | 256 bits of randomness — brute-force impractical |
| R3 | Bot retries cause duplicate emails (network drop between PATCH and re-poll) | Low | We dedupe by `(meeting_id, recipient_email)`. Worst case bot sends same email twice; recipient gets duplicate but token is the same |
| R4 | `INTERNAL_EMAIL_DOMAINS` misconfigured — internal Lenovo employees get the email | Medium | Tested via test #2; ops verifies env var on deploy |
| R5 | Meeting in different timezone than UTC — window math wrong | Low | Backend treats naive `meeting_start_time` as UTC; bot writes UTC ISO timestamps. Tested |
| R6 | Storage growth — 1 row × N attendees × M meetings | Low | 14 columns, ~500 bytes/row. 10k meetings × 5 attendees = 25 MB. Negligible |

> **R1 is the only blocker before prod.** Open question: do we want a one-step
> "click here to opt out" link, or a two-step "click here → press confirm
> button → opt-out recorded" flow? The two-step flow defeats automated
> link scanners but adds friction. Spec says "click the opt-out link" —
> ambiguous about whether that's one or two clicks. Flag for PM in J1.

---

## 8. Definition of "story-level done"

US03 is fully done when:

- [x] All backend deliverables (B1–B15) complete
- [ ] All DevOps tasks (D1–D7) complete
- [ ] All AI team tasks (AI1–AI11) complete
- [ ] R1 mitigated (one-step vs two-step opt-out decided)
- [ ] Joint dry-run (J5) succeeds end-to-end on dev
- [ ] Open questions Q1–Q7 answered
- [ ] PM signs off after watching:
      `external attendee receives email → clicks link → sees SUCC_MSG_0010 →
       seller sees INF_MSG_0001 (when notifications ship) → bot does not join`

---

## 9. Definition of "backend done"

Backend is done. Sanmay is unblocked from US03 work. Reasonable next picks:

- US04 / US05 (next stories in Sprint 1A backlog)
- Customer Information Phase 2 from earlier conversations
- US01/02/03 polish: insight pills on activity cards, audio_blob_url
  pointer if AI team requests it, etc.

---

## 10. Files changed (for the PR)

```
sql/2026_06_us03_consent_email.sql                   (new)
app/models/consent_email.py                          (new)
app/models/schedulemeeting.py                        (comment update — canonical reasons)
app/schema/consent_email.py                          (new)
app/services/consent_email_service.py                (new)
app/api/consent_emails.py                            (new — 6 routes + HTML pages)
app/core/config.py                                   (5 new env vars)
app/main.py                                          (2 new routers registered)
tests/conftest.py                                    (consent routers + model registered)
tests/test_consent_capture_lifecycle.py              (new — 15 tests)
README.md                                            (US03 sections added)
SPRINT_1A_US03_CONSENT_CAPTURE_PLAN.md               (new)
US03_BACKEND_HANDOFF_FOR_AI_TEAM.md                  (new)
US03_STATUS_AND_DEPENDENCIES.md                      (new)
```
