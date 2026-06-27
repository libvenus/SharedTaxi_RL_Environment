import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchAccountKpiSummary,
  fetchAccounts,
  fetchAccountFilters,
  fetchFilteredAccounts
} from "../../api/client";
import {
  buildPageList,
  formatCurrencyShort,
  formatDateMMDDYY,
  formatDealCount,
  formatDelta,
} from "../../utils/format";
import greenArrowIcon from "../../assets/icons/green.png";
import redArrowIcon from "../../assets/icons/red.png";
import downloadIcon from "../../assets/download.png";
/* ─────────────────────────────────────────────
   GLOBAL STYLES  (injected once into <head>)
───────────────────────────────────────────── */
// const STYLES = `
// @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

// :root {
//   --opd-border:     #CBD5E1;
//   --opd-muted:      #9ca3af;
//   --opd-accent:     #1a56db;
//   --opd-bg:         #f4f6f9;
//   --opd-card:       #ffffff;
//   --opd-text:       #111827;
//   --opd-text2:      #374151;
//   --opd-text3:      #6b7280;
// }

// /* reset */
// .opd-root *, .opd-root *::before, .opd-root *::after { box-sizing: border-box; margin: 0; padding: 0; }
// .opd-root { font-family: 'DM Sans', sans-serif; background: var(--opd-bg); min-height: 100vh; color: var(--opd-text); font-size: 13px; }

// /* ── Page ── */
// .opd-page { padding: 44px 0px; }
// .opd-h1 { font-size: 28px; font-weight: 700; margin-bottom: 16px; }

// /* ── Period tabs ── */
// .opd-period-tabs {height:40px; display: flex; border: 1px solid var(--opd-border); border-radius: 8px; overflow: hidden; width: fit-content; background: #fff; margin-bottom: 16px; }
// .opd-period-tab { padding: 7px 16px; border: none; border-right: 1px solid var(--opd-border); background: transparent; font-family: 'DM Sans', sans-serif; font-size: 14px; color: var(--opd-text3); cursor: pointer; transition: background .15s, color .15s; white-space: nowrap; }
// .opd-period-tab:last-child { border-right: none; }
// .opd-period-tab.opd-period-active { background: #e8f0fe; color: var(--opd-accent); font-weight: 600;height:40px; }

// /* ── Summary cards ── */
// .opd-summary-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; }
// .opd-summary-col { flex: 1 1 calc(25% - 12px); min-width: 220px; }
// @media(max-width:992px) { .opd-summary-col { flex: 1 1 calc(50% - 8px); } }
// @media(max-width:576px) { .opd-summary-col { flex: 1 1 100%; } }

// .opd-scard { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: hidden; text-align: center; display: flex; flex-direction: column; cursor: pointer; transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; user-select: none; height: 96px; }
// .opd-scard:hover { box-shadow: 0 2px 10px rgba(0,0,0,.08); border-color: #c7d2e3; }
// .opd-scard-active { border-color: #1a56db; box-shadow: 0 0 0 2px rgba(26,86,219,.18); }
// .opd-scard-active .opd-scard-head { background: #eff4ff; }
// .opd-scard-active .opd-scard-label { color: #1a56db; }
// .opd-scard-head { padding: 6px 10px; border-bottom: 1px solid var(--opd-border); background: #fff; }
// .opd-scard-label { font-size: 12px; color: var(--opd-text3); margin: 0; font-weight: 600; }
// .opd-scard-body { padding: 6px 10px; display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
// .opd-scard-value { font-size: 16px; font-weight: 700; margin-bottom: 2px; }
// .opd-scard-meta { display: flex; align-items: center; justify-content: center; gap: 6px; font-size: 12px; }
// .opd-scard-count { color: var(--opd-text3); }
// .opd-scard-sep   { color: #dee2e6; }
// .opd-delta-up    { color: var(--opd-text3); font-weight: 600; display: flex; align-items: center; gap: 3px; }
// .opd-delta-down  { color: var(--opd-text3); font-weight: 600; display: flex; align-items: center, gap: 3px; }
// .opd-delta-icon  { width: 15px; height: 15px; display: inline-block; object-fit: contain; }

// /* ── Filter bar ── */
// .opd-filter-bar { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; width: 100%; }
// .opd-search-box input { border: none; outline: none; font-size: 14px; color: #495057; background: transparent; width: 100%; font-family: 'DM Sans', sans-serif; }
// .opd-search-icon { color: #475569; flex-shrink: 0; height: 18px;
//     width: 18px;}
// .opd-btn-dl { width: 40px; height: 40px; border-radius: 50px; border:none; background: #E2E8F0; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--opd-text3); font-size: 16px; flex-shrink: 0; }
// .opd-scroll-panel table {
//   width: 100%;
//   border-collapse: collapse;
//   table-layout: fixed; /* 🔥 this locks column widths */
// }
// /* ── Custom dropdown ── */
// .opd-dd-wrap { position: relative; flex: 1 1 140px; min-width: 120px; }
// .opd-dd-trigger { display: flex; align-items: center; border: 2px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; font-size: 12px; color: #495057; background: #f8fafc; cursor: pointer; white-space: nowrap; min-width: 110px; width: 100%; user-select: none; transition: border-color .15s; font-family: 'DM Sans', sans-serif;height:40px }
// .opd-dd-trigger.opd-dd-open { border-color: var(--opd-accent); border-radius: 6px 6px 0 0; }
// .opd-dd-trigger span { flex: 1; }
// .opd-dd-arrow { width: 10px; height: 10px; margin-left: 10px; flex-shrink: 0; transition: transform .15s; }
// .opd-dd-trigger.opd-dd-open .opd-dd-arrow { transform: rotate(180deg); }
// .opd-dd-menu { position: absolute; top: 100%; left: 0; right: 0; background: #fff; border-top: none; border-radius: 0 0 6px 6px; z-index: 500; overflow: hidden; box-shadow: 0 4px 4px rgb(0 0 0 / 21%); }
// .opd-dd-item { display: flex; align-items: center; gap: 7px; padding: 7px 10px; font-size: 12px; cursor: pointer; color: #495057; transition: background .1s; }
// .opd-dd-item:hover { background: #f1f5ff; }
// .opd-dd-item.opd-dd-selected { color: var(--opd-accent); background: #f1f5f9; font-weight: 500; }

// /* ── Table card ── */
// .opd-table-card { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: hidden; }
// .opd-table-wrap { overflow-x: auto; }
// .opd-table-wrap table { border-collapse: collapse; width: 100%; min-width: 1200px; }
// .opd-th { padding: 12px 14px; text-align: left; font-weight: 600; font-size: 13px; color: #212529; white-space: nowrap; background: #f8fafc; border-bottom: 1px solid var(--opd-border); }
// .opd-td { padding: 12px 14px; height: 54px; vertical-align: middle; font-size: 13px; border-bottom: 1px solid #CBD5E1; }
// .opd-row-even { background: #fff; }
// .opd-row-odd  { background: #fff; }
// .opd-row-even:hover, .opd-row-odd:hover { background: #f8fafc; }
// .opd-no-results { padding: 28px 16px; text-align: center; color: var(--opd-muted); font-size: 13px; }

// /* checkbox - sticky */
// .opd-checkbox-col { width: 44px; text-align: center; position: sticky; left: 0; z-index: 2; background: #f8fafc; }
// .opd-checkbox { width: 16px; height: 16px; cursor: pointer; accent-color: var(--opd-accent); }

// /* name col - sticky */
// .opd-name-col { font-weight: 600; color: #0f172a; position: sticky; left: 44px; z-index: 2; background: #f8fafc; min-width: 180px; }
// .opd-th.opd-name-col { background: #f8fafc; }

// /* frozen columns shadow */
// .opd-name-col::after { content: ''; position: absolute; top: 0; right: -8px; bottom: 0; width: 8px; background: linear-gradient(to right, rgba(0,0,0,0.06), transparent); pointer-events: none; }

// /* ── Table footer ── */
// .opd-table-footer { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-top: 1px solid var(--opd-border); background: #f8fafc; flex-wrap: wrap; gap: 8px; }
// .opd-pag-row { display: flex; align-items: center; gap: 3px; }
// .opd-pag-btn { width: 26px; height: 26px; border-radius: 5px; border: 1px solid var(--opd-border); background: #fff; color: #495057; font-size: 11px; cursor: pointer; font-family: 'DM Sans', sans-serif; transition: background .1s; }
// .opd-pag-btn.opd-pag-active { background: var(--opd-accent); color: #fff; border-color: var(--opd-accent); font-weight: 600; }

// /* status badge */
// .opd-status-badge { padding: 4px 12px; border-radius: 0px; font-size: 12px; font-weight: 500; display: inline-block; }
// .opd-status-active { background: #dcfce7; color: #166534; }
// .opd-status-at-risk { background: #fee2e2; color: #991b1b; }
// .opd-status-warning { background: #ffedd5; color: #9a3412; }

// /* status with solid backgrounds */
// .opd-status-badge.opd-status-active { background: #22C55E; color: #fff;     padding: 4px 10px; width:65px}
// .opd-status-badge.opd-status-at-risk { background: #EF4444; color: #fff;    padding: 4px 10px; }
// .opd-status-badge.opd-status-pending, .opd-status-badge.opd-status-warning { background: #F97316;     padding: 4px 10px;color: #fff; }

// /* trend */
// .opd-trend-up { color: #16a34a; font-weight: 500; }
// .opd-trend-down { color: #dc2626; font-weight: 500; }

// .opp-header{ display: flex; align-items: center; justify-content: space-between; margin-top: 14px; }

// @media(max-width:768px) {
//   .opd-page { padding: 12px; }
// }
//   .opd-action-btn{
//       border: none;
//     background: none;
//   }
// `;


const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --opd-border:     #CBD5E1;
  --opd-muted:      #9ca3af;
  --opd-accent:     #1a56db;
  --opd-bg:         #f4f6f9;
  --opd-card:       #ffffff;
  --opd-text:       #111827;
  --opd-text2:      #374151;
  --opd-text3:      #6b7280;
}

/* reset */
.opd-root *, .opd-root *::before, .opd-root *::after { box-sizing: border-box; margin: 0; padding: 0; }
.opd-root { background: var(--opd-bg); height: 100vh; overflow: hidden; color: var(--opd-text); font-size: 14px; padding: 20px; width: 100%; max-width: 100%; }
/* ── Page ── */
.opd-page { padding: 44px 0px; height: 100%; overflow-y: auto; overflow-x: hidden; max-width: 100%; }
.opd-h1 { font-size: 28px; font-weight: 700; margin-bottom: 16px; }

/* ── Period tabs ── */
.opd-period-tabs {height:40px; display: flex; border: 1px solid var(--opd-border); border-radius: 8px; overflow: hidden; width: fit-content; background: #fff; margin-bottom: 16px; }
.opd-period-tab { padding: 7px 16px; border: none; border-right: 1px solid var(--opd-border); background: transparent; font-family: 'DM Sans', sans-serif; font-size: 14px; color: var(--opd-text3); cursor: pointer; transition: background .15s, color .15s; white-space: nowrap; }
.opd-period-tab:last-child { border-right: none; }
.opd-period-tab.opd-period-active { background: #e8f0fe; color: var(--opd-accent); font-weight: 600;height:40px; }

/* ── Summary cards ── */
.opd-summary-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; }
.opd-summary-col { flex: 1 1 calc(25% - 12px); min-width: 220px; }
@media(max-width:992px) { .opd-summary-col { flex: 1 1 calc(50% - 8px); } }
@media(max-width:576px) { .opd-summary-col { flex: 1 1 100%; } }

.opd-scard { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: visible; text-align: center; display: flex; flex-direction: column; cursor: pointer; transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; user-select: none; min-height: 96px; position: relative; }
.opd-scard:hover { box-shadow: 0 2px 10px rgba(0,0,0,.08); border-color: #c7d2e3; z-index: 20; }
.opd-scard:focus-visible { z-index: 20; }

.opd-scard[data-tip]::after,
.opd-scard[data-tip]::before { opacity: 0; pointer-events: none; transition: opacity .12s ease; }

.opd-scard[data-tip]::after {
  content: attr(data-tip);
  position: absolute;
  left: 50%;
  bottom: calc(100% + 10px);
  transform: translateX(-50%);
  background: #0f172a;
  color: #fff;
  font-size: 11px;
  font-weight: 500;
  line-height: 1.3;
  padding: 6px 8px;
  border-radius: 6px;
  white-space: nowrap;
  box-shadow: 0 4px 10px rgba(0,0,0,.2);
}

.opd-scard[data-tip]::before {
  content: "";
  position: absolute;
  left: 50%;
  bottom: calc(100% + 4px);
  transform: translateX(-50%);
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 6px solid #0f172a;
}

.opd-scard[data-tip]:hover::after,
.opd-scard[data-tip]:hover::before,
.opd-scard[data-tip]:focus-visible::after,
.opd-scard[data-tip]:focus-visible::before { opacity: 1; }
.opd-scard-active { border-color: #1a56db; box-shadow: 0 0 0 2px rgba(26,86,219,.18); }
.opd-scard-active .opd-scard-head { background: #eff4ff; }
.opd-scard-active .opd-scard-label { color: #1a56db; }
.opd-scard-head { padding: 6px 10px; border-bottom: 1px solid var(--opd-border); background: #fff; }
.opd-scard-label { font-size: 12px; color: var(--opd-text3); margin: 0; font-weight: 600; }
.opd-scard-body { padding: 6px 10px; display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
.opd-scard-value { font-size: 16px; font-weight: 700; margin-bottom: 2px; }
.opd-scard-meta { display: flex; align-items: center; justify-content: center; gap: 6px; font-size: 12px; }
.opd-scard-count { color: var(--opd-text3); }
.opd-scard-sep   { color: #dee2e6; }
.opd-delta-up    { color: var(--opd-text3); font-weight: 600; display: flex; align-items: center; gap: 3px; }
.opd-delta-down  { color: var(--opd-text3); font-weight: 600; display: flex; align-items: center; gap: 3px; }
.opd-delta-icon  { width: 15px; height: 15px; display: inline-block; object-fit: contain; }

/* ── Filter bar ── */
.opd-filter-bar { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; width: 100%; }
.opd-search-box { display: flex; align-items: center; gap: 6px; border: 2px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; min-width: 160px; flex: 1 1 220px; background: #f8fafc; height:40px}
.opd-search-box input { border: none; outline: none; font-size: 14px; color: #495057; background: transparent; width: 100%;!important; }
.opd-search-icon { color: #475569; flex-shrink: 0; height: 18px; width: 18px; }
.opd-btn-dl { width: 40px; height: 40px; border-radius: 50px; border:none; background: #E2E8F0; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--opd-text3); font-size: 16px; flex-shrink: 0; }

/* ── Custom dropdown ── */
.opd-dd-wrap { position: relative; flex: 1 1 140px; min-width: 120px; }
.opd-dd-trigger { display: flex; align-items: center; border: 2px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; font-size: 12px; color: #495057; background: #f8fafc; cursor: pointer; white-space: nowrap; min-width: 110px; width: 100%; user-select: none; transition: border-color .15s; font-family: 'DM Sans', sans-serif; height: 40px; }
.opd-dd-trigger.opd-dd-open { border-color: var(--opd-accent); border-radius: 6px 6px 0 0; }
.opd-dd-trigger span:not(.opd-dd-arrow) { flex: 1; }
.opd-dd-arrow { display: inline-flex; align-items: center; margin-left: 10px; flex-shrink: 0; font-size: 10px; color: #adb5bd; transition: transform .15s; }
.opd-dd-trigger.opd-dd-open .opd-dd-arrow { transform: rotate(180deg); }
.opd-dd-menu { position: absolute; top: 100%; left: 0; right: 0; background: #fff; border-top: none; border-radius: 0 0 6px 6px; z-index: 500; overflow: hidden; box-shadow: 0 4px 4px rgb(0 0 0 / 21%); }
.opd-dd-item { display: flex; align-items: center; gap: 7px; padding: 7px 10px; font-size: 12px; cursor: pointer; color: #495057; transition: background .1s; }
.opd-dd-item:hover { background: #f1f5ff; }
.opd-dd-item.opd-dd-selected { color: var(--opd-accent); background: #f1f5f9; font-weight: 500; }

/* ── Table card ── */
.opd-table-card { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: hidden; max-width: 100%; }
.opd-table-outer { display: flex; overflow: hidden; max-width: 100%; }
.opd-frozen-panel { flex-shrink: 0; border-right: 1px solid var(--opd-border); background: #f8fafc; z-index: 3;     width: 310px;}
.opd-frozen-panel table { border-collapse: collapse; width: 100%; }
.opd-frozen-panel tbody .opd-td { background: #F8FAFC; }
.opd-scroll-panel { flex: 1; overflow-x: auto; overflow-y: hidden; min-width: 0; }
.opd-scroll-panel table { border-collapse: collapse; min-width: 900px; }

.opd-th { padding: 12px 14px; text-align: left; font-weight: 600; font-size: 14px; color: #212529; white-space: nowrap; background: #f8fafc; border-bottom: 1px solid var(--opd-border); }
.opd-td { padding: 12px 14px; height: 66px; vertical-align: middle; font-size: 14px; border-bottom: 1px solid #CBD5E1; }
.opd-row-even { background: #fff; }
.opd-row-odd  { background: #fff; }
.opd-row-even:hover, .opd-row-odd:hover { background: #f8fafc; }
.opd-no-results { padding: 28px 16px; text-align: center; color: var(--opd-muted); font-size: 13px; }
.opd-th-sortable { cursor: pointer; user-select: none; }
.opd-th-sortable:hover { background: #eef2f7; }
.opd-th-label-wrap { display: inline-flex; align-items: center; }
.opd-th-sortable .opd-dd-arrow { margin-left: 6px; }
.opd-th-sortable-active .opd-dd-arrow { color: #1a56db; }


/* name / deal */
.opd-deal-name { font-weight: 600; color: #0f172a; cursor: pointer; white-space: nowrap; }
.opd-deal-co { font-size: 11px; color: var(--opd-text3); margin-top: 2px; }

/* ── Table footer ── */
.opd-table-footer { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-top: 1px solid var(--opd-border); background: #f8fafc; flex-wrap: wrap; gap: 8px; }
.opd-pag-row { display: flex; align-items: center; gap: 3px; }
.opd-pag-btn { width: 26px; height: 26px; border-radius: 5px; border: 1px solid var(--opd-border); background: #fff; color: #495057; font-size: 11px; cursor: pointer; font-family: 'DM Sans', sans-serif; transition: background .1s; }
.opd-pag-btn.opd-pag-active { background: var(--opd-accent); color: #fff; border-color: var(--opd-accent); font-weight: 600; }
.opd-pag-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* status badge */
.opd-status-badge { padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 500; display: inline-block; text-align: center; }
.opd-status-badge.opd-status-active { background: #22C55E; color: #fff; padding: 4px 10px; width:70px; text-align: center; }
.opd-status-badge.opd-status-at-risk { background: #EF4444; color: #fff; padding: 4px 10px; width:70px; text-align: center; }
.opd-status-badge.opd-status-pending, .opd-status-badge.opd-status-warning { background: #F97316; color: #fff; padding: 4px 10px; width:70px; text-align: center; }

/* trend */
.opd-trend-up   { color: #16a34a; font-weight: 500; }
.opd-trend-down { color: #dc2626; font-weight: 500; }

.opp-header { display: flex; align-items: center; justify-content: space-between; margin-top: 14px; }

.opd-action-btn { border: none; background: none; cursor: pointer; color: var(--opd-text3); padding: 4px; border-radius: 4px; }
.opd-action-btn:hover { background: #f1f5f9; color: var(--opd-text); }

@media(max-width:768px) {
  .opd-page { padding: 12px; }
}
`;

/* ─────────────────────────────────────────────
   CONSTANTS / DATA
───────────────────────────────────────────── */
const SUMMARY_PLACEHOLDER = [
  {
    label: "Total Accounts",
    bucket: "open",
    value: "—",
    count: "—",
    delta: "",
    up: true,
  },
  {
    label: "Account Value(ACV)",
    bucket: "pipeline",
    value: "—",
    count: "—",
    delta: "",
    up: true,
  },
  {
    label: "Active Accounts",
    bucket: "best_case",
    value: "—",
    count: "—",
    delta: "",
    up: false,
  },
  {
    label: "Accounts at Risk",
    bucket: "commit",
    value: "—",
    count: "—",
    delta: "",
    up: false,
  },
];

const DD_PLACEHOLDER = {
  Regions: [{ v: "all", l: "Regions" }],
  Industries: [{ v: "all", l: "Industries" }],
  Segment: [{ v: "all", l: "Segment" }],
  Status: [{ v: "all", l: "Status" }],
};

const DD_COLORS = [
  "#1a56db",
  "#0e9f6e",
  "#e3a008",
  "#7c3aed",
  "#e74c3c",
  "#3b5bdb",
  "#6c757d",
  "#0ea5e9",
];
const colorAt = (i) => DD_COLORS[i % DD_COLORS.length];
const INDUSTRY_COLORS = ["#EFF6FF", "#FFF7ED", "#ECFDF5"];
const getIndustryColor = (industry) => {
  if (!industry || industry === "—") return INDUSTRY_COLORS[0];
  let hash = 0;
  for (let i = 0; i < industry.length; i++) {
    hash = industry.charCodeAt(i) + ((hash << 5) - hash);
  }
  return INDUSTRY_COLORS[Math.abs(hash) % INDUSTRY_COLORS.length];
};


function mapKpiSummary(payload) {
  if (!payload) return SUMMARY_PLACEHOLDER;
  console.log("payload", payload);
  return [
    {
      label: "Total Accounts",
      bucket: "totalAccounts",
      value: payload.totalAccounts?.count ?? 0,
      count: payload.totalAccounts?.count ?? 0,
      delta: payload.totalAccounts?.trend?.deltaCount ?? 0,
      up: payload.totalAccounts?.trend?.direction === "up",
    },
    {
      label: "Account Value(ACV)",
      bucket: "accountValue",
      value: formatCurrencyShort(
        payload.accountValue?.value ?? 0,
        payload.currency,
      ),
      count: payload.accountValue?.count ?? 0,
      delta: formatDelta(
        payload.accountValue?.trend?.deltaValue ?? 0,
        payload.currency,
      ),
      up: payload.accountValue?.trend?.direction === "up",
    },
    {
      label: "Active Accounts",
      bucket: "activeAccounts",
      value: payload.activeAccounts?.count ?? 0,
      count: payload.activeAccounts?.count ?? 0,
      delta: payload.activeAccounts?.trend?.deltaCount ?? 0,
      up: payload.activeAccounts?.trend?.direction === "up",
    },
    {
      label: "Accounts at Risk",
      bucket: "accountsAtRisk",
      value: payload.accountsAtRisk?.count ?? 0,
      count: payload.accountsAtRisk?.count ?? 0,
      delta: payload.accountsAtRisk?.trend?.deltaCount ?? 0,
      up: payload.accountsAtRisk?.trend?.direction === "up",
    },
  ];
}

/* ─────────────────────────────────────────────
   CUSTOM DROPDOWN
───────────────────────────────────────────── */
function Dropdown({ filterKey, value, onChange, options }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const opts = options || [];
  const current = opts.find((o) => o.v === value) ||
    opts[0] || { v: "all", l: filterKey };

  useEffect(() => {
    const h = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div className="opd-dd-wrap" ref={ref}>
      <div
        className={`opd-dd-trigger${open ? " opd-dd-open" : ""}`}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{current.l}</span>
  <span className="opd-dd-arrow">
          <svg viewBox="0 0 10 10" width="10" height="10">
            <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round" />
          </svg>
        </span>

      </div>
      {open && (
        <div className="opd-dd-menu">
          {opts.map((opt) => (
            <div
              key={opt.v}
              className={`opd-dd-item${value === opt.v ? " opd-dd-selected" : ""}`}
              onClick={() => {
                onChange(filterKey, opt.v);
                setOpen(false);
              }}
            >
              {opt.l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────
   MAIN PAGE
───────────────────────────────────────────── */
const PAGE_SIZE = 60;
const PERIOD_TO_API = {
  week: "last_week",
  month: "past_month",
  quarter: "last_quarter",
};

export default function AccountsDashboard() {
  const navigate = useNavigate();
  const [periodTab, setPeriodTab] = useState("week");
  const [search, setSearch] = useState("");
  const [sortConfig, setSortConfig] = useState({
    key: "totalAccountValueRaw",
    direction: "desc",
  });
  const [filters, setFilters] = useState({
    Regions: "all",
    Industries: "all",
    Stage: "all",
    Segments: "all",
    AccountTypes: "all",
  });
  const [activeBucket, setActiveBucket] = useState(null);
  const [ddOptions, setDdOptions] = useState(DD_PLACEHOLDER);
  const [page, setPage] = useState(1);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [kpis, setKpis] = useState(SUMMARY_PLACEHOLDER);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRows, setSelectedRows] = useState({});
  const [selectAll, setSelectAll] = useState(false);

  // inject styles once
  useEffect(() => {
    if (document.getElementById("opd-styles-accounts")) return;
    const s = document.createElement("style");
    s.id = "opd-styles-accounts";
    s.textContent = STYLES;
    document.head.appendChild(s);
  }, []);

  // fetch dropdown options once on mount
  // useEffect(() => {
  //   const ctrl = new AbortController();
  //   fetchFilterOptions(ctrl.signal)
  //     .then(({ regions, industries, stages, products }) => {
  //       setDdOptions({
  //         Regions: [
  //           { v: "all", l: "Regions" },
  //           ...(regions?.businessGroups || []).map((b, i) => ({
  //             v: b.id,
  //             l: b.label,
  //             c: colorAt(i),
  //           })),
  //         ],
  //         Industries: [
  //           { v: "all", l: "Industries" },
  //           ...(industries?.items || []).map((b, i) => ({
  //             v: b.code,
  //             l: b.label,
  //             c: colorAt(i + 3),
  //           })),
  //         ],
  //         Segment: [
  //           { v: "all", l: "Segment" },
  //           { v: "strategic", l: "Strategic" },
  //           { v: "growth", l: "Growth" },
  //           { v: "core", l: "Core" },
  //         ],
  //         Status: [
  //           { v: "all", l: "Status" },
  //           { v: "active", l: "Active" },
  //           { v: "at-risk", l: "At Risk" },
  //         ],
  //       });
  //     })
  //     .catch((err) => {
  //       if (err.name !== "AbortError")
  //         console.error("filters load failed", err);
  //     });
  //   return () => ctrl.abort();
  // }, []);


  useEffect(() => {
    const ctrl = new AbortController();

    fetchAccountFilters(ctrl.signal)
      .then((data) => {
        setDdOptions({
          Regions: [
            { v: "all", l: "Regions" },
            ...(data.regions || []).map((r) => ({ v: r, l: r })),
          ],
          Industries: [
            { v: "all", l: "Industries" },
            ...(data.industries || []).map((i) => ({ v: i, l: i })),
          ],
          Stage: [
            { v: "all", l: "Stage" },
            ...(data.accountStatuses || []).map((s) => ({ v: s, l: s })),
          ],
          Segments: [
            { v: "all", l: "Segments" },
            ...(data.segments || []).map((s) => ({ v: s, l: s })),
          ],
          AccountTypes: [
            { v: "all", l: "Account Type" },
            ...(data.accountTypes || []).map((a) => ({ v: a, l: a })),
          ],
        });
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          console.error("filters load failed", err);
        }
      });

    return () => ctrl.abort();
  }, []);

  const filterSignature = JSON.stringify({
    search,
    filters,
    periodTab,
    activeBucket,
  });
  useEffect(() => {
    setPage(1);
  }, [filterSignature]);

  // fetch data
  useEffect(() => {
    const ctrl = new AbortController();

    const handle = setTimeout(() => {
      setLoading(true);
      setError(null);

      // ✅ build params correctly
      const params = {};
      if (search.trim()) params.search = search.trim();
      if (filters.Regions !== "all") params.regions = filters.Regions;
      if (filters.Industries !== "all") params.industries = filters.Industries;
      if (filters.Stage !== "all") params.accountStatuses = filters.Stage;
      if (filters.Segments !== "all") params.segments = filters.Segments;
      if (filters.AccountTypes !== "all") params.accountTypes = filters.AccountTypes;

      const hasFilters = Object.keys(params).length > 0;

      const BUCKET_MAP = {
        totalAccounts: "total",
        accountValue: "value",
        activeAccounts: "active",
        accountsAtRisk: "at_risk",
      };

      // ✅ switch API dynamically
      const apiCall = hasFilters
        ? fetchFilteredAccounts(params, ctrl.signal)
        : fetchAccounts(
          {
            page,
            pageSize: PAGE_SIZE,
            sortBy: "totalAccountValue",
            sortOrder: "desc",
            bucket: activeBucket
              ? BUCKET_MAP[activeBucket]
              : undefined,
          },
          ctrl.signal
        );

      Promise.all([
        apiCall,
        fetchAccountKpiSummary(
          { ...params, comparePeriod: PERIOD_TO_API[periodTab] },
          ctrl.signal
        ),
      ])
        .then(([opps, kpi]) => {
          const ownerId = sessionStorage.getItem("ownerId");
          const filtered = ownerId
            ? (opps.items || []).filter(
                (item) => item.ownerId?.toLowerCase() === ownerId.toLowerCase(),
              )
            : (opps.items || []);
          const mapped = filtered.map((item) => ({
            id: item.id,
            name: item.accountName || item.name || "—",
            seller: item.ownerName || "—",
            accountType: item.accountType || "Enterprise",
            industry: item.industry || "—",
            segment: item.segment || "Strategic",
            region: item.region || "—",
            status: item.status || "Active",
            lastInteraction: formatDateMMDDYY(item.lastInteraction),
            lastInteractionRaw: item.lastInteraction || null,
            opps: item.oppsCount || 0,
            totalAccountValueRaw: Number(item.totalAccountValue) || 0,
            totalAccountValue: formatCurrencyShort(item.totalAccountValue),
          }));

          setItems(mapped);
          setTotal(mapped.length);
          setTotalPages(Math.ceil(mapped.length / PAGE_SIZE) || 1);
          setKpis(mapKpiSummary(kpi));
        })
        .catch((err) => {
          if (err.name === "AbortError") return;
          console.error(err);
          setError(err.message || "Failed to load data");
          setItems([]);
          setKpis(SUMMARY_PLACEHOLDER);
        })
        .finally(() => setLoading(false));
    }, 250);

    return () => {
      clearTimeout(handle);
      ctrl.abort();
    };
  }, [
    search,
    filters.Regions,
    filters.Industries,
    filters.Stage,
    filters.Segments,
    filters.AccountTypes,
    periodTab,
    page,
    activeBucket,
  ]);


  const handleBucketToggle = useCallback((bucket) => {
    if (!bucket) return;
    if (bucket === "accountValue") return;
    setActiveBucket((prev) => (prev === bucket ? null : bucket));
  }, []);

  const handleFilterChange = useCallback((key, val) => {
    setFilters((prev) => ({ ...prev, [key]: val }));
  }, []);

  const handleSelectAll = useCallback(() => {
    const newSelectAll = !selectAll;
    setSelectAll(newSelectAll);
    const newSelected = {};
    items.forEach((item) => {
      newSelected[item.id] = newSelectAll;
    });
    setSelectedRows(newSelected);
  }, [selectAll, items]);

  const handleSelectRow = useCallback(
    (id) => {
      const newSelected = { ...selectedRows, [id]: !selectedRows[id] };
      setSelectedRows(newSelected);
      setSelectAll(items.every((item) => newSelected[item.id]));
    },
    [selectedRows, items],
  );

  const handleDownload = useCallback(() => {
    const headers = [
      "Name",
      "Seller",
      "Account Type",
      "Industry",
      "Segment",
      "Region",
      "Status",
      "Last Interaction",
      "Account KPI",
      "Daily KPI Trend",
      "Opps",
      "TAV",
    ];
    const rows = items.map((d) => [
      d.name,
      d.seller,
      d.accountType,
      d.industry,
      d.segment,
      d.region,
      d.status,
      d.lastInteraction,
      d.accountKpi,
      d.dailyKpiTrend,
      d.opps,
      d.tav,
    ]);
    const escapeCsv = (value) => {
      const text = String(value ?? "");
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    const csv = [headers, ...rows]
      .map((row) => row.map(escapeCsv).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `accounts-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }, [items]);

  const pageButtons = useMemo(
    () => buildPageList(totalPages, page, 3),
    [totalPages, page],
  );

  const sortedItems = useMemo(() => {
    const list = [...items];
    const { key, direction } = sortConfig;

    list.sort((a, b) => {
      const aVal = a?.[key];
      const bVal = b?.[key];

      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;

      if (key === "lastInteractionRaw") {
        const aDate = new Date(aVal).getTime();
        const bDate = new Date(bVal).getTime();
        const diff = aDate - bDate;
        return direction === "asc" ? diff : -diff;
      }

      if (typeof aVal === "number" && typeof bVal === "number") {
        const diff = aVal - bVal;
        return direction === "asc" ? diff : -diff;
      }

      const cmp = String(aVal).localeCompare(String(bVal));
      return direction === "asc" ? cmp : -cmp;
    });

    return list;
  }, [items, sortConfig]);

  const handleSort = useCallback((key) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return {
          key,
          direction: prev.direction === "asc" ? "desc" : "asc",
        };
      }
      return { key, direction: "asc" };
    });
  }, []);

  useEffect(() => {
    const id = "opd-global-styles";
    if (!document.getElementById(id)) {
      const style = document.createElement("style");
      style.id = id;
      style.innerHTML = STYLES;
      document.head.appendChild(style);
    }
  }, []);
  const renderSortableHeader = (label, key, minWidth) => {
    const isActive = sortConfig.key === key;
     const rotation = isActive && sortConfig.direction === "asc" ? "rotate(180deg)" : "rotate(0deg)";

     

    return (
      <th
        className={`opd-th opd-th-sortable${isActive ? " opd-th-sortable-active" : ""}`}
        style={{ minWidth }}
        onClick={() => handleSort(key)}
      >
        <span className="opd-th-label-wrap">
          <span>{label}</span>
          <span className="opd-dd-arrow" style={{ transform: rotation }}>
            <svg viewBox="0 0 10 10" width="10" height="10">
              <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            </svg>

          </span>
        </span>
      </th>
    );
  };

  return (
    <div className="opd-root">
      <div className="opd-page">
        <div className="opp-header">
          <h2 className="opd-h1">Accounts</h2>
          <div className="opd-period-tabs">
            {[
              ["week", "vs. Last Week"],
              ["month", "vs. Past Month"],
              ["quarter", "vs. Last Quarter"],
            ].map(([id, lbl]) => (
              <button
                key={id}
                className={`opd-period-tab${periodTab === id ? " opd-period-active" : ""}`}
                onClick={() => setPeriodTab(id)}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>

        {/* summary cards */}
        <div className="opd-summary-row">
          {kpis.map((c, i) => {
            const isActive = c.bucket && c.bucket === activeBucket;
            const tipText = isActive
              ? `Showing ${c.label} only — click to clear`
              : `Filter grid to ${c.label}`;

            return (
              <div key={c.bucket || i} className="opd-summary-col">
                <div
                  className={`opd-scard w-100${isActive ? " opd-scard-active" : ""}`}
                  role="button"
                  tabIndex={0}
                      data-tip={tipText}
                  onClick={() => handleBucketToggle(c.bucket)}
                >
                  <div className="opd-scard-head">
                    <div className="opd-scard-label">{c.label}</div>
                  </div>
                  <div className="opd-scard-body">
                    <div className="opd-scard-value">{c.value}</div>
                    <div className="opd-scard-meta">
                      <span className="opd-scard-count">#Accounts</span>
                      {c.delta && (
                        <>
                          <span className="opd-scard-sep">|</span>
                          <span
                            className={c.up ? "opd-delta-up" : "opd-delta-down"}
                          >
                            <img
                              src={c.up ? greenArrowIcon : redArrowIcon}
                              alt=""
                              className="opd-delta-icon"
                            />
                            {c.delta}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* filter bar */}
        <div className="opd-filter-bar">
          <div className="opd-search-box">
            <svg
              className="opd-search-icon"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
            >
              <circle cx="11" cy="11" r="7" />
              <line x1="16.5" y1="16.5" x2="21" y2="21" />
            </svg>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search Accounts"
            />
          </div>
          {["Regions", "Industries", "Stage", "Segments", "AccountTypes"].map((key) => (
            <Dropdown
              key={key}
              filterKey={key}
              value={filters[key]}
              options={ddOptions[key]}
              onChange={handleFilterChange}
            />
          ))}
         <button className="opd-btn-dl" onClick={handleDownload}>
            <img alt="Download" src={downloadIcon} style={{ height: 12 }} />
              </button>
        </div>

        {/* table */}
        <div className="opd-table-card">
          <div className="opd-table-outer">
            {/* frozen left */}
            <div className="opd-frozen-panel">
              <table>
                <colgroup>
                  <col style={{ width: 228 }} />
                </colgroup>
                <thead style={{ height: 42.3 }}>
                  <tr>
                    <th className="opd-th">Name</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && items.length === 0 ? (
                    <tr>
                      <td colSpan={2} className="opd-no-results">
                        Loading…
                      </td>
                    </tr>
                  ) : items.length === 0 ? (
                    <tr>
                      <td colSpan={2} className="opd-no-results">
                        No deals match
                      </td>
                    </tr>
                  ) : (
                    sortedItems.map((d, i) => (
                      <tr
                        key={d.id || i}
                        className={i % 2 === 0 ? "opd-row-even" : "opd-row-odd"}
                      >
                        {/* <td className="opd-td">
                          <div
                            className="opd-name-cell"
                            style={{ cursor: "pointer" }}
                            onClick={() => navigate(`/accounts/${d.id}`)}
                          >
                            {d.name}
                          </div>
                          <div className="opd-deal-co">{d.id}</div>
                        </td> */}

                        <td className="opd-td">
                          <div
                            className="opd-deal-name"
                            onClick={() => navigate(`/accounts/${d.id}`)}
                          >
                            {d.name}
                          </div>
                          <div className="opd-deal-co">{d.id}</div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* scrollable right */}
            <div className="opd-scroll-panel">
              <table>
                <thead>
                  <tr>
                    {renderSortableHeader("Seller", "seller", 140)}
                    {renderSortableHeader("Account Type", "accountType", 140)}
                    {renderSortableHeader("Industry", "industry", 160)}
                    {renderSortableHeader("Segment", "segment", 140)}
                    {renderSortableHeader("Region", "region", 140)}
                    {renderSortableHeader("Status", "status", 120)}
                    {renderSortableHeader("Last Interaction", "lastInteractionRaw", 170)}
                    {renderSortableHeader("Opps", "opps", 100)}
                    {renderSortableHeader("Account Value", "totalAccountValueRaw", 140)}

                    <th
                      className="opd-th opd-action-col"
                      style={{ minWidth: 60 }}
                    ></th>
                  </tr>
                </thead>
                <tbody>
                  {loading && items.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="opd-no-results">
                        Loading opportunities…
                      </td>
                    </tr>
                  ) : error ? (
                    <tr>
                      <td
                        colSpan={8}
                        className="opd-no-results"
                        style={{ color: "#ef4444" }}
                      >
                        {error}
                      </td>
                    </tr>
                  ) : items.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="opd-no-results">
                        No deals match your filters
                      </td>
                    </tr>
                  ) : (
                    sortedItems.map((d, i) => (
                      <tr
                        key={d.id || i}
                        className={i % 2 === 0 ? "opd-row-even" : "opd-row-odd"}
                      >
                        <td className="opd-td">{d.seller}</td>
                        <td className="opd-td">{d.accountType}</td>
                        <td className="opd-td">
                          <span
                            style={{
                              background: getIndustryColor(d.industry),
                              padding: "5px 14px",
                              borderRadius: "5px",
                              fontSize: "12px",
                              fontWeight: 600,
                              color: "#495057",
                              whiteSpace: "nowrap",
                              display: "inline-block",
                              width: "180px",
                              textAlign: "center",
                            }}
                          >
                            {d.industry}
                          </span>
                        </td>

                        <td className="opd-td">{d.segment}</td>
                        <td className="opd-td">{d.region}</td>
                        <td className="opd-td">
                          <span
                            className={`opd-status-badge opd-status-${d.status.toLowerCase().replace(" ", "-")}`}
                          >
                            {d.status}
                          </span>
                        </td>
                        <td className="opd-td">{d.lastInteraction}</td>
                        <td className="opd-td">{d.opps}</td>
                        <td className="opd-td" style={{ fontWeight: 600 }}>
                          {d.totalAccountValue}
                        </td>
                        <td className="opd-td opd-action-col">
                          <button
                            className="opd-action-btn"
                            aria-label="More options"
                          >
                            <svg
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="currentColor"
                            >
                              <circle cx="12" cy="5" r="2" />
                              <circle cx="12" cy="12" r="2" />
                              <circle cx="12" cy="19" r="2" />
                            </svg>
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* table footer */}
          <div className="opd-table-footer">
            <span style={{ fontSize: 11, color: "var(--opd-text3)" }}>
              {total === 0
                ? "Showing 0 accounts"
                : `Showing ${items.length} of ${total} accounts`}
            </span>
            <div className="opd-pag-row">
              <span
                style={{
                  fontSize: 14,
                  color: "var(--opd-text3)",
                  marginRight: 4,
                }}
              >
                Show
              </span>
                <button
                className="opd-pag-btn"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                <i className="bi bi-chevron-left" style={{ fontSize: 10 }}></i>
              </button>
              {pageButtons.map((p, i) =>
                p === "..." ? (
                  <span
                    key={`gap-${i}`}
                    style={{
                      fontSize: 14,
                      color: "var(--opd-text3)",
                      padding: "0 3px",
                    }}
                  >
                    …
                  </span>
                ) : (
                  <button
                    key={p}
                    className={`opd-pag-btn${p === page ? " opd-pag-active" : ""}`}
                    onClick={() => setPage(p)}
                  >
                    {p}
                  </button>
                ),
              )}
              <button
                className="opd-pag-btn"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                <i className="bi bi-chevron-right" style={{ fontSize: 10 }}></i>
                
              </button>
              <span
                style={{
                  fontSize: 11,
                  color: "var(--opd-text3)",
                  marginLeft: 4,
                }}
              >
                {totalPages} {totalPages === 1 ? "page" : "pages"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}