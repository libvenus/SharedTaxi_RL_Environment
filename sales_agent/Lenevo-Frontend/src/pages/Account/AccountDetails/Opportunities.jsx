import { useState, useRef, useCallback, useEffect, useMemo } from "react";

import {
  fetchKpiSummary,
  fetchAccountOpportunities,
  fetchAccountFilters,
  fetchFilteredAccounts,
} from "../../../api/client";
import {
  buildPageList,
  formatCurrencyShort,
  formatDateMMDDYY,
  formatDealCount,
  formatDelta,
} from "../../../utils/format";
import { useParams } from "react-router-dom";
import leftArrowIcon from "../../../assets/icons/left.png";
import rightArrowIcon from "../../../assets/icons/right1.png";
import downloadIcon from "../../../assets/download.png";
/* ─────────────────────────────────────────────
   GLOBAL STYLES  (injected once into <head>)
───────────────────────────────────────────── */
const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --opd-crm:        #1dcb8a;
  --opd-email:      #1e3a5f;
  --opd-meeting:    #5bb8f5;
  --opd-multiple:   #1e5fa8;
  --opd-h-high:     #22c55e;
  --opd-h-mid:      #f59e0b;
  --opd-h-low:      #ef4444;
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
.opd-root { background: var(--opd-bg); height: 100vh; overflow: hidden; color: var(--opd-text); font-size: 13px; padding: 20px; width: 100%; max-width: 100%; }

/* ── Page ── */
.opd-page { padding: 44px 0px; height: 100%; overflow-y: auto; overflow-x: hidden; max-width: 100%; }
.opd-breadcrumb { font-size: 12px; color: var(--opd-text3); margin-bottom: 10px; }
.opd-breadcrumb strong { color: var(--opd-text); }
.opd-breadcrumb-closure { color: #e74c3c; font-weight: 600; }
.opd-h1 { font-size: 28px; font-weight: 700; margin-bottom: 16px; }

/* ── Period tabs ── */
.opd-period-tabs {height:40px; display: flex; border: 1px solid var(--opd-border); border-radius: 8px; overflow: hidden; width: fit-content; background: #fff; margin-bottom: 16px; }
.opd-period-tab { padding: 7px 16px; border: none; border-right: 1px solid var(--opd-border); background: transparent; font-family: 'DM Sans', sans-serif; font-size: 14px; color: var(--opd-text3); cursor: pointer; transition: background .15s, color .15s; white-space: nowrap; }
.opd-period-tab:last-child { border-right: none; }
.opd-period-tab.opd-period-active { background: #e8f0fe; color: var(--opd-accent); font-weight: 600;height:40px; }

/* ── Summary cards ── */
.opd-summary-grid { display: grid; grid-template-columns: repeat(7,1fr); gap: 10px; margin-top: 6px; margin-bottom: 20px !important; overflow: visible; max-width: 100%; }

.opd-scard { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: visible; text-align: center; display: flex; flex-direction: column; cursor: pointer; transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; user-select: none; min-height: 96px; position: relative; }
.opd-scard:hover,
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
  font-size: 12px;
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
.opd-summary-grid { display: grid; grid-template-columns: repeat(7,1fr); gap: 10px;     margin-top: 6px;
    margin-bottom: 20px !important;}
@media(max-width:992px){ .opd-summary-grid { grid-template-columns: repeat(3,1fr); } }
@media(max-width:576px){ .opd-summary-grid { grid-template-columns: repeat(2,1fr); } }
.opd-scard { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: hidden; text-align: center; display: flex; flex-direction: column; cursor: pointer; transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; user-select: none; min-height: 96px; }
.opd-scard:hover { box-shadow: 0 2px 10px rgba(0,0,0,.08); border-color: #c7d2e3; }
.opd-scard:focus-visible { outline: 2px solid #1a56db; outline-offset: 2px; }
.opd-scard-active { border-color: #1a56db; box-shadow: 0 0 0 2px rgba(26,86,219,.18); }
.opd-scard-active .opd-scard-head { background: #eff4ff; }
.opd-scard-active .opd-scard-label { color: #1a56db; }
.opd-scard-head { padding: 6px 10px; border-bottom: 1px solid var(--opd-border); background: #fff; border-radius: 12px 12px 0px 0px;}
.opd-scard-label { font-size: 14px; color: #334155; margin: 0; }
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
.opd-search-box input { border: none; outline: none; font-size: 14px; color: #495057; background: transparent; width: 100%; font-family: 'DM Sans', sans-serif; }
.opd-search-icon { color:#475569; flex-shrink: 0; }
.opd-btn-clear { padding: 5px 12px; border-radius: 6px; border: 1px solid var(--opd-border); background: #fff; color: var(--opd-text3); font-size: 12px; cursor: pointer; font-family: 'DM Sans', sans-serif; white-space: nowrap; }
.opd-btn-clear:hover { background: #f8f9fa; }
.opd-btn-dl { width: 40px; height: 40px; border-radius: 50px; border:none; background: #E2E8F0; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--opd-text3); font-size: 16px; flex-shrink: 0; }

/* ── Custom dropdown ── */
.opd-dd-wrap { position: relative; flex: 1 1 140px; min-width: 120px; }
.opd-dd-trigger { display: flex; align-items: center; border: 2px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; font-size: 12px; color: #495057; background: #f8fafc; cursor: pointer; white-space: nowrap; min-width: 110px; width: 100%; user-select: none; transition: border-color .15s; font-family: 'DM Sans', sans-serif;height:40px }
.opd-dd-trigger.opd-dd-open { border-color: var(--opd-accent); border-radius: 6px 6px 0 0; }
.opd-dd-trigger span:not(.opd-dd-arrow) { flex: 1; }
.opd-dd-arrow {      width: 10px; display: inline-flex; align-items: center; margin-left: 10px; flex-shrink: 0; font-size: 10px; color: #adb5bd; transition: transform .15s; }
.opd-dd-trigger.opd-dd-open .opd-dd-arrow { transform: rotate(180deg); }
.opd-dd-menu { position: absolute; top: 100%; left: 0; right: 0; background: #fff;  border-top: none; border-radius: 0 0 6px 6px; z-index: 500; overflow: hidden; box-shadow: 0 4px 4px rgb(0 0 0 / 21%); }
.opd-dd-item { display: flex; align-items: center; gap: 7px; padding: 7px 10px; font-size: 12px; cursor: pointer; color: #495057; transition: background .1s; }
.opd-dd-item:hover { background: #f1f5ff; }
.opd-dd-item.opd-dd-selected { color: var(--opd-accent); background: #f1f5f9; ; font-weight: 500; }
.opd-dd-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

.opd-btn-clear,
.opd-btn-dl { flex-shrink: 0; }

/* ── Table card ── */
.opd-table-card { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: hidden; max-width: 100%; width: 100%; }
.opd-table-outer { display: flex; overflow: hidden; max-width: 100%; width: 100%; }
.opd-frozen-panel { flex-shrink: 0; border-right: 2px solid var(--opd-border); z-index: 2;     width: 310px;}
.opd-frozen-panel table { border-collapse: collapse; table-layout: fixed; width: 300px; }
.opd-scroll-panel { flex: 1; overflow-x: auto; overflow-y: hidden; min-width: 0; width: 0; }
.opd-scroll-panel table { border-collapse: collapse; min-width: 900px; }
.opd-th { padding: 10px 13px; text-align: left; font-weight: 600; font-size: 14px; color: #212529; white-space: nowrap; background: #f8fafc; border-bottom: 1px solid var(--opd-border); }
.opd-td { padding: 0 13px; height: 54px; vertical-align: middle; font-size: 14px; }
.opd-row-even { background: #fff; border-bottom: 1px solid var(--opd-border) }
.opd-row-odd  { background: #fff; border-bottom: 1px solid var(--opd-border) }
.opd-spacer-row td { height: 25px; padding: 0; border-bottom: 1px solid var(--opd-border); background: #f8fafc;}
.opd-spacer-cell { padding: 0 13px; }
.opd-tl-header-spacer { height: 24px; display: flex; align-items: center; background:white }
.opd-frozen-panel tbody .opd-td { background: #f8fafc; }
.opd-deal-name { color: #0f172a; font-size: 14px; line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.opd-deal-co   { font-size: 12px; color: #64748b; font-weight: 500; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.opd-risk-badge { background: #FFF7ED !important; color: #856404; font-size: 13px; padding: 3px 9px !important;  border-radius: 5px; font-weight: 600; white-space: nowrap; display: inline-flex; align-items: center; gap: 4px; cursor: help; }
.opd-risk-badge .bi { color: #c53030; font-size: 13px; }
.opd-no-risk { color: #adb5bd; }
.opd-stage-badge { background: #EFF6FF; color: #495057; padding: 3px 10px; border-radius: 5px; font-size: 12px; font-weight: 600; display: inline-block; width: 86px; text-align: center; white-space: nowrap; }
.opd-stage-Discovery    { background: #EFF6FF; color: #495057; }
.opd-stage-Qualification{ background: #fff7ed; color: #495057; }
.opd-stage-Proposal     { background: #EFF6FF; color: #495057; }
.opd-stage-Execute      { background: #EFF6FF; color: #495057; }
.opd-motion-badge { background: #EFF6FF; color: #495057; padding: 3px 10px; border-radius: 5px; font-size: 12px; font-weight: 600;white-space: nowrap; display: inline-block; width: 90px; text-align: center; }
.opd-motion-net-new { background: #EFF6FF; display: inline-block; width: 86px; text-align: center; }
.opd-motion-expansion { background: #fff7ed; display: inline-block; width: 86px; text-align: center; }
.opd-no-results { padding: 28px 16px; text-align: center; color: var(--opd-muted); font-size: 13px; }

/* ── Health ring ── */
.opd-ring-wrap { position: relative; width: 38px; height: 38px; flex-shrink: 0; }
.opd-ring-label { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 9.6px; font-weight: 400;}

/* ── Timeline cell ── */
.opd-tl-cell { position: relative; width: 600px; height: 28px; display: flex; align-items: center; }
.opd-tl-line { position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: #e9ecef; }
.opd-tl-ticks { position: relative; width: 600px; height: 16px;margin-left: 50px; }
.opd-tl-tick { position: absolute; transform: translateX(-50%); font-size: 10px; color: #334155; white-space: nowrap; }

/* ── Activity dot ── */
.opd-dot-anchor { position: absolute; transform: translateX(-50%); }
.opd-dot-btn { border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 9px; font-weight: 700; border: none; transition: transform .12s, box-shadow .12s; }
.opd-dot-btn:hover { transform: scale(1.4); box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
.opd-dot-sm { width: 10px; height: 10px; }
.opd-dot-lg { width: 20px; height: 20px; }

/* ── DOT POPUP  (email-card style) ── */
.opd-popup-portal { position: fixed; z-index: 9999; pointer-events: none; }
.opd-popup-card {
  width: 300px;
  background: #fff;
      border: 1px solid var(--opd-border);
  border-radius: 10px;
  box-shadow: 0 4px 4px rgb(0 0 0 / 13%);
  overflow: hidden;
  pointer-events: all;
  animation: opd-pop-in .15s ease;
}
@keyframes opd-pop-in { from { opacity:0; transform:translateY(6px) scale(.97); } to { opacity:1; transform:none; } }

.opd-popup-header { display: flex; align-items: center; gap: 8px; padding: 7px 12px 6px; border-bottom: 1px solid #f3f4f6; }
.opd-popup-header-title { font-size:12px; font-weight: 600; color: #000;  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }

/* mini right-spine timeline in popup */
.opd-ptl-body { overflow-y: auto; padding: 10px 12px 10px 8px; }
.opd-ptl-item { display: flex; align-items: stretch; }
.opd-ptl-content { flex: 1; padding-left: 10px; padding-bottom: 16px; }
.opd-ptl-item:last-child .opd-ptl-content { padding-bottom: 4px; }
.opd-ptl-type { font-size: 10px; font-weight: 600; color: var(--opd-accent); text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 1px; }
.opd-ptl-title-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 2px; }
.opd-ptl-title { font-size: 11.5px; font-weight: 600; color: var(--opd-text); line-height: 1.35; }
.opd-ptl-time { font-size: 10px; color: #000;  white-space: nowrap; flex-shrink: 0; }
.opd-ptl-body-text { font-size: 12px; color: #000; line-height: 1.5; }
.opd-ptl-node-col { flex-shrink: 0; width: 18px; display: flex; flex-direction: column; align-items: center;color: #000;  }
.opd-ptl-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
.opd-ptl-line { width: 1.5px; flex: 1; background: #e5e7eb; margin-top: 3px;min-height: 44px; }
.opd-ptl-item:last-child .opd-ptl-line { display: none; }

/* ── Table footer ── */
.opd-table-footer { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-top: 1px solid var(--opd-border); background: #f8fafc;; flex-wrap: wrap; gap: 8px; }
.opd-legend-row { display: flex; align-items: center; gap: 30px; flex-wrap: wrap; }
.opd-legend-item { display: flex; align-items: center; gap: 5px; font-size: 14px; color: #0F172A; }
.opd-legend-dot { border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; color: #fff; font-weight: 700; flex-shrink: 0; }
.opd-pag-row { display: flex; align-items: center; gap: 3px; }
.opd-pag-btn { width: 26px; height: 26px; border-radius: 5px; border: 1px solid var(--opd-border); background: #fff; color: #495057; font-size: 12px; cursor: pointer; font-family: 'DM Sans', sans-serif; transition: background .1s; }
.opd-pag-btn.opd-pag-active { background: var(--opd-accent); color: #fff; border-color: var(--opd-accent); font-weight: 600; }

/* ── Offcanvas ── */
.opd-oc-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.22); z-index: 8000; opacity: 0; pointer-events: none; transition: opacity .28s; }
.opd-oc-overlay.opd-oc-open { opacity: 1; pointer-events: all; }
.opd-oc-panel { position: fixed; top: 0; right: 0; height: 100%; width: 520px; max-width: 96vw; background: var(--opd-bg); z-index: 8100; display: flex; flex-direction: column; border-left: 1.5px solid var(--opd-border); border-top-left-radius: 18px; border-bottom-left-radius: 18px; overflow: hidden; transform: translateX(100%); transition: transform .28s cubic-bezier(.4,0,.2,1); }
.opd-oc-panel.opd-oc-open { transform: translateX(0); }
.opd-oc-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px 14px; border-bottom: 1px solid var(--opd-border); flex-shrink: 0; }
.opd-oc-title { font-size: 16px; font-weight: 700; color: var(--opd-text); }
.opd-oc-subtitle { font-size: 12px; color: var(--opd-muted); margin-left: 8px; }
.opd-oc-close { width: 28px; height: 28px; border-radius: 6px; border:none; background: #ffffff00; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px;  }

/* OC tabs — same pill design as period tabs */
.opd-oc-tabs { display: flex; width: 300px; border: 1px solid var(--opd-border); border-radius: 8px; overflow: hidden; margin: 12px auto; background: #fff; flex-shrink: 0; }
.opd-oc-tab { flex: 1; padding: 7px 10px; border: none; border-right: 1px solid var(--opd-border); background: transparent; font-family: 'DM Sans', sans-serif; font-size: 12px; color: var(--opd-text3); cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 5px; transition: background .15s, color .15s; white-space: nowrap; font-weight: 400; }
.opd-oc-tab:last-child { border-right: none; }
.opd-oc-tab.opd-oc-tab-active { background: #e8f0fe; color: var(--opd-accent); font-weight: 600; }

/* OC activity subtitles (left=Inbound, right=Outbound on the zigzag) */
.opd-sub-tabs { display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 16px; background: transparent; }
.opd-sub-title { padding: 6px 12px; font-family: 'DM Sans', sans-serif; font-size: 16px; color: #334155; font-weight: 600; line-height: 1; white-space: nowrap; }
.opd-sub-icon { width: 18px; height: 18px; object-fit: contain; vertical-align: middle; margin-right: 5px; }

.opd-oc-body { flex: 1; overflow-y: auto; padding: 16px 20px 28px; }
.opd-oc-pane { display: none; }
.opd-oc-pane.opd-oc-pane-active { display: block; }

/* zigzag timeline */
.opd-zz-wrap { position: relative; }
.opd-zz-spine { position: absolute; left: calc(50% - 0.5px); top: 0; bottom: 0; width: 1px; background: #e5e7eb; }
.opd-zz-item { display: flex; align-items: flex-start; margin-bottom: 20px; position: relative; z-index: 1; }
.opd-zz-left { flex-direction: row; }
.opd-zz-right { flex-direction: row-reverse; }
.opd-zz-card-wrap { flex: 1; max-width: calc(50% - 28px); }
.opd-zz-card { background: #fff; border: 1px solid var(--opd-border); border-radius: 12px; padding: 14px 15px; }
.opd-zz-card-hd { display: flex; align-items: center; gap: 7px; margin: -14px -15px 10px; padding: 10px 15px; background: #f8fafc; border-bottom: 1px solid var(--opd-border); border-radius: 12px 12px 0 0; }
.opd-zz-type-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.opd-zz-type-lbl { font-size: 16px; font-weight: 400; color: var(--opd-text2); }
.opd-zz-dir-lbl { margin-left: auto; display: flex; align-items: center; gap: 3px; font-size: 10px; color: var(--opd-muted); }
.opd-zz-date { font-size: 12px; color: var(--opd-muted); margin-bottom: 4px; }
.opd-zz-title { font-size: 16px; font-weight: 600; color: var(--opd-text); margin-bottom: 6px; line-height: 1.4; }
.opd-zz-body { font-size: 12px;  line-height: 1.65; }
.opd-zz-body p, .opd-ptl-body-text p { margin: 0 0 6px; }
.opd-zz-body p:last-child, .opd-ptl-body-text p:last-child { margin-bottom: 0; }
.opd-zz-body ul, .opd-zz-body ol, .opd-ptl-body-text ul, .opd-ptl-body-text ol { margin: 4px 0; padding-left: 18px; }
.opd-zz-node { width: 56px; flex-shrink: 0; display: flex; justify-content: center; padding-top: 4px; }
.opd-zz-node-outer { width: 22px; height: 22px; border-radius: 50%; background: #fff; border: 2px solid #d1d5db; display: flex; align-items: center; justify-content: center; }
.opd-zz-node-inner { width: 11px; height: 11px; border-radius: 50%; }

/* warnings */
.opd-warn-item { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
.opd-warn-head { display: flex; align-items: center; gap: 8px; margin: -14px -16px 10px; padding: 10px 16px; background: #f8fafc; border-bottom: 1px solid var(--opd-border); border-radius: 10px 10px 0 0; }
.opd-warn-icon { width: 32px; height: 32px; border-radius: 8px; background: #fff3cd; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.opd-warn-title { font-size: 13px; font-weight: 600; color: var(--opd-text); }
.opd-warn-badge { margin-left: auto; font-size: 10px; padding: 2px 8px; border-radius: 20px; font-weight: 600; white-space: nowrap; }
.opd-warn-high { background: #fee2e2; color: #991b1b; }
.opd-warn-med  { background: #fff3cd; color: #856404; }
.opd-warn-low  { background: #dcfce7; color: #166534; }
.opd-warn-desc { font-size: 12px; color: var(--opd-text3); line-height: 1.6; }

/* contacts */
.opd-contact-item { display: flex; align-items: center; gap: 12px; padding: 14px 0; border-bottom: 1px solid #f3f4f6; }
.opd-contact-item:last-child { border-bottom: none; }
.opd-contact-avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; color: #fff; flex-shrink: 0; }
.opd-contact-name { font-size: 13px; font-weight: 600; color: var(--opd-text); }
.opd-contact-role { font-size: 12px; color: var(--opd-muted); }
.opd-contact-email { font-size: 12px; color: var(--opd-accent); margin-top: 2px; }
.opd-contact-tag { margin-left: auto; font-size: 12px; padding: 3px 8px; border-radius: 20px; background: #e8f0fe; color: var(--opd-accent); font-weight: 500; white-space: nowrap; }

@media(max-width:768px) {
  .opd-page { padding: 12px; }
  .opd-tl-cell { width: 200px; }
  .opd-frozen-panel table { width: 220px; }
  .opd-oc-panel { width: 100vw; max-width: 100vw; }
}

.opp-header{ display: flex; align-items: center;justify-content: space-between;margin-top: 14px; }
`;

/* ─────────────────────────────────────────────
   CONSTANTS / DATA
───────────────────────────────────────────── */
const DC = {
  crm: "#1dcb8a",
  email: "#1e3a5f",
  meeting: "#5bb8f5",
  multiple: "#1e5fa8",
};
const DL = {
  crm: "CRM Updates",
  email: "Email",
  meeting: "Meeting",
  multiple: "Multiple Events",
};
const hColor = (p) => (p >= 70 ? "#22c55e" : p >= 40 ? "#f59e0b" : "#ef4444");

/** KPI strip falls back to these zero cards while the API call is in flight.
 *  `bucket` matches the `bucket` query-param on /api/opportunities — clicking a
 *  card sends that value to the list endpoint to filter the grid.
 */

/** KPI strip falls back to these zero cards while the API call is in flight.
 *  `bucket` matches the `bucket` query-param on /api/opportunities — clicking a
 *  card sends that value to the list endpoint to filter the grid.
 */
const SUMMARY_PLACEHOLDER = [
  {
    label: "Open Deals",
    bucket: "open",
    value: "—",
    count: "—",
    delta: "",
    up: true,
  },
  {
    label: "Pipeline",
    bucket: "pipeline",
    value: "—",
    count: "—",
    delta: "",
    up: true,
  },
  {
    label: "Best Case",
    bucket: "best_case",
    value: "—",
    count: "—",
    delta: "",
    up: false,
  },
  {
    label: "Commit",
    bucket: "commit",
    value: "—",
    count: "—",
    delta: "",
    up: false,
  },
  { label: "Won", bucket: "won", value: "—", count: "—", delta: "", up: true },
  {
    label: "Loss",
    bucket: "loss",
    value: "—",
    count: "—",
    delta: "",
    up: false,
  },
];

/** Dropdown placeholders shown until /api/filters/* responds. */
// const DD_PLACEHOLDER = {
//   Regions: [{ v: "all", l: "Regions" }],
//   Industries: [{ v: "all", l: "Industries" }],
//   Stage: [{ v: "all", l: "Stage" }],
//   Products: [{ v: "all", l: "Products" }],
// };

const DD_PLACEHOLDER = {
  Regions: [{ v: "all", l: "Regions" }],
  Industries: [{ v: "all", l: "Industries" }],
  Stage: [{ v: "all", l: "Stage" }],
  Products: [{ v: "all", l: "Products" }],
};

const PAGE_SIZE = 10;

const PERIOD_TO_API = {
  week: "last_week",
  month: "past_month",
  quarter: "last_quarter",
};

function splitDealAndCompany(rawName, accountName) {
  const original = (rawName || "").trim();
  if (!original) return { name: "", company: accountName || "" };

  const dashRe = /\s+[\u2013\u2014\-]\s+/;
  const dashMatch = original.match(dashRe);

  if (accountName) {
    const acct = accountName.trim();
    if (dashMatch) {
      const escaped = acct.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const exact = new RegExp(`^${escaped}\\s+[\\u2013\\u2014\\-]\\s+`, "i");
      if (exact.test(original)) {
        return { name: original.replace(exact, "").trim(), company: acct };
      }
    }
    return { name: original, company: acct };
  }

  if (dashMatch) {
    const idx = dashMatch.index;
    const left = original.slice(0, idx).trim();
    const right = original.slice(idx + dashMatch[0].length).trim();
    if (left && right) return { name: right, company: left };
  }

  return { name: original, company: "" };
}

/** Days between now and an ISO timestamp, clamped to the 0–90d timeline. */
function daysAgo(iso) {
  if (!iso) return 0;
  const ms = Date.now() - new Date(iso).getTime();
  const d = Math.round(ms / 86_400_000);
  return Math.max(0, Math.min(90, d));
}

function formatActivityWhen(iso) {
  if (!iso) return "";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return "";
  const date = dt.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
  const time = dt.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  return `${date} · ${time}`;
}

const stageClass = (s) =>
  `opd-stage-badge opd-stage-${(s || "").replace(/\s+/g, "")}`;

/** Map a row from the /api/opportunities response into the UI's shape. */
function mapOpportunityRow(item) {
  // Accept either camelCase (FastAPI default) or snake_case (in case
  // response_model_by_alias gets turned off somewhere).
  const account = item.accountName || item.account_name || "";
  const { name, company } = splitDealAndCompany(item.name, account);
  return {
    id: item.id,
    name,
    company,
    risk: item.risk ?? null,
    riskScore: item.riskScore ?? item.risk_score ?? null,
    region: item.region || "",
    industry: item.industry || "",
    stage: item.stage?.label || item.stage?.raw || "—",
    stageClass: item.stage?.label || item.stage?.raw || "Other",
    product: "",
    comp: (item.competitors || []).join(", ") || "—",
    value: formatCurrencyShort(item.value, item.currency),
    closeDate: formatDateMMDDYY(item.closeDate),
    motion: item.saleMotion?.label || item.saleMotion?.raw || "—",
    health: item.dealHealth ?? null,
    lastAct: formatDateMMDDYY(item.lastActivity),
    acts: (item.activities || []).map(mapActivity),
  };
}

/** Map /api/opportunities/kpi-summary into the 6-card structure the JSX expects. */
function mapKpiSummary(payload) {
  if (!payload) return SUMMARY_PLACEHOLDER;
  const cards = [
    ["Open Deals", "open", payload.openDeals],
    ["Pipeline", "pipeline", payload.pipeline],
    ["Best Case", "best_case", payload.bestCase],
    ["Commit", "commit", payload.commit],
    ["Won", "won", payload.won],
    ["Loss", "loss", payload.loss],
  ];
  return cards.map(([label, bucket, c]) => ({
    label,
    bucket,
    value: formatCurrencyShort(c?.value, payload.currency),
    count: formatDealCount(c?.count),
    delta: c?.trend ? formatDelta(c.trend.deltaValue, payload.currency) : "",
    up: c?.trend ? c.trend.direction !== "down" : true,
  }));
}

/** Translate an API ActivityItem into the {t,d,date,title,body} shape the UI expects. */
function mapActivity(a) {
  return {
    t: a.type,
    d: daysAgo(a.activityDate),
    date: formatActivityWhen(a.activityDate),
    title: a.subject || "",
    body: a.body || "",
    direction: a.direction || null,
    count: a.groupedCount || null,
  };
}

/* ─────────────────────────────────────────────
   SHARED: build padded activity list (min 3)
───────────────────────────────────────────── */
function buildActivityItems(deal) {
  const source = deal?.acts || [];
  const items = source.map((a, j) => ({
    type: a.t,
    date: a.date,
    title: a.title,
    body: a.body,
    direction: a.direction || (j % 2 === 0 ? "inbound" : "outbound"),
  }));
  while (items.length < 3) {
    const direction = items.length % 2 === 0 ? "inbound" : "outbound";
    items.push(
      direction === "inbound"
        ? {
            type: "crm",
            date: "—",
            title: "No further activity",
            body: "No additional inbound activity recorded.",
            direction,
          }
        : {
            type: "email",
            date: "—",
            title: "Awaiting response",
            body: "Follow-up email sent. Client response pending.",
            direction,
          },
    );
  }
  return items;
}

function sanitizeActivityHtml(html) {
  return String(html || "").replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, "");
}


/* ─────────────────────────────────────────────
   DOT POPUP  (portal-based so it's never clipped)
───────────────────────────────────────────── */
function DotPopup({ acts, deal, onOpenOC }) {
  const items = buildActivityItems(deal);

  return (
    <div
      className="opd-popup-card"
      onMouseDown={(e) => e.stopPropagation()}
      onClick={() => onOpenOC(deal)}
    >
      {/* start date row */}
      {items[0]?.date && items[0].date !== "—" && (
        <div className="opd-popup-header">
          <div className="opd-popup-header-title">
            {items[0].date.split(" · ")[0]}
          </div>
        </div>
      )}

      {/* left-spine mini timeline */}
      <div className="opd-ptl-body">
        {items.map((act, i) => (
          <div key={i} className="opd-ptl-item">
            {/* dot + spine on left */}
            <div className="opd-ptl-node-col">
              <div
                className="opd-ptl-dot"
                style={{ background: DC[act.type] }}
              />
              {i < items.length - 1 && <div className="opd-ptl-line" />}
            </div>
            {/* content on right */}
            <div className="opd-ptl-content">
              <div className="opd-ptl-title-row">
                <span className="opd-ptl-title">{act.title}</span>
                {act.date && act.date !== "—" && act.date.includes(" · ") && (
                  <span className="opd-ptl-time">
                    {act.date.split(" · ")[1]}
                  </span>
                )}
              </div>
              <div
                className="opd-ptl-body-text"
                dangerouslySetInnerHTML={{
                  __html: sanitizeActivityHtml(act.body),
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   ACTIVITY DOT + FLOATING POPUP
───────────────────────────────────────────── */
function ActivityDot({ act, acts, deal, onOpenOC }) {
  const [open, setOpen] = useState(false);
  const [popupPos, setPopupPos] = useState({ top: 0, left: 0 });
  const dotRef = useRef(null);
  const popupRef = useRef(null);
  const closeTimer = useRef(null);
  const isM = act.t === "multiple";
  const pct = 100 - (act.d / 90) * 100;

  const dotSizeMap = { crm: 10, email: 12, meeting: 16, multiple: 20 };
  const dotSize = dotSizeMap[act.t] || 10;

  const updatePos = () => {
    if (!dotRef.current) return;
    const rect = dotRef.current.getBoundingClientRect();
    const popupW = 300;
    let left = rect.left + rect.width / 2 - popupW / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - popupW - 8));
    const top = rect.top - 10;
    setPopupPos({ top, left });
  };

  const handleDotEnter = () => {
    clearTimeout(closeTimer.current);
    updatePos();
    setOpen(true);
  };

  const handleDotLeave = () => {
    closeTimer.current = setTimeout(() => setOpen(false), 120);
  };

  const handlePopupEnter = () => {
    clearTimeout(closeTimer.current);
  };

  const handlePopupLeave = () => {
    closeTimer.current = setTimeout(() => setOpen(false), 120);
  };

  const handleOpenOCFromPopup = (target) => {
    setOpen(false);
    onOpenOC(target);
  };

  return (
    <>
      <div
        className="opd-dot-anchor"
        style={{ left: `${pct}%`, zIndex: open ? 200 : 10 }}
      >
        <button
          ref={dotRef}
          className="opd-dot-btn"
          style={{ background: DC[act.t], width: dotSize, height: dotSize, border: '2px solid #E2E8F0' }}
          onMouseEnter={handleDotEnter}
          onMouseLeave={handleDotLeave}
          onClick={() => {
            clearTimeout(closeTimer.current);
            setOpen(false);
            onOpenOC(deal);
          }}
        >
          {isM ? act.count || "+" : ""}
        </button>
      </div>

      {open && (
        <div
          ref={popupRef}
          className="opd-popup-portal"
          style={{
            top: popupPos.top,
            left: popupPos.left,
            transform: "translateY(-100%)",
            marginTop: -10,
          }}
          onMouseEnter={handlePopupEnter}
          onMouseLeave={handlePopupLeave}
        >
          <DotPopup acts={acts} deal={deal} onOpenOC={handleOpenOCFromPopup} />
        </div>
      )}
    </>
  );
}

/* ─────────────────────────────────────────────
   TIMELINE CELL
───────────────────────────────────────────── */
function TimelineCell({ acts, deal, onOpenOC }) {
  const safeActs = acts && acts.length ? acts : [];

  return (
    <div className="opd-tl-cell">
      <div className="opd-tl-line" />
      {safeActs.map((act, i) => (
        <ActivityDot
          key={i}
          act={act}
          acts={safeActs}
          deal={deal}
          onOpenOC={onOpenOC}
        />
      ))}
    </div>
  );
}

function OffcanvasPanel({ deal, open, onClose }) {
  const [ocTab, setOcTab] = useState("activity");

  useEffect(() => {
    if (open) {
      setOcTab("activity");
    }
  }, [open]);

  if (!deal) return null;

  const items = buildActivityItems(deal);

  return (
    <>
      <div
        className={`opd-oc-overlay${open ? " opd-oc-open" : ""}`}
        onClick={onClose}
      />
      <div className={`opd-oc-panel${open ? " opd-oc-open" : ""}`}>
        <div className="opd-oc-header">
          <div>
            <span className="opd-oc-title">Activity Timeline</span>
            <span className="opd-oc-subtitle">· {deal.name}</span>
          </div>
          <button className="opd-oc-close" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* period-tab style OC tabs */}
        <div className="opd-oc-tabs">
          {[
            { id: "activity", label: "Activity" },
            { id: "warnings", label: "Warnings" },
            { id: "contacts", label: "Contacts" },
          ].map((tab) => (
            <button
              key={tab.id}
              className={`opd-oc-tab${ocTab === tab.id ? " opd-oc-tab-active" : ""}`}
              onClick={() => setOcTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="opd-oc-body">
          {/* Activity pane */}
          <div className={`opd-oc-pane${ocTab === "activity" ? " opd-oc-pane-active" : ""}`}>
            <div className="opd-sub-tabs">
              <span className="opd-sub-title">
                <img
                  src={leftArrowIcon}
                  alt="left"
                  className="opd-sub-icon me-1 pb-1"
                />
                Inbound
              </span>
              <span className="opd-sub-title">
                Outbound
                <img
                  src={rightArrowIcon}
                  alt="right"
                  className="opd-sub-icon ms-1 pb-1"
                />
              </span>
            </div>
            <div className="opd-zz-wrap">
              <div className="opd-zz-spine" />
              {items.map((item, i) => {
                const isLeft = item.direction === "inbound";
                return (
                  <div
                    key={i}
                    className={`opd-zz-item ${isLeft ? "opd-zz-left" : "opd-zz-right"}`}
                  >
                    <div className="opd-zz-card-wrap">
                      <div className="opd-zz-card">
                        <div className="opd-zz-card-hd">
                          <span
                            className="opd-zz-type-dot"
                            style={{ background: DC[item.type] }}
                          />
                          <span className="opd-zz-type-lbl">
                            {DL[item.type]}
                          </span>
                        </div>
                        <div className="opd-zz-date">{item.date}</div>
                        <div className="opd-zz-title">{item.title}</div>
                        <div
                          className="opd-zz-body"
                          dangerouslySetInnerHTML={{
                            __html: sanitizeActivityHtml(item.body),
                          }}
                        />
                      </div>
                    </div>
                    <div className="opd-zz-node">
                      <div className="opd-zz-node-outer">
                        <div
                          className="opd-zz-node-inner"
                          style={{ background: DC[item.type] }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          
        </div>
      </div>
    </>
  );
}

function HealthRing({ percent }) {
  const color = hColor(percent);
  const r = 15,
    cx = 19,
    cy = 19,
    circ = 2 * Math.PI * r;
  const dash = (percent / 100) * circ;
  return (
    <div className="opd-ring-wrap">
      <svg width="38" height="38" viewBox="0 0 38 38">
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="#e9ecef"
          strokeWidth="3.5"
        />
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="3.5"
          strokeDasharray={`${dash.toFixed(2)} ${circ.toFixed(2)}`}
          strokeLinecap="butt"
          transform="rotate(-90 19 19)"
        />
      </svg>
      <span className="opd-ring-label" style={{ color: "#334155" }}>
        {percent}%
      </span>
    </div>
  );
}

export default function Opportunities() {
  const { id } = useParams();
  const opportunities = [
    {
      id: "OPP-001",
      name: "ThinkPad Fleet Refresh",
      value: "$1.2M",
      stage: "Proposal",
      closeDate: "Jun 30, 2026",
      health: "85%",
    },
    {
      id: "OPP-002",
      name: "Server Infrastructure Upgrade",
      value: "$850K",
      stage: "Negotiation",
      closeDate: "Jul 15, 2026",
      health: "72%",
    },
    {
      id: "OPP-003",
      name: "Cloud Migration Services",
      value: "$400K",
      stage: "Discovery",
      closeDate: "Aug 20, 2026",
      health: "60%",
    },
  ];

  const DD_PLACEHOLDER = {
    Regions: [{ v: "all", l: "Regions" }],
    Industries: [{ v: "all", l: "Industries" }],
    Stage: [{ v: "all", l: "Stage" }],
    Products: [{ v: "all", l: "Products" }],
  };

  const [activeBucket, setActiveBucket] = useState(null);
  const [search, setSearch] = useState("");
  // const [filters, setFilters] = useState({ Regions: "all", Industries: "all", Stage: "all", Products: "all" });
  const [filters, setFilters] = useState({
    Regions: "all",
    Industries: "all",
    Stage: "all",
    Segments: "all",
    AccountTypes: "all",
  });
  const [ddOptions, setDdOptions] = useState(DD_PLACEHOLDER);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [kpis, setKpis] = useState(SUMMARY_PLACEHOLDER);
  const [page, setPage] = useState(1);
  const [periodTab, setPeriodTab] = useState("week");
  const [error, setError] = useState(null);
  const [ocOpen, setOcOpen] = useState(false);
  const [ocDeal, setOcDeal] = useState(null);

  const openOC = useCallback((deal) => {
    setOcDeal(deal);
    setOcOpen(true);
  }, []);

  useEffect(() => {
    const id = "opd-styles";
    let style = document.getElementById(id);
    if (!style) {
      style = document.createElement("style");
      style.id = id;
      document.head.appendChild(style);
    }
    style.textContent = STYLES;
  }, []);

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
      .catch((err) => console.error(err));

    return () => ctrl.abort();
  }, []);

  // useEffect(() => {
  //   const ctrl = new AbortController();
  //   // const params = {
  //   //   search: search.trim() || undefined,
  //   //   regions: filters.Regions !== "all" ? filters.Regions : undefined,
  //   //   industries: filters.Industries !== "all" ? filters.Industries : undefined,
  //   //   stages: filters.Stage !== "all" ? filters.Stage : undefined,
  //   //   products: filters.Products !== "all" ? filters.Products : undefined,
  //   // };

  //   const params = {
  //     page,
  //     pageSize: PAGE_SIZE,
  //     sortBy: "closeDate",
  //     sortOrder: "desc",
  //   };

  //   if (search.trim()) params.search = search.trim();

  //   if (filters.Regions !== "all") params.regions = filters.Regions;
  //   if (filters.Industries !== "all") params.industries = filters.Industries;
  //   if (filters.Stage !== "all") params.accountStatuses = filters.Stage;
  //   if (filters.Segments !== "all") params.segments = filters.Segments;
  //   if (filters.AccountTypes !== "all") params.accountTypes = filters.AccountTypes;

  //   const handle = setTimeout(() => {
  //     setLoading(true);
  //     setError(null);

  //     console.log("API PARAMS:", params);

  //     fetchAccountOpportunities(
  //       {
  //         ...params,
  //         bucket: activeBucket || undefined,
  //         page,
  //         pageSize: PAGE_SIZE,
  //         sortBy: "closeDate",
  //         sortOrder: "desc"
  //       },
  //       ctrl.signal
  //     )
  //       .then(opps => {
  //         setItems((opps.items || []).map(mapOpportunityRow));
  //         setTotal(opps.total ?? 0);
  //         setTotalPages(opps.totalPages ?? 1);
  //       })
  //       .catch(err => {
  //         if (err.name === "AbortError") return;

  //         console.error("API ERROR:", err);
  //         setError(err.message);
  //         setItems([]);
  //       })
  //       .finally(() => setLoading(false));

  //   }, 250);

  //   return () => {
  //     clearTimeout(handle);
  //     ctrl.abort();
  //   };
  // }, [
  //   search,
  //   filters.Regions,
  //   filters.Industries,
  //   filters.Stage,
  //   filters.Segments,
  //   filters.AccountTypes,
  //   periodTab,
  //   page,
  //   activeBucket
  // ]);

  useEffect(() => {
    const ctrl = new AbortController();

    const hasSearch = !!search.trim();
    const hasDropdownFilters =
      filters.Regions !== "all" ||
      filters.Industries !== "all" ||
      filters.Stage !== "all" ||
      filters.Segments !== "all" ||
      filters.AccountTypes !== "all";

    const handle = setTimeout(() => {
      setLoading(true);
      setError(null);

      // ✅ BUILD PARAMS
      const params = {};

      if (search.trim()) params.search = search.trim();
      if (filters.Regions !== "all") params.regions = filters.Regions;
      if (filters.Industries !== "all") params.industries = filters.Industries;
      if (filters.Stage !== "all") params.accountStatuses = filters.Stage;
      if (filters.Segments !== "all") params.segments = filters.Segments;
      if (filters.AccountTypes !== "all")
        params.accountTypes = filters.AccountTypes;

      console.log("FILTER PARAMS:", params);

      // ✅ DECIDE API
      const apiCall = hasSearch
        ? fetchAccountOpportunities(
            id,
            {
              ...params,
              page,
              pageSize: PAGE_SIZE,
              sortBy: "closeDate",
              sortOrder: "desc",
            },
            ctrl.signal,
          )
        : hasDropdownFilters
          ? fetchFilteredAccounts(params, ctrl.signal)
          : fetchAccountOpportunities(
              id, // ✅ INITIAL API
              {
                page,
                pageSize: PAGE_SIZE,
                sortBy: "closeDate",
                sortOrder: "desc",
              },
              ctrl.signal,
            );

      apiCall
        .then((data) => {
          // ⚠️ RESPONSE SHAPE MAY DIFFER!
          const list = data.items || data || [];
          console.log(list);
          setItems(list.map(mapOpportunityRow));
          setTotal(data.total ?? list.length);
          setTotalPages(data.totalPages ?? 1);
        })
        .catch((err) => {
          if (err.name === "AbortError") return;
          console.error(err);
          setError(err.message);
          setItems([]);
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
    page,
  ]);

  const ticks = [
    { l: "90d", p: 0 },
    { l: "60d", p: 33.3 },
    { l: "30d", p: 66.6 },
    { l: "0d", p: 100 },
  ];
  const handleFilterChange = useCallback((key, val) => {
    setPage(1);
    setFilters((prev) => ({ ...prev, [key]: val }));
  }, []);
  const handleSearchChange = useCallback((e) => {
    setPage(1);
    setSearch(e.target.value);
  }, []);
  const pageButtons = useMemo(
    () => buildPageList(totalPages, page, 3),
    [totalPages, page],
  );

  const handleDownload = useCallback(() => {
    const headers = [
      "Name",
      "Company",
      "Risk",
      "Region",
      "Industry",
      "Stage",
      "Product",
      "Competitors",
      "Value",
      "Close Date",
      "Sale Motion",
      "Health",
      "Last Activity",
    ];

    const normalizeForCsv = (value) => {
      const raw = String(value ?? "").trim();
      if (!raw) return "N/A";
      const lowered = raw.toLowerCase();
      if (
        raw === "—" ||
        raw === "â€”" ||
        raw === "#" ||
        lowered === "none" ||
        lowered === "null"
      )
        return "N/A";
      return raw;
    };

    const rows = items.map((d) => [
      normalizeForCsv(d.name),
      normalizeForCsv(d.company),
      normalizeForCsv(d.risk),
      normalizeForCsv(d.region),
      normalizeForCsv(d.industry),
      normalizeForCsv(d.stage),
      normalizeForCsv(d.product),
      normalizeForCsv(d.comp),
      normalizeForCsv(d.value),
      normalizeForCsv(d.closeDate),
      normalizeForCsv(d.motion),
      normalizeForCsv(d.health != null ? `${d.health}%` : "-"),
      normalizeForCsv(d.lastAct),
    ]);

    const escapeCsv = (value) => {
      const text = normalizeForCsv(value);
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };

    const csv = [headers, ...rows]
      .map((row) => row.map(escapeCsv).join(","))
      .join("\n");

    const bom = "\uFEFF";
    const blob = new Blob([bom + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `opportunities-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }, [items]);
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

  return (
    <>
      <style>{`
        .opp-wrap { background: #fff; border: 1px solid var(--opd-border);border-radius: 12px; padding: 24px; overflow: hidden; max-width: 100%; }
        .opp-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
        .opp-header h2 { font-size: 20px; font-weight: 700; color: #0f172a; margin: 0; }
        .opp-btn { background: #1D4ED8; color: #fff; border: none; border-radius: 50px; padding: 8px 20px; font-size: 13px; font-weight: 600; cursor: pointer; }
        .opp-btn:hover { background: #1F3A8A; }
        .opp-table { width: 100%; border-collapse: collapse; }
        .opp-table th { text-align: left; padding: 12px 16px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; font-size: 12px; font-weight: 600; color: #64748b; }
        .opp-table td { padding: 16px; border-bottom: 1px solid #f1f5f9; font-size: 14px; color: #334155; }
        .opp-table tr:hover { background: #f8fafc; }
        .opp-name { font-weight: 600; color: #1a56db; cursor: pointer; }
        .opp-name:hover { text-decoration: underline; }
        .opp-stage { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 500; }
        .opp-stage-proposal { background: #eff6ff; color: #1d4ed8; }
        .opp-stage-negotiation { background: #fff7ed; color: #c2410c; }
        .opp-stage-discovery { background: #f0fdf4; color: #16a34a; }
        .opp-health { font-weight: 600; }
        .opp-health-high { color: #16a34a; }
        .opp-health-mid { color: #f59e0b; }
        .opp-health-low { color: #dc2626; }
      `}</style>

      <div className="opp-wrap">
        {/* <div className="opp-header">
          <h2>Opportunities</h2>
          <button className="opp-btn">+ New Opportunity</button>
        </div>

        <table className="opp-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Opportunity Name</th>
              <th>Value</th>
              <th>Stage</th>
              <th>Close Date</th>
              <th>Health</th>
            </tr>
          </thead>
          <tbody>
            {opportunities.map((opp) => (
              <tr key={opp.id}>
                <td>{opp.id}</td>
                <td><span className="opp-name">{opp.name}</span></td>
                <td>{opp.value}</td>
                <td>
                  <span className={`opp-stage opp-stage-${opp.stage.toLowerCase()}`}>{opp.stage}</span>
                </td>
                <td>{opp.closeDate}</td>
                <td>
                  <span className={`opp-health ${parseInt(opp.health) >= 80 ? 'opp-health-high' : parseInt(opp.health) >= 60 ? 'opp-health-mid' : 'opp-health-low'}`}>
                    {opp.health}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table> */}

        <div className="opd-filter-bar">
          <div className="opd-search-box">
            <svg
              className="opd-search-icon"
              width="13"
              height="13"
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
              onChange={handleSearchChange}
              placeholder="Search Opportunity"
            />
          </div>
          {["Regions", "Industries", "Stage", "Segments", "AccountTypes"].map(
            (key) => (
              <Dropdown
                key={key}
                filterKey={key}
                value={filters[key]}
                options={ddOptions[key]}
                onChange={handleFilterChange}
              />
            ),
          )}
          {/* <button className="opd-btn-clear" onClick={clearFilters}>Clear filters</button> */}
          {/* <button className="opd-btn-dl" onClick={handleDownload}>
            <i className="bi bi-download"></i>
          </button> */}
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
                  <col style={{ width: 80 }} />
                </colgroup>
                <thead style={{ height: 66.3 }}>
                  <tr>
                    <th className="opd-th">Name</th>
                    <th className="opd-th">Risk</th>
                  </tr>
                  <tr className="opd-spacer-row">
                    <td colSpan={2} />
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
                    items.map((d, i) => (
                      <tr
                       
                        key={d.id || i}
                        className={i % 2 === 0 ? "opd-row-even" : "opd-row-odd"}
                      >
                        <td className="opd-td">
                          <div className="opd-deal-name">{d.name}</div>
                          <div className="opd-deal-co">{d.company}</div>
                        </td>
                        <td className="opd-td">
                          {d.risk || d.riskScore != null ? (
                            <span
                              className="opd-risk-badge"
                              title={d.risk || "Risk flagged"}
                            >
                              <span className="sd-badge-icon-warning">
                                <i className="bi bi-exclamation-triangle-fill me-2"></i>
                              </span>
                              {d.riskScore ?? 1}
                            </span>
                          ) : (
                            <span className="opd-no-risk">–</span>
                          )}
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
                    <th className="opd-th" style={{ minWidth: 530 }}>
                      Activity Timeline
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 4, marginLeft: 8 }}>
                        <svg width="10" height="10" viewBox="0 0 10 10" style={{ transform: "rotate(90deg)" }}>
                          <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                        </svg>
                        <span style={{ fontSize: 12, fontWeight: 400, color: "#0F172A" }}>scroll</span>
                        <svg width="10" height="10" viewBox="0 0 10 10" style={{ transform: "rotate(-90deg)" }}>
                          <path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                        </svg>
                      </span>
                    </th>
                    <th className="opd-th" style={{ minWidth: 115 }}>
                      Sale Motion
                    </th>
                    <th className="opd-th" style={{ minWidth: 110 }}>
                      Deal Health
                    </th>
                    <th className="opd-th" style={{ minWidth: 190 }}>
                      Competitors
                    </th>
                    <th className="opd-th" style={{ minWidth: 140 }}>
                      Next Action
                    </th>
                    <th className="opd-th" style={{ minWidth: 90 }}>
                      Value
                    </th>
                    <th className="opd-th" style={{ minWidth: 100 }}>
                      Close date
                    </th>
                    <th className="opd-th" style={{ minWidth: 115 }}>
                      Stage
                    </th>
                    <th className="opd-th" style={{ minWidth: 110 }}>
                      Last Activity
                    </th>
                  </tr>

                  <tr className="opd-spacer-row">
                    <td colSpan={9} className="opd-spacer-cell">
                      <div className="opd-tl-header-spacer">
                        <div className="opd-tl-ticks">
                          {ticks.map((t) => (
                            <span
                              key={t.l}
                              className="opd-tl-tick"
                              style={{ left: `${(t.p / 100) * 540}px` }}
                            >
                              {t.l}
                            </span>
                          ))}
                        </div>
                      </div>
                    </td>
                  </tr>
                </thead>
                <tbody>
                  {loading && items.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="opd-no-results">
                        Loading opportunities…
                      </td>
                    </tr>
                  ) : error ? (
                    <tr>
                      <td
                        colSpan={9}
                        className="opd-no-results"
                        style={{ color: "#ef4444" }}
                      >
                        {error}
                      </td>
                    </tr>
                  ) : items.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="opd-no-results">
                        No deals match your filters
                      </td>
                    </tr>
                  ) : (
                    items.map((d, i) => (
                      <tr
                        key={d.id || i}
                        className={i % 2 === 0 ? "opd-row-even" : "opd-row-odd"}
                      >
                        <td
                          className="opd-td"
                          style={{ paddingTop: 0, paddingBottom: 0 }}
                        >
                          <TimelineCell
                            acts={d.acts}
                            deal={d}
                            onOpenOC={openOC}
                          />
                        </td>
                        <td className="opd-td">
                          <span
                            className={`opd-motion-badge opd-motion-${String(d.motion).toLowerCase().replace(/\s+/g, "-")}`}
                          >
                            {d.motion}
                          </span>
                        </td>
                        <td className="opd-td">
                          {d.health != null ? (
                            <HealthRing percent={d.health} />
                          ) : (
                            <span className="opd-no-risk">—</span>
                          )}
                        </td>
                        <td
                          className="opd-td"
                          style={{  fontSize: 14 }}
                        >
                          {d.comp}
                        </td>
                        <td
                          className="opd-td"
                          style={{ color: "var(--opd-text3)", fontSize: 12 }}
                        >
                          —
                        </td>
                        <td
                          className="opd-td"
                          style={{
                            fontWeight: 700,
                            color: "#212529",
                            fontSize: 13,
                          }}
                        >
                          {d.value}
                        </td>
                        <td
                          className="opd-td"
                          style={{ color: "var(--opd-text3)" }}
                        >
                          {d.closeDate}
                        </td>
                        <td className="opd-td">
                          <span className={stageClass(d.stageClass)}>
                            {d.stage}
                          </span>
                        </td>
                        <td
                          className="opd-td"
                          style={{ color: "var(--opd-text3)" }}
                        >
                          {d.lastAct}
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
            <span style={{ fontSize: 14, color: "var(--opd-text3)" }}>
              {total === 0
                ? "Showing 0 deals"
                : `Showing ${items.length} of ${total} deals`}
            </span>
            <div className="opd-legend-row">
              {Object.entries(DC).map(([type, color]) => {
                const sizeMap = { crm: 10, email: 12, meeting: 16, multiple: 20 };
                const size = sizeMap[type] || 10;
                return (
                  <span key={type} className="opd-legend-item">
                    <span
                      className="opd-legend-dot"
                      style={{
                        border: `2px solid #E2E8F0`,
                        background: color,
                        width: size,
                        height: size,
                      }}
                    >
                      {type === "multiple" ? "5" : ""}
                    </span>
                    {DL[type]}
                  </span>
                );
              })}
            </div>
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
                <i className="bi bi-chevron-left"></i>
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
                <i className="bi bi-chevron-right"></i>
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

      <OffcanvasPanel
        deal={ocDeal}
        open={ocOpen}
        onClose={() => setOcOpen(false)}
      />
    </>
  );
}
