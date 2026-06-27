import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  useLayoutEffect,
} from "react";
import {
  fetchKpiSummary,
  fetchOpportunities,
  fetchFilterOptions,
} from "../api/client";
import {
  buildPageList,
  formatCurrencyShort,
  formatDateMMDDYY,
  formatDealCount,
  formatDelta,
} from "../utils/format";
import { useNavigate } from "react-router-dom";
import greenArrowIcon from "../assets/icons/green.png";
import redArrowIcon from "../assets/icons/red.png";
import leftArrowIcon from "../assets/icons/left.png";
import rightArrowIcon from "../assets/icons/right1.png";
import downloadIcon from "../assets/download.png";

/* ─────────────────────────────────────────────
   GLOBAL STYLES  (injected once into <head>)
───────────────────────────────────────────── */
const STYLES = `

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
.opd-period-tab { padding: 7px 16px; border: none; border-right: 1px solid var(--opd-border); background: transparent;  font-size: 14px; color: var(--opd-text3); cursor: pointer; transition: background .15s, color .15s; white-space: nowrap; }.opd-period-tab:last-child { border-right: none; }
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
.opd-risk-badge { background: #FFF7ED !important; padding:4px 10px !important}
.opd-scard[data-tip]:hover::after,
.opd-scard[data-tip]:hover::before,
.opd-scard[data-tip]:focus-visible::after,
.opd-scard[data-tip]:focus-visible::before { opacity: 1; }
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
.opd-search-box input { border: none; outline: none; font-size: 14px; color: #495057; background: transparent; width: 100%;  }
.opd-search-icon { color: #475569; flex-shrink: 0; }
.opd-btn-clear { padding: 5px 12px; border-radius: 6px; border: 1px solid var(--opd-border); background: #fff; color: var(--opd-text3); font-size: 12px; cursor: pointer;  white-space: nowrap; }
.opd-btn-clear:hover { background: #f8f9fa; }
.opd-btn-dl { width: 40px; height: 40px; border-radius: 50px; border:none; background: #E2E8F0; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--opd-text3); font-size: 16px; flex-shrink: 0; }

/* ── Custom dropdown ── */
.opd-dd-wrap { position: relative; flex: 1 1 140px; min-width: 120px; }
.opd-dd-trigger { display: flex; align-items: center; border: 2px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; font-size: 12px; color: #495057; background: #f8fafc; cursor: pointer; white-space: nowrap; min-width: 110px; width: 100%; user-select: none; transition: border-color .15s; height:40px }
.opd-dd-trigger.opd-dd-open { border-color: var(--opd-accent); border-radius: 6px 6px 0 0; }
.opd-dd-trigger span:not(.opd-dd-arrow) { flex: 1; }
.opd-dd-arrow {        width: 10px; display: inline-flex; align-items: center; margin-left: 10px; flex-shrink: 0; font-size: 10px; color: #adb5bd; transition: transform .15s; }
.opd-dd-trigger.opd-dd-open .opd-dd-arrow { transform: rotate(180deg); }
.opd-dd-menu { position: absolute; top: 100%; left: 0; right: 0; background: #fff;  border-top: none; border-radius: 0 0 6px 6px; z-index: 500; overflow: hidden; box-shadow: 0 4px 4px rgb(0 0 0 / 21%); }
.opd-dd-item { display: flex; align-items: center; gap: 7px; padding: 7px 10px; font-size: 12px; cursor: pointer; color: #495057; transition: background .1s; }
.opd-dd-item:hover { background: #f1f5ff; }
.opd-dd-item.opd-dd-selected { color: var(--opd-accent); background: #f1f5f9; ; font-weight: 500; }
.opd-dd-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

.opd-btn-clear,
.opd-btn-dl { flex-shrink: 0; }

/* ── Table card ── */
.opd-table-card { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; overflow: visible; max-width: 100%; }
.opd-table-outer { display: flex; overflow: visible; max-width: 100%; }
.opd-frozen-panel { flex-shrink: 0; border-right: 2px solid var(--opd-border); z-index: 2;    width: 310px; overflow: visible; }
.opd-frozen-panel table { border-collapse: collapse; table-layout: fixed; width: 100%; }
.opd-scroll-panel { flex: 1; overflow-x: auto; overflow-y: hidden; min-width: 0; }
.opd-scroll-panel table { border-collapse: collapse; min-width: 900px; }
.opd-th { padding: 10px 13px; text-align: left; font-weight: 600; font-size: 14px; color: #212529; white-space: nowrap; background: #f8fafc; border-bottom: 1px solid var(--opd-border); }
.opd-td { padding: 0 13px; height: 54px; vertical-align: middle; font-size: 14px; }
.opd-row-even { background: #fff; border-bottom: 1px solid var(--opd-border); position: relative; }
.opd-row-odd  { background: #fff; border-bottom: 1px solid var(--opd-border); position: relative; }
.opd-row-even:hover, .opd-row-odd:hover { z-index: 60; }
.opd-spacer-row td { height: 25px; padding: 0; border-bottom: 1px solid var(--opd-border) ; background: #f8fafc;}
.opd-spacer-cell { padding: 0 13px; }
.opd-tl-header-spacer { height: 24px; display: flex; align-items: center; }
.opd-frozen-panel tbody .opd-td { background: #f8fafc; }
.opd-deal-name {  color: #0f172a; font-size: 14px; line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; position: relative; }
.opd-deal-co   { font-size: 12px; color: #64748b; font-weight: 500; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }

/* ── Deal name tooltip ── */
.opd-name-cell { position: relative; overflow: visible; }
.opd-name-cell[data-tip]::after,
.opd-name-cell[data-tip]::before { opacity: 0; pointer-events: none; transition: opacity .15s ease; position: absolute; z-index: 99999; }
.opd-name-cell[data-tip]::after {
  content: attr(data-tip);
  left: 25%;
  bottom: calc(100% + 8px);
  transform: translateX(-50%);
  margin-left: 100px;
  background: #0f172a;
  color: #fff;
  font-size: 11px;
  font-weight: 500;
  line-height: 1.4;
  padding: 6px 10px;
  border-radius: 6px;
  white-space: nowrap;
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  box-shadow: 0 4px 10px rgba(0,0,0,.2);
}
.opd-name-cell[data-tip]::before {
  content: "";
  left: 25%;
  bottom: calc(100% + 2px);
  transform: translateX(-50%);
  margin-left: 100px;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 6px solid #0f172a;
}
.opd-name-cell[data-tip]:hover::after,
.opd-name-cell[data-tip]:hover::before { opacity: 1; }

/* ── Health ring ── */
.opd-ring-wrap { position: relative; width: 38px; height: 38px; flex-shrink: 0; }
.opd-ring-label { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 9.6px; font-weight: 400;}

/* ── Timeline cell ── */
.opd-tl-cell { position: relative; width: 600px; height: 28px; display: flex; align-items: center; }
.opd-tl-line { position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: #e9ecef; }
.opd-tl-ticks { position: relative; width: 600px; height: 16px;margin-left: 60px !important; }
.opd-tl-tick { position: absolute; transform: translateX(-50%); font-size: 10px; color: #334155 white-space: nowrap; }

/* ── Activity dot ── */
.opd-dot-anchor { position: absolute; top: 50%; transform: translate(-50%, -50%); }
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
  overflow: visible;
  pointer-events: all;
  animation: opd-pop-in .15s ease;
  padding: 6px;
  margin: 10px;
}
@keyframes opd-pop-in { from { opacity:0; transform:translateY(6px) scale(.97); } to { opacity:1; transform:none; } }

.opd-popup-header { display: flex; align-items: center; gap: 8px; padding: 7px 12px 6px; border-bottom: 1px solid #f3f4f6; }
.opd-popup-header-title { font-size: 11px; font-weight: 600; color: #000; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }

/* mini right-spine timeline in popup */
.opd-ptl-body { overflow-y: auto; padding: 10px 12px 10px 8px; }
.opd-ptl-item { display: flex; align-items: stretch; }
.opd-ptl-content { flex: 1; padding-left: 10px; padding-bottom: 16px; }
.opd-ptl-item:last-child .opd-ptl-content { padding-bottom: 4px; }
.opd-ptl-type { font-size: 10px; font-weight: 600; color: var(--opd-accent); text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 1px; }
.opd-ptl-title-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 2px; }
.opd-ptl-title { font-size: 12px; font-weight: 600; color: var(--opd-text); line-height: 1.35; }
.opd-ptl-time { font-size: 10px; color: #000; white-space: nowrap; flex-shrink: 0; }
.opd-ptl-body-text { font-size: 12px; color: #000; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; }
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
.opd-pag-btn { width: 26px; height: 26px; border-radius: 5px; border: 1px solid var(--opd-border); background: #fff; color: #495057; font-size: 11px; cursor: pointer;  transition: background .1s; }

/* ── Offcanvas ── */
.opd-oc-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.22); z-index: 8000; opacity: 0; pointer-events: none; transition: opacity .28s; }
.opd-oc-overlay.opd-oc-open { opacity: 1; pointer-events: all; }
.opd-oc-panel { position: fixed; top: 0; right: 0; height: 100%; width: 520px; max-width: 96vw; background: var(--opd-bg); z-index: 8100; display: flex; flex-direction: column; border-left: 1.5px solid var(--opd-border); border-top-left-radius: 18px; border-bottom-left-radius: 18px; overflow: hidden; transform: translateX(100%); transition: transform .28s cubic-bezier(.4,0,.2,1); }
.opd-oc-panel.opd-oc-open { transform: translateX(0); }
.opd-oc-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px 14px; border-bottom: 1px solid var(--opd-border); flex-shrink: 0; }
.opd-oc-title { font-size: 16px; font-weight: 700; color: var(--opd-text); }
.opd-oc-subtitle { font-size: 11px; color: var(--opd-muted); margin-left: 8px; }
.opd-oc-close { width: 28px; height: 28px; border-radius: 6px; border:none; background: #ffffff00; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px;  }

/* OC tabs — same pill design as period tabs */
.opd-oc-tabs { display: flex; width: 300px; border: 1px solid var(--opd-border); border-radius: 8px; overflow: hidden; margin: 12px auto; background: #fff; flex-shrink: 0; }
.opd-oc-tab { flex: 1; padding: 7px 10px; border: none; border-right: 1px solid var(--opd-border); background: transparent;  font-size: 12px; color: var(--opd-text3); cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 5px; transition: background .15s, color .15s; white-space: nowrap; font-weight: 400; }
.opd-oc-tab:last-child { border-right: none; }
.opd-oc-tab.opd-oc-tab-active { background: #e8f0fe; color: var(--opd-accent); font-weight: 600; }

/* OC activity subtitles (left=Inbound, right=Outbound on the zigzag) */
.opd-sub-tabs { display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 16px; background: transparent; }
.opd-sub-title { padding: 6px 12px; font-size: 16px; color: #334155; font-weight: 600; line-height: 1; white-space: nowrap; }
.opd-sub-icon { width: 18px; height: 18px; object-fit: contain; vertical-align: middle; margin-right: 5px; }

.opd-oc-panel .opd-oc-body { flex: 1; overflow-y: auto; padding: 16px 20px 28px; }
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
.opd-zz-date { font-size: 11px; color: var(--opd-muted); margin-bottom: 4px; }
.opd-zz-title { font-size: 16px; font-weight: 600; color: var(--opd-text); margin-bottom: 6px; line-height: 1.4; }
.opd-zz-body { font-size: 12px;  line-height: 1.65; }
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
.opd-contact-role { font-size: 11px; color: var(--opd-muted); }
.opd-contact-email { font-size: 11px; color: var(--opd-accent); margin-top: 2px; }
.opd-contact-tag { margin-left: auto; font-size: 10px; padding: 3px 8px; border-radius: 20px; background: #e8f0fe; color: var(--opd-accent); font-weight: 500; white-space: nowrap; }

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
    label: "Identified",
    bucket: "pipeline",
    value: "—",
    count: "—",
    delta: "",
    up: true,
  },

  {
    label: "Commit",
    bucket: "commit",
    value: "—",
    count: "—",
    delta: "",
    up: false,
  },
  {
    label: "Most Likely",
    bucket: "most_likely",
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
const DD_PLACEHOLDER = {
  Regions: [{ v: "all", l: "Regions" }],
  Industries: [{ v: "all", l: "Industries" }],
  Stage: [{ v: "all", l: "Stage" }],
  Products: [{ v: "all", l: "Products" }],
};

/** Deterministic colour palette used for the dot on each dropdown option. */
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

/** Days between now and an ISO timestamp, clamped to the 0–90d timeline. */
function daysAgo(iso) {
  if (!iso) return 0;
  const ms = Date.now() - new Date(iso).getTime();
  const d = Math.round(ms / 86_400_000);
  return Math.max(0, Math.min(90, d));
}

/** Format an activity timestamp, e.g. "18 Apr 2026 · 10:00 AM". */
function formatActivityWhen(iso) {
  if (!iso) return "";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return "";
  const date = dt
    .toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    })
    .replace(/\//g, ".");
  const time = dt.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  return `${date} · ${time}`;
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

/** Group activities into 0d/30d/60d/90d buckets for the timeline dots. */
function buildTimelineBuckets(activities) {
  const mapped = (activities || []).map(mapActivity);

  const buckets = [
    { label: "0d", min: 0, max: 14 },
    { label: "30d", min: 15, max: 44 },
    { label: "60d", min: 45, max: 74 },
    { label: "90d", min: 75, max: 90 },
  ];

  return buckets
    .map((bucket) => {
      const items = mapped.filter(
        (a) => a.d >= bucket.min && a.d <= bucket.max,
      );
      if (items.length === 0) return null;
      if (items.length === 1) {
        const pct = 100 - (items[0].d / 90) * 100;
        return { ...items[0], bucketPct: pct, bucketItems: items };
      }
      const avgD = items.reduce((sum, a) => sum + a.d, 0) / items.length;
      const pct = 100 - (avgD / 90) * 100;
      return {
        t: "multiple",
        d: avgD,
        date: items[0].date,
        title: `${items.length} Activities`,
        body: "",
        direction: null,
        count: items.length,
        bucketPct: pct,
        bucketItems: items,
      };
    })
    .filter(Boolean);
}

/**
 * Seeded opportunities are stored as "<Brand alias> – <Deal Title>"
 * (en/em-dash or hyphen, always padded with spaces). The grid renders the
 * deal title on row 1 and the company name on row 2.
 *
 * Returns `{ name, company }`:
 *   1. If `accountName` was returned by the API, use it as `company` and
 *      strip a matching prefix off `name` (handles future, correctly-joined
 *      data and accounts where the alias differs from the brand prefix).
 *   2. Else, split the name on its first space-padded dash:
 *      `company = left`, `name = right` (this is the path the screenshot
 *      hits today since `accountName` is null).
 *   3. Else (no dash at all), keep the name and leave company empty.
 *
 * The dash must be space-padded (` – `, ` — `, or ` - `) so titles like
 * "End-to-End Migration" or "R&D-Lab Rollout" are preserved.
 */
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
    nextAction: item.nextAction || item.next_action || "—",
    value: formatCurrencyShort(item.value, item.currency),
    closeDate: formatDateMMDDYY(item.closeDate),
    motion: item.saleMotion?.label || item.saleMotion?.raw || "—",
    health: item.dealHealth ?? null,
    lastAct: formatDateMMDDYY(item.lastActivity),
    acts: buildTimelineBuckets(item.activities),
  };
}

/** Map /api/opportunities/kpi-summary into the 6-card structure the JSX expects. */
function mapKpiSummary(payload) {
  if (!payload) return SUMMARY_PLACEHOLDER;
  const cards = [
    ["Open Deals", "open", payload.openDeals],
    ["Identified", "pipeline", payload.pipeline],

    ["Commit", "commit", payload.commit],
    ["Most Likely", "most_likely", payload.mostLikely],
    ["Best Case", "best_case", payload.bestCase],
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

const OC_WARNINGS = [
  {
    icon: "⚠️",
    title: "Budget Freeze Risk",
    cls: "opd-warn-high",
    badge: "High",
    badgeCls: "opd-warn-high",
    desc: "Client flagged a potential Q2 budget freeze. ROI deck submitted to CIO but formal approval is still pending.",
  },
  {
    icon: "🔴",
    title: "Competitor Threat",
    cls: "opd-warn-high",
    badge: "High",
    badgeCls: "opd-warn-high",
    desc: "HP Inc. confirmed as active bid. Pricing revision submitted but competitive response has not been acknowledged.",
  },
  {
    icon: "⏱️",
    title: "Stale Activity",
    cls: "opd-warn-med",
    badge: "Medium",
    badgeCls: "opd-warn-med",
    desc: "No outbound engagement recorded in last 14 days. Risk of deal going cold without a follow-up touchpoint.",
  },
  {
    icon: "📋",
    title: "PO Delay",
    cls: "opd-warn-low",
    badge: "Low",
    badgeCls: "opd-warn-low",
    desc: "Purchase order expected end of June. No formal confirmation received from procurement as of last check.",
  },
];

const OC_CONTACTS = [
  {
    name: "Kiran Reddy",
    role: "VP Procurement",
    email: "k.reddy@infosys.com",
    tag: "Decision Maker",
    color: "#3b5bdb",
    initials: "KR",
  },
  {
    name: "James O'Brien",
    role: "IT Director",
    email: "j.obrien@infosys.com",
    tag: "Technical Lead",
    color: "#0e9f6e",
    initials: "JO",
  },
  {
    name: "Priya Nair",
    role: "Finance Controller",
    email: "p.nair@infosys.com",
    tag: "Budget Owner",
    color: "#7c3aed",
    initials: "PN",
  },
  {
    name: "Arjun Mehta",
    role: "Procurement Manager",
    email: "a.mehta@infosys.com",
    tag: "Influencer",
    color: "#e3a008",
    initials: "AM",
  },
];

/* ─────────────────────────────────────────────
   SMALL SHARED COMPONENTS
───────────────────────────────────────────── */
const TypeIcon = ({ type }) => {
  const icons = {
    email: (
      <>
        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
        <polyline points="22,6 12,13 2,6" />
      </>
    ),
    crm: (
      <>
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <path d="M8 21h8M12 17v4" />
      </>
    ),
    meeting: (
      <>
        <rect x="3" y="4" width="18" height="18" rx="2" />
        <line x1="16" y1="2" x2="16" y2="6" />
        <line x1="8" y1="2" x2="8" y2="6" />
        <line x1="3" y1="10" x2="21" y2="10" />
      </>
    ),
    multiple: (
      <>
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </>
    ),
  };
  return (
    <div></div>
    // <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    //   {icons[type] || icons.crm}
    // </svg>
  );
};

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
        <svg class="opd-dd-arrow" viewBox="0 0 10 10">
          <path
            d="M1 3l4 4 4-4"
            stroke="#0F172A"
            stroke-width="1.5"
            fill="none"
            stroke-linecap="round"
          ></path>
        </svg>
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

/* ─────────────────────────────────────────────
   DOT POPUP  (portal-based so it's never clipped)
───────────────────────────────────────────── */
function DotPopup({ acts, deal, onOpenOC }) {
  const items = buildActivityItems(deal);

  return (
    <div
      className="opd-popup-card"
       style={{ padding: 6, margin: 10 }}
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
              <div className="opd-ptl-body-text">
                {act.body}Lorem ipsum is a dummy{" "}
              </div>
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
  const pct = act.bucketPct != null ? act.bucketPct : 100 - (act.d / 90) * 100;

   const dotSizeMap = { crm: 8, email: 10, meeting: 16, multiple: 20 };
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
          style={{ background: DC[act.t], width: dotSize, height: dotSize }}
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
          {act.bucketItems && act.bucketItems.length > 0 ? (
            <div
              className="opd-popup-card"
              style={{ padding: 0, margin: 10 }}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={() => handleOpenOCFromPopup(deal)}
            >
              <div className="opd-popup-header">
                <div className="opd-popup-header-title">
                  {act.bucketItems.length > 1
                    ? `${act.bucketItems.length} Activities`
                    //  : act.bucketItems[0].date}
                    : act.bucketItems[0].date.split(" \u00b7 ")[0]}
                </div>
              </div>
              <div className="opd-ptl-body">
                {act.bucketItems.map((item, idx) => (
                  <div key={item.title + idx} className="opd-ptl-item">
                    <div className="opd-ptl-node-col">
                      <div
                        className="opd-ptl-dot"
                        style={{ background: DC[item.t] || "#6366f1" }}
                      />
                      {idx < act.bucketItems.length - 1 && (
                        <div className="opd-ptl-line" />
                      )}
                    </div>
                    <div className="opd-ptl-content">
                      <div className="opd-ptl-title-row">
                        <span className="opd-ptl-title">{item.title}</span>
                        {item.date && item.date.includes(" \u00b7 ") && (
                          <span className="opd-ptl-time">
                            {item.date.split(" \u00b7 ")[1]}
                          </span>
                        )}
                      </div>
                      <div className="opd-ptl-body-text">{item.body}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <DotPopup acts={acts} deal={deal} onOpenOC={handleOpenOCFromPopup} />
          )}
        </div>
      )}
    </>
  );
}

/* ─────────────────────────────────────────────
   TIMELINE CELL
───────────────────────────────────────────── */
function TimelineCell({ acts, deal, onOpenOC }) {
  return (
    <div className="opd-tl-cell">
      <div className="opd-tl-line" />
      {acts.map((act, i) => (
        <ActivityDot
          key={i}
          act={act}
          acts={acts}
          deal={deal}
          onOpenOC={onOpenOC}
        />
      ))}
    </div>
  );
}

/* ─────────────────────────────────────────────
   OFFCANVAS PANEL
───────────────────────────────────────────── */
function OffcanvasPanel({ deal, open, onClose }) {
  const [ocTab, setOcTab] = useState("activity");

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

        <div className="opd-oc-body" style={{ padding: "16px 20px 28px" }}>
          <div
            className={`opd-oc-pane${ocTab === "activity" ? " opd-oc-pane-active" : ""}`}
          >
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
                          <span className="opd-zz-type-dot" />
                          <span className="opd-zz-type-lbl">
                            {DL[item.type]}
                          </span>
                        </div>
                        <div className="opd-zz-date">{item.date}</div>
                        <div className="opd-zz-title">{item.title}</div>
                        <div className="opd-zz-body">
                          {item.body}Lorem ipsum is a dummy{" "}
                        </div>
                      </div>
                    </div>
                    <div className="opd-zz-node">
                      <div className="opd-zz-node-outer">
                        <div className="opd-zz-node-inner" />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div
            className={`opd-oc-pane${ocTab === "warnings" ? " opd-oc-pane-active" : ""}`}
          >
            {/* content removed */}
          </div>

          <div
            className={`opd-oc-pane${ocTab === "contacts" ? " opd-oc-pane-active" : ""}`}
          >
            {/* content removed */}
          </div>
        </div>
      </div>
    </>
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

export default function OpportunitiesDashboard() {
  const navigate = useNavigate();
  const [periodTab, setPeriodTab] = useState("week");
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState({
    Regions: "all",
    Industries: "all",
    Stage: "all",
    Products: "all",
  });
  // One of "open" | "pipeline" | "best_case" | "commit" | "won" | "loss" | null.
  // Driven by clicking a KPI card; sent as the `bucket` query-param to /api/opportunities.
  const [activeBucket, setActiveBucket] = useState(null);
  const [ddOptions, setDdOptions] = useState(DD_PLACEHOLDER);
  const [page, setPage] = useState(1);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [kpis, setKpis] = useState(SUMMARY_PLACEHOLDER);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [ocOpen, setOcOpen] = useState(false);
  const [ocDeal, setOcDeal] = useState(null);

  // Ensure this page always has its full stylesheet on first render.
  useLayoutEffect(() => {
    // Load font via <link> instead of @import to avoid blocking CSS
    const fontId = "opd-font-link";
    if (!document.getElementById(fontId)) {
      const link = document.createElement("link");
      link.id = fontId;
      link.rel = "stylesheet";
      link.href = "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap";
      document.head.appendChild(link);
    }

    const id = "opd-styles-main";
    let style = document.getElementById(id);
    if (!style) {
      style = document.createElement("style");
      style.id = id;
    }
    style.textContent = STYLES;
    // Re-append so this page's styles always win the cascade over any
    // stylesheet another page injected after navigating away and back.
    document.head.appendChild(style);
  }, []);

  // fetch dropdown options once on mount
  useEffect(() => {
    const ctrl = new AbortController();
    fetchFilterOptions(ctrl.signal)
      .then(({ regions, industries, stages, products }) => {
        setDdOptions({
          Regions: [
            { v: "all", l: "Regions" },
            ...(regions?.businessGroups || []).map((b, i) => ({
              v: b.id,
              l: b.label,
              c: colorAt(i),
            })),
          ],
          Industries: [
            { v: "all", l: "Industries" },
            ...(industries?.items || []).map((b, i) => ({
              v: b.code,
              l: b.label,
              c: colorAt(i + 3),
            })),
          ],
          Stage: [
            { v: "all", l: "Stage" },
            ...(stages?.items || []).map((s, i) => ({
              v: s.raw,
              l: s.label,
              c: colorAt(i + 5),
            })),
          ],
          Products: [
            { v: "all", l: "Products" },
            ...(products?.items || []).map((p, i) => ({
              v: p.id,
              l: p.label,
              c: colorAt(i + 1),
            })),
          ],
        });
      })
      .catch((err) => {
        if (err.name !== "AbortError")
          console.error("filters load failed", err);
      });
    return () => ctrl.abort();
  }, []);

  // any filter change resets pagination back to page 1
  const filterSignature = JSON.stringify({
    search,
    filters,
    periodTab,
    activeBucket,
  });
  useEffect(() => {
    setPage(1);
  }, [filterSignature]);

  // debounced fetch for the grid + KPI strip
  useEffect(() => {
    const ctrl = new AbortController();
    const params = {
      search: search.trim() || undefined,
      regions: filters.Regions !== "all" ? filters.Regions : undefined,
      industries: filters.Industries !== "all" ? filters.Industries : undefined,
      stages: filters.Stage !== "all" ? filters.Stage : undefined,
      products: filters.Products !== "all" ? filters.Products : undefined,
    };

    const handle = setTimeout(() => {
      setLoading(true);
      setError(null);
      Promise.all([
        // Only the grid is constrained by the clicked KPI card; the strip
        // keeps showing every bucket so the user can still see the others.
        fetchOpportunities(
          {
            ...params,
            bucket: activeBucket || undefined,
            page,
            pageSize: PAGE_SIZE,
            sortBy: "closeDate",
            sortOrder: "desc",
          },
          ctrl.signal,
        ),
        fetchKpiSummary(
          { ...params, comparePeriod: PERIOD_TO_API[periodTab] },
          ctrl.signal,
        ),
      ])
        .then(([opps, kpi]) => {
          console.log(opps);
          const ownerId = sessionStorage.getItem("ownerId");
          const filtered = ownerId
            ? (opps.items || []).filter(
                (item) => item.ownerId?.toLowerCase() === ownerId.toLowerCase(),
              )
            : (opps.items || []);
          setItems(filtered.map(mapOpportunityRow));
          setTotal(filtered.length);
          setTotalPages(Math.ceil(filtered.length / PAGE_SIZE) || 1);
          setKpis(mapKpiSummary(kpi));
        })
        .catch((err) => {
          if (err.name === "AbortError") return;
          console.error(err);
          setError(err.message || "Failed to load data");
          setItems([]);
          setKpis(SUMMARY_PLACEHOLDER);
          setTotal(0);
          setTotalPages(1);
        })
        .finally(() => setLoading(false));
    }, 250); // debounce typing in the search box
    return () => {
      clearTimeout(handle);
      ctrl.abort();
    };
  }, [
    search,
    filters.Regions,
    filters.Industries,
    filters.Stage,
    filters.Products,
    periodTab,
    page,
    activeBucket,
  ]);
  console.log(items);
  console.log(kpis);

  // Click a KPI card to toggle the bucket filter on/off.
  const handleBucketToggle = useCallback((bucket) => {
    if (!bucket) return;
    setActiveBucket((prev) => (prev === bucket ? null : bucket));
  }, []);

  const handleFilterChange = useCallback((key, val) => {
    setFilters((prev) => ({ ...prev, [key]: val }));
  }, []);

  const openOC = useCallback((deal) => {
    setOcDeal(deal);
    setOcOpen(true);
  }, []);

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
      "Next Action",
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
      normalizeForCsv(d.nextAction),
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

  const ticks = [
    { l: "90d", p: 0 },
    { l: "60d", p: 33.3 },
    { l: "30d", p: 66.6 },
    { l: "0d", p: 100 },
  ];

  const stageClass = (s) =>
    `opd-stage-badge opd-stage-${(s || "").replace(/\s+/g, "")}`;

  const pageButtons = useMemo(
    () => buildPageList(totalPages, page, 3),
    [totalPages, page],
  );

  return (
    <div className="opd-root">
      <div className="opd-page">
        {/* breadcrumb */}
        {/* <div className="opd-breadcrumb">
          <strong>Q2 FY2024</strong> · Week 10 of 12 · <span className="opd-breadcrumb-closure">Closure Phase</span>
        </div> */}
        <div className="opp-header">
          <h2 className="opd-h1">Opportunities</h2>

          {/* period tabs */}
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
          </div >       </div>
        {/* summary cards */}
        <div className="opd-summary-grid">
          {kpis.map((c, i) => {
            const isActive = c.bucket && c.bucket === activeBucket;
            const tipText = isActive
              ? `Showing ${c.label} only — click to clear`
              : `Filter grid to ${c.label}`;
            return (
              <div
                key={c.bucket || i}
               
                className={`opd-scard${isActive ? " opd-scard-active" : ""}`}
                role="button"
                tabIndex={0}
                aria-pressed={isActive}
                aria-label={tipText}
                data-tip={tipText}
                onClick={() => handleBucketToggle(c.bucket)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    handleBucketToggle(c.bucket);
                  }
                }}
              >
                <div className="opd-scard-head">
                  <div className="opd-scard-label">{c.label}</div>
                </div>
                <div className="opd-scard-body">
                  <div className="opd-scard-value">{c.value}</div>
                  <div className="opd-scard-meta">
                    <span className="opd-scard-count">{c.count}</span>
                    {c.delta ? <span className="opd-scard-sep">|</span> : null}
                    {c.delta ? (
                      <span
                        className={c.up ? "opd-delta-up" : "opd-delta-down"}
                      >
                        <img
                          src={c.up ? greenArrowIcon : redArrowIcon}
                          alt={c.up ? "up" : "down"}
                          className="opd-delta-icon"
                        />
                        {c.delta}
                      </span>
                    ) : null}
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
              placeholder="Search Opportunity"
            />
          </div>
          {["Regions", "Industries", "Stage", "Products"].map((key) => (
            <Dropdown
              key={key}
              filterKey={key}
              value={filters[key]}
              options={ddOptions[key]}
              onChange={handleFilterChange}
            />
          ))}
          {/* <button className="opd-btn-clear" onClick={clearFilters}>Clear filters</button> */}
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
                        <td
                          className="opd-td opd-name-cell"
                          style={{ cursor: "pointer" }}
                          onClick={() => navigate(`/opportunities/${d.id}`)}
                          data-tip={d.name}
                        >
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
                      <span
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 4,
                          marginLeft: 8,
                        }}
                      >
                        <svg
                          width="10"
                          height="10"
                          viewBox="0 0 10 10"
                          style={{ transform: "rotate(90deg)" }}
                        >
                          <path
                            d="M1 3l4 4 4-4"
                            stroke="#0F172A"
                            strokeWidth="1.5"
                            fill="none"
                            strokeLinecap="butt"
                          />
                        </svg>
                        <span
                          style={{
                            fontSize: 12,
                            fontWeight: 400,
                            color: "#0F172A",
                          }}
                        >
                          scroll
                        </span>
                        <svg
                          width="10"
                          height="10"
                          viewBox="0 0 10 10"
                          style={{ transform: "rotate(-90deg)" }}
                        >
                          <path
                            d="M1 3l4 4 4-4"
                            stroke="#0F172A"
                            strokeWidth="1.5"
                            fill="none"
                            strokeLinecap="butt"
                          />
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
                    <th className="opd-th" style={{ minWidth: 160 }}>
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
                          style={{ color: "#495057", fontSize: 14 }}
                        >
                          {d.comp}
                        </td>
                        <td
                          className="opd-td"
                          style={{ color: "var(--opd-text3)" }}
                        >
                          {d.nextAction}
                        </td>
                        <td
                          className="opd-td"
                          style={{ fontWeight: 700, color: "#212529", fontSize: 13 }}
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
                const sizeMap = { crm: 8, email: 10, meeting: 16, multiple: 20 };
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
                <i className="bi bi-chevron-left" style={{ fontSize: 10 }}></i>
              </button>

              {pageButtons.map((p, i) =>
                p === "..." ? (
                  <span
                    key={`gap-${i}`}
                    style={{
                      fontSize: 12,
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
                  fontSize: 14,
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

      {/* Offcanvas */}
      <OffcanvasPanel
        deal={ocDeal}
        open={ocOpen}
        onClose={() => setOcOpen(false)}
      />
    </div>
  );
}
