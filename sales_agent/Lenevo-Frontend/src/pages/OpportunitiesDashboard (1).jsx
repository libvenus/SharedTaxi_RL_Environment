import { useState, useRef, useCallback, useEffect } from "react";

// ─── Colour + label helpers ───────────────────────────────────────────────────
const DC = { crm:"#1dcb8a", email:"#1e3a5f", meeting:"#5bb8f5", multiple:"#1e5fa8" };
const DL = { crm:"CRM Updates", email:"Email", meeting:"Meeting", multiple:"Multiple Events" };
const hColor = p => p >= 70 ? "#22c55e" : p >= 40 ? "#f59e0b" : "#ef4444";

// ─── Static data ──────────────────────────────────────────────────────────────
const summaryCards = [
  { label:"Open Deals", value:"$9.75M", count:"12 Deals", delta:"+$2.7M",  up:true  },
  { label:"Pipeline",   value:"$2.67M", count:"19 Deals", delta:"+$230K",  up:true  },
  { label:"Best Case",  value:"$4.85M", count:"4 Deals",  delta:"-$200K",  up:false },
  { label:"Commit",     value:"$8.4M",  count:"6 Deals",  delta:"-$265K",  up:false },
  { label:"Won",        value:"$3.21M", count:"12 Deals", delta:"+$2.7M",  up:true  },
  { label:"Loss",       value:"$2.67M", count:"6 Deals",  delta:"-$265K",  up:false },
];

const allDeals = [
  { name:"ThinkPad Fleet Refresh",    company:"Infosys", risk:null, region:"APAC BG",     industry:"IT Services",           stage:"Discovery",     product:"ThinkPad",     comp:"Dell Technologies, HP Inc.",    value:"$420.0K", closeDate:"09/01/26", stageC:"#e8f0fe", stageT:"#3b5bdb", lastAct:"05/04/26", motion:"Net new",   health:80, acts:[{t:"email",d:45,date:"18 Apr 2026 · 10:00 AM",title:"Proposal Review Request",body:"Client requested revised pricing for 1,400 units instead of 1,200."},{t:"meeting",d:38,date:"25 Apr 2026 · 2:30 PM",title:"Demo Walkthrough",body:"Completed live product demo with IT decision team."},{t:"multiple",d:10,date:"23 May 2026 · 4:00 PM",title:"Multiple Events",body:"Contract review + pricing call + CRM update logged same day."}] },
  { name:"Server Expansion",          company:"Infosys", risk:null, region:"EMEA BG",     industry:"Financial Services",    stage:"Qualification", product:"ThinkSystem",  comp:"HP Inc., Panasonic",            value:"$111.0K", closeDate:"10/01/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"04/04/26", motion:"Net new",   health:80, acts:[{t:"crm",d:60,date:"04 Apr 2026 · 9:15 AM",title:"Stage Update",body:"Moved to Qualify. Next: technical validation."},{t:"crm",d:5,date:"29 May 2026 · 11:00 AM",title:"Note Added",body:"PO expected by end of June."}] },
  { name:"Server Expansion",          company:"Infosys", risk:null, region:"EMEA BG",     industry:"Industrial Manufacturing",stage:"Qualification",product:"ThinkSystem",  comp:"HP Inc., Panasonic",            value:"$111.0K", closeDate:"08/02/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"03/05/26", motion:"Net new",   health:80, acts:[{t:"crm",d:30,date:"04 May 2026 · 8:45 AM",title:"Owner Changed",body:"Opportunity reassigned to James O'Brien."}] },
  { name:"Server Expansion",          company:"Infosys", risk:null, region:"Americas BG", industry:"Automotive",            stage:"Qualification", product:"ThinkSystem",  comp:"SGI / HPE",                     value:"$111.0K", closeDate:"06/23/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"02/03/26", motion:"Expansion", health:75, acts:[{t:"crm",d:45,date:"19 Apr 2026 · 3:00 PM",title:"Budget Confirmed",body:"Procurement confirmed Q2 budget allocation."},{t:"crm",d:8,date:"26 May 2026 · 10:30 AM",title:"Follow-up Logged",body:"Chased for PO status. Response pending."}] },
  { name:"Server Expansion",          company:"Infosys", risk:null, region:"Americas BG", industry:"Automotive",            stage:"Qualification", product:"ThinkSystem",  comp:"SGI / HPE",                     value:"$111.0K", closeDate:"06/23/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"01/04/26", motion:"Expansion", health:65, acts:[{t:"crm",d:70,date:"24 Mar 2026 · 1:00 PM",title:"Competitor Flagged",body:"HP confirmed as active bid."},{t:"crm",d:6,date:"28 May 2026 · 9:00 AM",title:"Pricing Resubmitted",body:"Revised pricing sent."}] },
  { name:"Server Expansion",          company:"Infosys", risk:null, region:"EMEA BG",     industry:"Industrial Manufacturing",stage:"Qualification",product:"ThinkStation", comp:"Dell Technologies, Cisco",      value:"$111.0K", closeDate:"05/22/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"05/19/26", motion:"Expansion", health:60, acts:[{t:"meeting",d:75,date:"19 Mar 2026 · 11:00 AM",title:"Kickoff Meeting",body:"Initial engagement with procurement lead."},{t:"multiple",d:72,date:"22 Mar 2026 · 2:00 PM",title:"Multiple Events",body:"Contract + pricing + legal review logged."},{t:"meeting",d:55,date:"08 Apr 2026 · 10:00 AM",title:"Pricing Review Call",body:"Walk through final pricing with VP Finance."},{t:"crm",d:8,date:"25 May 2026 · 9:30 AM",title:"PO Submitted",body:"Purchase order submitted by procurement."}] },
  { name:"ThinkStation Workstations", company:"Infosys", risk:null, region:"APAC BG",     industry:"IT Services",           stage:"Qualification", product:"ThinkStation", comp:"Dell Technologies",             value:"$111.0K", closeDate:"09/01/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"04/19/26", motion:"Expansion", health:80, acts:[{t:"multiple",d:72,date:"22 Mar 2026 · 9:00 AM",title:"Multiple Events",body:"Workshop + BOM request + CRM update same day."},{t:"crm",d:28,date:"05 May 2026 · 2:00 PM",title:"Solution Area Updated",body:"Changed to Enterprise IT. DQR approved."}] },
  { name:"ThinkStation Workstations", company:"Infosys", risk:3,    region:"EMEA BG",     industry:"Financial Services",    stage:"Qualification", product:"ThinkStation", comp:"HP Inc., Cisco",                value:"$111.0K", closeDate:"08/02/26", stageC:"#CBD5E1", stageT:"#495057", lastAct:"03/20/26", motion:"Expansion", health:50, acts:[{t:"email",d:68,date:"24 Mar 2026 · 8:00 AM",title:"Budget Freeze Alert",body:"Client flagged potential budget freeze for Q2."},{t:"email",d:55,date:"06 Apr 2026 · 3:30 PM",title:"Re-engagement Email",body:"Sent ROI deck to CIO to unblock budget."},{t:"email",d:42,date:"19 Apr 2026 · 10:00 AM",title:"Response Received",body:"Client confirmed budget review in progress."},{t:"crm",d:10,date:"23 May 2026 · 11:00 AM",title:"Risk Score Updated",body:"Risk elevated to 3 pending competitor evaluation."}] },
  { name:"ThinkStation Workstations", company:"Infosys", risk:3,    region:"APAC BG",     industry:"IT Services",           stage:"Proposal",      product:"ThinkStation", comp:"Panasonic Toughbook",           value:"$58.0K",  closeDate:"06/23/26", stageC:"#fff3cd", stageT:"#856404", lastAct:"04/04/26", motion:"Renewal",   health:25, acts:[{t:"crm",d:40,date:"24 Apr 2026 · 9:00 AM",title:"Renewal Window Open",body:"60-day renewal window flagged in CRM."},{t:"crm",d:8,date:"25 May 2026 · 4:00 PM",title:"Proposal Sent",body:"3-year renewal proposal sent to procurement."}] },
];

// build offcanvas activity data from acts
const ocData = {};
allDeals.forEach((d, i) => {
  const ins  = d.acts.filter((_, j) => j % 2 === 0).map(a => ({ type:a.t, date:a.date, title:a.title, body:a.body }));
  const outs = d.acts.filter((_, j) => j % 2 !== 0).map(a => ({ type:a.t, date:a.date, title:a.title, body:a.body }));
  if (!outs.length) outs.push({ type:"meeting", date:d.acts[0].date, title:"Follow-up Call", body:"Scheduled follow-up to review progress and next steps." });
  while (ins.length  < 4) ins.push({ type:"crm",   date:"—", title:"No further activity", body:"No additional inbound activity recorded." });
  while (outs.length < 4) outs.push({ type:"email", date:"—", title:"Awaiting response",   body:"Follow-up email sent. Client response pending." });
  ocData[i] = { in: ins, out: outs };
});

// filter config
const ddConfig = {
  Regions:    [{ v:"all", l:"Regions" }, { v:"EMEA BG", l:"EMEA BG", c:"#1a56db" }, { v:"APAC BG", l:"APAC BG", c:"#0e9f6e" }, { v:"Americas BG", l:"Americas BG", c:"#e3a008" }],
  Industries: [{ v:"all", l:"Industries" }, { v:"Financial Services", l:"Financial Services", c:"#7c3aed" }, { v:"Industrial Manufacturing", l:"Industrial Manufacturing", c:"#0e9f6e" }, { v:"IT Services", l:"IT Services", c:"#e3a008" }, { v:"Automotive", l:"Automotive", c:"#e74c3c" }],
  Stage:      [{ v:"all", l:"Stage" }, { v:"Discovery", l:"Discovery", c:"#3b5bdb" }, { v:"Qualification", l:"Qualification", c:"#6c757d" }, { v:"Proposal", l:"Proposal", c:"#e3a008" }, { v:"Execute", l:"Execute", c:"#0e9f6e" }],
  Products:   [{ v:"all", l:"Products" }, { v:"ThinkPad", l:"ThinkPad", c:"#0e9f6e" }, { v:"ThinkStation", l:"ThinkStation", c:"#7c3aed" }, { v:"ThinkSystem", l:"ThinkSystem", c:"#e74c3c" }],
};

const thSt = { padding:"10px 13px", textAlign:"left", fontWeight:600, fontSize:12, color:"#212529", whiteSpace:"nowrap", background:"#fff", borderBottom:"1px solid #dee2e6" };
const tdSt = { padding:"0 13px", height:54, verticalAlign:"middle", fontSize:12 };

// ─── Health ring ──────────────────────────────────────────────────────────────
function HealthRing({ percent }) {
  const color = hColor(percent);
  const r = 15, cx = 19, cy = 19, circ = 2 * Math.PI * r, dash = (percent / 100) * circ;
  return (
    <div style={{ position:"relative", width:38, height:38, flexShrink:0 }}>
      <svg width="38" height="38" viewBox="0 0 38 38">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e9ecef" strokeWidth="3.5"/>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="3.5"
          strokeDasharray={`${dash.toFixed(2)} ${circ.toFixed(2)}`}
          strokeLinecap="round" transform="rotate(-90 19 19)"/>
      </svg>
      <span style={{ position:"absolute", top:"50%", left:"50%", transform:"translate(-50%,-50%)", fontSize:10, fontWeight:700, color }}>{percent}%</span>
    </div>
  );
}

// ─── Timeline dot with hover popup ───────────────────────────────────────────
function ActivityDot({ act, dealIdx, onDotClick }) {
  const [hover, setHover] = useState(false);
  const isM = act.t === "multiple";
  const sz  = isM ? 20 : 10;
  const pct = 100 - (act.d / 90) * 100;

  return (
    <div style={{ position:"absolute", left:`${pct}%`, transform:"translateX(-50%)", zIndex: hover ? 50 : 10 }}>
      <div
        style={{ width:sz, height:sz, borderRadius:"50%", background:DC[act.t], cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", color:"#fff", fontSize:9, fontWeight:700, transform:hover?"scale(1.4)":"scale(1)", transition:"transform .12s" }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        onClick={e => { e.stopPropagation(); setHover(false); onDotClick(dealIdx); }}
      >
        {isM ? "5" : ""}
      </div>
      {hover && (
        <div style={{ position:"absolute", bottom:"calc(100% + 10px)", left:"50%", transform:"translateX(-50%)", background:"#fff", border:"1px solid #CBD5E1", borderRadius:10, padding:"12px 14px", width:205, boxShadow:"0 6px 20px rgba(0,0,0,0.13)", pointerEvents:"none", zIndex:600, whiteSpace:"normal" }}>
          <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:7 }}>
            <span style={{ width:9, height:9, borderRadius:"50%", background:DC[act.t], flexShrink:0 }}/>
            <span style={{ fontSize:11, fontWeight:600, color:"#374151" }}>{DL[act.t]}</span>
          </div>
          <div style={{ fontSize:10, color:"#9ca3af", marginBottom:3 }}>{act.date || `${act.d}d ago`}</div>
          <div style={{ fontSize:12, fontWeight:700, color:"#111827", marginBottom:4, lineHeight:1.4 }}>{act.title}</div>
          <div style={{ fontSize:11, color:"#6b7280", lineHeight:1.5 }}>{act.body}</div>
          <div style={{ fontSize:10, color:"#1a56db", marginTop:6, fontStyle:"italic" }}>Click to view full timeline</div>
        </div>
      )}
    </div>
  );
}

function TimelineCell({ acts, dealIdx, onDotClick }) {
  return (
    <div style={{ position:"relative", width:340, height:28, display:"flex", alignItems:"center" }}>
      <div style={{ position:"absolute", left:0, right:0, top:"50%", height:1, background:"#e9ecef" }}/>
      {acts.map((act, i) => <ActivityDot key={i} act={act} dealIdx={dealIdx} onDotClick={onDotClick}/>)}
    </div>
  );
}

// ─── Dropdown with real filtering ────────────────────────────────────────────
function Dropdown({ filterKey, filterState, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const opts = ddConfig[filterKey] || [];
  const current = opts.find(o => o.v === filterState) || opts[0];

  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div ref={ref} style={{ position:"relative" }}>
      <div onClick={() => setOpen(o => !o)} style={{ display:"flex", alignItems:"center", gap:0, border:`1px solid ${open?"#1a56db":"#dee2e6"}`, borderRadius:open?"6px 6px 0 0":6, padding:"5px 10px", fontSize:12, color:"#495057", background:"#fff", cursor:"pointer", whiteSpace:"nowrap", minWidth:115, userSelect:"none" }}>
        <span style={{ flex:1 }}>{current.l}</span>
        <svg width="10" height="10" viewBox="0 0 10 10" style={{ flexShrink:0, marginLeft:10, transform:open?"rotate(180deg)":"none", transition:"transform .15s" }}><path d="M1 3l4 4 4-4" stroke="#0F172A" strokeWidth="1.5" fill="none" strokeLinecap="round"/></svg>
      </div>
      {open && (
        <div style={{ position:"absolute", top:"100%", left:0, right:0, background:"#fff", border:"1px solid #1a56db", borderTop:"none", borderRadius:"0 0 6px 6px", zIndex:400, overflow:"hidden", boxShadow:"0 4px 16px rgba(0,0,0,.1)" }}>
          {opts.map(opt => (
            <div key={opt.v} onClick={() => { onChange(filterKey, opt.v); setOpen(false); }}
              style={{ display:"flex", alignItems:"center", gap:7, padding:"7px 10px", fontSize:12, cursor:"pointer", color:filterState===opt.v?"#1a56db":"#495057", background:filterState===opt.v?"#e8f0fe":"transparent", fontWeight:filterState===opt.v?500:400 }}
              onMouseEnter={e => { if (filterState !== opt.v) e.currentTarget.style.background = "#f1f5ff"; }}
              onMouseLeave={e => { if (filterState !== opt.v) e.currentTarget.style.background = "transparent"; }}>
              {opt.c && <span style={{ width:8, height:8, borderRadius:"50%", background:opt.c, flexShrink:0 }}/>}
              {opt.l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Zigzag offcanvas card ────────────────────────────────────────────────────
const ArrowR = () => <svg width="11" height="11" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="1" y1="7" x2="13" y2="7"/><polyline points="8,2 13,7 8,12"/></svg>;
const ArrowL = () => <svg width="11" height="11" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="13" y1="7" x2="1" y2="7"/><polyline points="6,2 1,7 6,12"/></svg>;

function ZigzagItem({ item, index, isInbound }) {
  const isLeft = isInbound ? index % 2 === 0 : index % 2 !== 0;
  return (
    <div style={{ display:"flex", alignItems:"flex-start", flexDirection:isLeft?"row":"row-reverse", marginBottom:20, position:"relative", zIndex:1 }}>
      <div style={{ flex:1, maxWidth:"calc(50% - 28px)", paddingRight:isLeft?16:0, paddingLeft:isLeft?0:16 }}>
        <div style={{ background:"#fff", border:"1px solid #CBD5E1", borderRadius:12, borderTopRightRadius:isLeft?3:12, borderTopLeftRadius:isLeft?12:3, padding:"14px 15px" }}>
          <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:8 }}>
            <span style={{ width:10, height:10, borderRadius:"50%", background:DC[item.type], flexShrink:0 }}/>
            <span style={{ fontSize:11, fontWeight:600, color:"#374151" }}>{DL[item.type]}</span>
            <span style={{ marginLeft:"auto", display:"flex", alignItems:"center", gap:3, fontSize:10, color:"#9ca3af" }}>
              {isInbound ? <ArrowR/> : <ArrowL/>} {isInbound?"Inbound":"Outbound"}
            </span>
          </div>
          <div style={{ fontSize:11, color:"#9ca3af", marginBottom:4 }}>{item.date}</div>
          <div style={{ fontSize:13, fontWeight:700, color:"#111827", marginBottom:6, lineHeight:1.4 }}>{item.title}</div>
          <div style={{ fontSize:12, color:"#6b7280", lineHeight:1.65 }}>{item.body}</div>
        </div>
      </div>
      <div style={{ width:56, flexShrink:0, display:"flex", justifyContent:"center", paddingTop:4 }}>
        <div style={{ width:22, height:22, borderRadius:"50%", background:"#fff", border:"2px solid #d1d5db", display:"flex", alignItems:"center", justifyContent:"center", zIndex:2, position:"relative" }}>
          <div style={{ width:11, height:11, borderRadius:"50%", background:DC[item.type] }}/>
        </div>
      </div>
    </div>
  );
}

function OffcanvasPanel({ dealIdx, open, onClose }) {
  const [tab, setTab] = useState("in");
  const det     = ocData[dealIdx] || { in:[], out:[] };
  const isIn    = tab === "in";
  const items   = isIn ? det.in : det.out;

  return (
    <>
      {open && <div onClick={onClose} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.18)", zIndex:8000 }}/>}
      <div style={{ position:"fixed", top:0, right:0, height:"100%", width:520, maxWidth:"96vw", background:"#fff", zIndex:8100, display:"flex", flexDirection:"column", borderLeft:"1.5px solid #CBD5E1", transform:open?"translateX(0)":"translateX(100%)", transition:"transform .28s cubic-bezier(.4,0,.2,1)" }}>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"16px 20px 14px", borderBottom:"1px solid #CBD5E1", flexShrink:0 }}>
          <div>
            <span style={{ fontSize:16, fontWeight:700, color:"#111827" }}>Activity Timeline</span>
            <span style={{ fontSize:11, color:"#9ca3af", marginLeft:8 }}>· {allDeals[dealIdx]?.name}</span>
          </div>
          <button onClick={onClose} style={{ width:28, height:28, borderRadius:6, border:"1px solid #CBD5E1", background:"#fff", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14, color:"#6b7280", fontFamily:"inherit" }}>✕</button>
        </div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", borderBottom:"1px solid #CBD5E1", flexShrink:0 }}>
          {[["in", <><ArrowR/> Inbound</>], ["out", <>Outbound <ArrowL/></>]].map(([id, lbl], i) => (
            <button key={id} onClick={() => setTab(id)} style={{ padding:"11px 0", border:"none", borderRight:i===0?"1px solid #CBD5E1":"none", background:"transparent", fontSize:13, color:tab===id?"#1a56db":"#6b7280", cursor:"pointer", borderBottom:tab===id?"2px solid #1a56db":"2px solid transparent", marginBottom:-1, fontWeight:tab===id?500:400, fontFamily:"inherit", display:"flex", alignItems:"center", justifyContent:"center", gap:7 }}>{lbl}</button>
          ))}
        </div>
        <div style={{ flex:1, overflowY:"auto", padding:"20px 20px 28px" }}>
          <div style={{ position:"relative" }}>
            <div style={{ position:"absolute", left:"calc(50% - 0.5px)", top:0, bottom:0, width:1, background:"#CBD5E1", zIndex:0 }}/>
            {items.map((item, i) => <ZigzagItem key={i} item={item} index={i} isInbound={isIn}/>)}
          </div>
        </div>
      </div>
    </>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function OpportunitiesDashboard() {
  const [periodTab, setPeriodTab] = useState("week");
  const [search, setSearch]       = useState("");
  const [filters, setFilters]     = useState({ Regions:"all", Industries:"all", Stage:"all", Products:"all" });
  const [ocOpen, setOcOpen]       = useState(false);
  const [ocDeal, setOcDeal]       = useState(0);

  const handleFilterChange = useCallback((key, val) => {
    setFilters(prev => ({ ...prev, [key]: val }));
  }, []);

  const clearFilters = () => {
    setSearch("");
    setFilters({ Regions:"all", Industries:"all", Stage:"all", Products:"all" });
  };

  const filtered = allDeals.filter(d => {
    const q = search.toLowerCase().trim();
    if (q && !d.name.toLowerCase().includes(q) && !d.company.toLowerCase().includes(q) && !d.comp.toLowerCase().includes(q)) return false;
    if (filters.Regions    !== "all" && d.region   !== filters.Regions)    return false;
    if (filters.Industries  !== "all" && d.industry !== filters.Industries) return false;
    if (filters.Stage       !== "all" && d.stage    !== filters.Stage)      return false;
    if (filters.Products    !== "all" && d.product  !== filters.Products)   return false;
    return true;
  });

  const handleDotClick = useCallback(idx => { setOcDeal(idx); setOcOpen(true); }, []);
  const ticks = [{ l:"90d",p:0 },{ l:"60d",p:33.3 },{ l:"30d",p:66.6 },{ l:"0d",p:100 }];

  return (
    <div style={{ fontFamily:"'Inter','Segoe UI',sans-serif", background:"#f8f9fa", minHeight:"100vh", padding:"20px 22px", color:"#212529", fontSize:13 }}>

      <div style={{ fontSize:12, color:"#6c757d", marginBottom:12 }}>
        <strong style={{ color:"#212529" }}>Q2 FY2024</strong> · Week 10 of 12 · <span style={{ color:"#e74c3c", fontWeight:600 }}>Closure Phase</span>
      </div>
      <h2 style={{ fontSize:21, fontWeight:700, marginBottom:16 }}>Opportunities</h2>

      <div style={{ display:"flex", border:"1px solid #dee2e6", borderRadius:8, overflow:"hidden", width:"fit-content", background:"#fff", marginBottom:16 }}>
        {[["week","vs. Last Week"],["month","vs. Past Month"],["quarter","vs. Last Quarter"]].map(([id,lbl]) => (
          <button key={id} onClick={() => setPeriodTab(id)} style={{ padding:"7px 18px", border:"none", fontFamily:"inherit", background:periodTab===id?"#e8f0fe":"transparent", color:periodTab===id?"#1a56db":"#6c757d", fontWeight:periodTab===id?600:400, fontSize:12, cursor:"pointer", borderRight:id!=="quarter"?"1px solid #dee2e6":"none" }}>{lbl}</button>
        ))}
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(6,1fr)", gap:10, marginBottom:16 }}>
        {summaryCards.map((c,i) => (
          <div key={i} style={{ background:"#fff", border:"1px solid #dee2e6", borderRadius:10, padding:"12px 14px" }}>
            <div style={{ fontSize:11, color:"#6c757d", marginBottom:5 }}>{c.label}</div>
            <div style={{ fontSize:17, fontWeight:700, color:"#212529", marginBottom:3 }}>{c.value}</div>
            <div style={{ display:"flex", alignItems:"center", gap:6, fontSize:11 }}>
              <span style={{ color:"#6c757d" }}>{c.count}</span>
              <span style={{ color:"#dee2e6" }}>|</span>
              <span style={{ color:c.up?"#2ecc71":"#e74c3c", fontWeight:600 }}>{c.up?"▲":"▼"} {c.delta}</span>
            </div>
          </div>
        ))}
      </div>

      {/* filters */}
      <div style={{ display:"flex", gap:8, alignItems:"center", background:"#fff", border:"1px solid #dee2e6", borderRadius:10, padding:"10px 14px", marginBottom:14, flexWrap:"wrap" }}>
        <div style={{ display:"flex", alignItems:"center", gap:6, border:"1px solid #dee2e6", borderRadius:6, padding:"5px 10px", minWidth:160, flex:1, maxWidth:200 }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#adb5bd" strokeWidth="2.5"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search Opportunity" style={{ border:"none", outline:"none", fontSize:12, color:"#495057", background:"transparent", width:"100%", fontFamily:"inherit" }}/>
        </div>
        {["Regions","Industries","Stage","Products"].map(key => (
          <Dropdown key={key} filterKey={key} filterState={filters[key]} onChange={handleFilterChange}/>
        ))}
        <button onClick={clearFilters} style={{ padding:"5px 12px", borderRadius:6, border:"1px solid #dee2e6", background:"#fff", color:"#6c757d", fontSize:12, cursor:"pointer", fontFamily:"inherit", whiteSpace:"nowrap" }}>Clear filters</button>
        <button style={{ width:30, height:30, borderRadius:6, border:"1px solid #dee2e6", background:"#fff", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", color:"#6c757d", fontSize:16 }}>↓</button>
      </div>

      {/* table */}
      <div style={{ background:"#fff", border:"1px solid #dee2e6", borderRadius:10, overflow:"hidden" }}>
        <div style={{ display:"flex", overflow:"hidden" }}>
          <div style={{ flexShrink:0, borderRight:"2px solid #dee2e6", background:"#fff", zIndex:2 }}>
            <table style={{ borderCollapse:"collapse", tableLayout:"fixed", width:256 }}>
              <colgroup><col style={{ width:194 }}/><col style={{ width:62 }}/></colgroup>
              <thead>
                <tr><th style={thSt}>Name</th><th style={thSt}>Risk</th></tr>
                <tr><td colSpan={2} style={{ height:24, padding:0, borderBottom:"1px solid #CBD5E1", background:"#fff" }}/></tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr><td colSpan={2} style={{ padding:"28px 16px", textAlign:"center", color:"#9ca3af", fontSize:13 }}>No deals match</td></tr>
                ) : filtered.map((d,i) => (
                  <tr key={i} style={{ background:i%2===0?"#fff":"#fafafa", borderBottom:"1px solid #CBD5E1" }}>
                    <td style={{ ...tdSt, whiteSpace:"nowrap" }}>
                      <div style={{ fontWeight:500, color:"#212529", fontSize:13, lineHeight:1.3 }}>{d.name}</div>
                      <div style={{ fontSize:11, color:"#adb5bd" }}>{d.company}</div>
                    </td>
                    <td style={tdSt}>
                      {d.risk ? <span style={{ background:"#fff3cd", color:"#856404", fontSize:11, padding:"3px 7px", borderRadius:5, fontWeight:600, whiteSpace:"nowrap" }}>⚠ {d.risk}</span>
                               : <span style={{ color:"#adb5bd" }}>–</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ flex:1, overflowX:"auto" }}>
            <table style={{ borderCollapse:"collapse", width:"max-content", minWidth:"100%" }}>
              <thead>
                <tr>
                  <th style={{ ...thSt, minWidth:370, paddingBottom:0, verticalAlign:"top" }}>
                    <div style={{ fontWeight:600, fontSize:12, color:"#212529", marginBottom:5 }}>Activity Timeline <span style={{ fontSize:10, color:"#adb5bd", fontWeight:400, marginLeft:6 }}>‹ scroll ›</span></div>
                    <div style={{ position:"relative", width:340, height:16, marginBottom:2 }}>
                      {ticks.map(t => <span key={t.l} style={{ position:"absolute", left:`${(t.p/100)*340}px`, transform:"translateX(-50%)", fontSize:10, color:"#adb5bd", whiteSpace:"nowrap" }}>{t.l}</span>)}
                    </div>
                  </th>
                  <th style={{ ...thSt, minWidth:115 }}>Sale Motion</th>
                  <th style={{ ...thSt, minWidth:110 }}>Deal Health</th>
                  <th style={{ ...thSt, minWidth:90  }}>Value</th>
                  <th style={{ ...thSt, minWidth:100 }}>Close date</th>
                  <th style={{ ...thSt, minWidth:115 }}>Stage</th>
                  <th style={{ ...thSt, minWidth:190 }}>Competitors</th>
                  <th style={{ ...thSt, minWidth:110 }}>Last Activity</th>
                </tr>
                <tr><td colSpan={8} style={{ height:6, padding:0, borderBottom:"1px solid #CBD5E1", background:"#fff" }}/></tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr><td colSpan={8} style={{ padding:"28px 16px", textAlign:"center", color:"#9ca3af", fontSize:13 }}>No deals match your filters</td></tr>
                ) : filtered.map((d,i) => {
                  const di = allDeals.indexOf(d);
                  return (
                    <tr key={i} style={{ background:i%2===0?"#fff":"#fafafa", borderBottom:"1px solid #CBD5E1" }}>
                      <td style={{ ...tdSt, paddingTop:0, paddingBottom:0 }}>
                        <TimelineCell acts={d.acts} dealIdx={di} onDotClick={handleDotClick}/>
                      </td>
                      <td style={tdSt}><span style={{ background:"#CBD5E1", color:"#495057", padding:"3px 10px", borderRadius:5, fontSize:11, fontWeight:500, whiteSpace:"nowrap" }}>{d.motion}</span></td>
                      <td style={tdSt}><div style={{ display:"flex", alignItems:"center" }}><HealthRing percent={d.health}/></div></td>
                      <td style={{ ...tdSt, fontWeight:700, color:"#212529", fontSize:13 }}>{d.value}</td>
                      <td style={{ ...tdSt, color:"#6c757d" }}>{d.closeDate}</td>
                      <td style={tdSt}><span style={{ background:d.stageC, color:d.stageT, padding:"3px 10px", borderRadius:5, fontSize:11, fontWeight:500 }}>{d.stage}</span></td>
                      <td style={{ ...tdSt, color:"#495057", fontSize:11 }}>{d.comp}</td>
                      <td style={{ ...tdSt, color:"#6c757d" }}>{d.lastAct}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"10px 14px", borderTop:"1px solid #dee2e6", background:"#fafafa", flexWrap:"wrap", gap:8 }}>
          <span style={{ fontSize:11, color:"#6c757d" }}>Showing {filtered.length} of {allDeals.length} deals</span>
          <div style={{ display:"flex", alignItems:"center", gap:12, flexWrap:"wrap" }}>
            {Object.entries(DC).map(([type,color]) => (
              <span key={type} style={{ display:"flex", alignItems:"center", gap:5, fontSize:11, color:"#6c757d" }}>
                <span style={{ width:type==="multiple"?18:10, height:type==="multiple"?18:10, borderRadius:"50%", background:color, display:"inline-flex", alignItems:"center", justifyContent:"center", fontSize:8, color:"#fff", fontWeight:700, flexShrink:0 }}>{type==="multiple"?"5":""}</span>
                {DL[type]}
              </span>
            ))}
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:3 }}>
            <span style={{ fontSize:11, color:"#6c757d", marginRight:4 }}>Show</span>
            {[1,2,3].map(p => <button key={p} style={{ width:26, height:26, borderRadius:5, border:"1px solid #dee2e6", background:p===1?"#1a56db":"#fff", color:p===1?"#fff":"#495057", fontSize:11, cursor:"pointer", fontWeight:p===1?600:400, fontFamily:"inherit" }}>{p}</button>)}
            <span style={{ fontSize:11, color:"#6c757d", padding:"0 3px" }}>...</span>
            <button style={{ width:26, height:26, borderRadius:5, border:"1px solid #dee2e6", background:"#fff", color:"#495057", fontSize:11, cursor:"pointer", fontFamily:"inherit" }}>5</button>
            <button style={{ width:26, height:26, borderRadius:5, border:"1px solid #dee2e6", background:"#fff", color:"#495057", fontSize:11, cursor:"pointer", fontFamily:"inherit" }}>›</button>
            <span style={{ fontSize:11, color:"#6c757d", marginLeft:4 }}>5 pages</span>
          </div>
        </div>
      </div>

      <OffcanvasPanel dealIdx={ocDeal} open={ocOpen} onClose={() => setOcOpen(false)}/>
    </div>
  );
}
