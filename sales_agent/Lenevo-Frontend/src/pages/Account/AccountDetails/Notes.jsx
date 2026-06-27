export default function Notes() {
  const notes = [
    { title: "Q2 Strategy Meeting Notes", meta: "Created Jun 15, 2026 • Amit Ranjan" },
    { title: "Product Demo Feedback", meta: "Created Jun 10, 2026 • Priya Sharma" },
    { title: "Pricing Discussion Summary", meta: "Created Jun 5, 2026 • Rajesh Kumar" },
  ];

  return (
    <>
      <style>{`
        .notes-wrap { background: #fff; border: 1px solid #CBD5E1; border-radius: 12px; padding: 24px; }
        .notes-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
        .notes-header h2 { font-size: 20px; font-weight: 700; color: #0f172a; margin: 0; }
        .notes-btn { background: #1D4ED8; color: #fff; border: none; border-radius: 50px; padding: 8px 20px; font-size: 13px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; }
        .notes-btn:hover { background: #1F3A8A; }
        .note-editor { background: #fff; border: 1px solid #CBD5E1; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
        .note-textarea { width: 100%; min-height: 150px; resize: vertical; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px; font: inherit; color: #111827; margin-bottom: 16px; background: #F8FAFC; }
        .note-actions { display: flex; gap: 8px; }
        .note-clear { 
   display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #E2E8F0;
    color: #374151;
    border: none;
    font-size: 12.5px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    padding: 7px 16px;
    border-radius: 50px;
    cursor: pointer;
    transition: background .15s, border-color .15s;         }
        .note-clear:hover { background: #d1d5db; }
        .note-save { padding: 10px 20px; border: none; border-radius: 50px; background: #1D4ED8; color: #fff; font-weight: 600; cursor: pointer; }
        .note-save:hover { background: #1F3A8A; }
        .notes-list { display: flex; flex-direction: column; gap: 12px; }
        .note-card { background: #fff; border: 1px solid #CBD5E1; border-radius: 12px; padding: 16px; display: flex; align-items: center; gap: 12px; }
        .note-card:hover { box-shadow: 0 4px 6px rgba(0,0,0,.08); }
        .note-icon { width: 48px; height: 48px; border-radius: 8px; background: #f3f4f6; display: flex; align-items: center; justify-content: center; color: #6b7280; font-size: 20px; }
        .note-info { flex: 1; }
        .note-title { font-size: 16px; font-weight: 500; color: #111827; margin-bottom: 4px; }
        .note-meta { font-size: 14px; color: #6b7280; }
        .note-card-actions { display: flex; gap: 6px; }
        .note-card-btn { width: 36px; height: 36px; border: none; background: transparent; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; color: #0F172A; }
        .note-card-btn:hover { background: #f3f4f6; }
        .note-card-btn.delete { color: #ef4444; }
        .note-card-btn.delete:hover { background: #fef2f2; }
      `}</style>

      <div className="notes-wrap">
        <div className="notes-header">
          <h2>Notes</h2>
          <button className="notes-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            Add Note
          </button>
        </div>

        <div className="note-editor">
          <textarea className="note-textarea" placeholder="Write a new note..."></textarea>
          <div className="note-actions">
            <button className="note-clear">Clear All</button>
            <button className="note-save">Save</button>
          </div>
        </div>

        <div className="notes-list">
          {notes.map((note, i) => (
            <div className="note-card" key={i}>
              <div className="note-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                
              </div>
              <div className="note-info">
                <div className="note-title">{note.title}</div>
                <div className="note-meta">{note.meta}</div>
              </div>
              <div className="note-card-actions">
                <button className="note-card-btn" aria-label="Edit">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>
                </button>
                <button className="note-card-btn" aria-label="Download">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                </button>
                <button className="note-card-btn delete" aria-label="Delete">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
