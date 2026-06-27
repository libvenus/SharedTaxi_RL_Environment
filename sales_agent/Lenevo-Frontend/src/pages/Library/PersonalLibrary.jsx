import { useState, useRef } from "react";

export default function PersonalLibrary({ showUploadForm, setShowUploadForm }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const fileInputRef = useRef(null);

  const documents = [
    {
      id: 1,
      name: "Q2 Sales Proposal",
      source: "Google Drive",
      updatedOn: "2026-06-18",
      badge: "Document",
    },
    {
      id: 2,
      name: "Product Comparison Sheet",
      source: "OneDrive",
      updatedOn: "2026-06-15",
      badge: "Document",
    },
    {
      id: 3,
      name: "Client Meeting Notes",
      source: "Local Upload",
      updatedOn: "2026-06-12",
      badge: "Document",
    },
    {
      id: 4,
      name: "Pricing Guidelines 2026",
      source: "SharePoint",
      updatedOn: "2026-06-10",
      badge: "Document",
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Upload Form */}
      {showUploadForm && (
        <div className="ct-new-form">
          <div className="ct-form-header">
            <div className="ct-form-title">Upload Asset</div>
            <button className="ct-form-close" onClick={() => setShowUploadForm(false)}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div className="ct-fields">
            <div className="ct-field">
              <label>Asset Name</label>
              <input type="text" placeholder="Enter asset name" />
            </div>
            <div className="ct-field">
              <label>Type</label>
              <select>
                <option value="">Select</option>
                <option value="document">Document</option>
                <option value="presentation">Presentation</option>
                <option value="spreadsheet">Spreadsheet</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="ct-field">
              <label>Category</label>
              <select>
                <option value="">Select</option>
                <option value="sales">Sales</option>
                <option value="marketing">Marketing</option>
                <option value="operations">Operations</option>
                <option value="finance">Finance</option>
              </select>
            </div>
            <div className="ct-field">
              <label>Subcategory</label>
              <input type="text" placeholder="Enter subcategory" />
            </div>
          </div>
          <div
            className={`library-dropzone${dragActive ? " library-dropzone-active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files?.[0]) setUploadedFile(e.dataTransfer.files[0]);
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              onChange={(e) => { if (e.target.files?.[0]) setUploadedFile(e.target.files[0]); }}
            />
            <i className="ri-upload-cloud-2-line" style={{ fontSize: 32, color: "#94a3b8" }}></i>
            <div style={{ fontSize: 14, color: "#475569", fontWeight: 500 }}>
              {uploadedFile ? uploadedFile.name : "Drag and Drop files here to upload"}
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8" }}>or click to browse</div>
          </div>
          <div className="ct-form-actions">
            <button className="ct-btn-cancel" onClick={() => { setShowUploadForm(false); setUploadedFile(null); }}>Clear</button>
            <button className="ct-btn-save">Upload</button>
          </div>
        </div>
      )}

      {/* Document Grid */}
      <div className="library-grid">
      {documents.map((doc) => (
        <div className="library-card" key={doc.id}>
          <div className="library-card-left">
            <div className="library-card-icon">
              <i className="ri-file-text-line"></i>
            </div>
            <div className="library-card-info">
              <div className="library-card-name">
                {doc.name}
              </div>
              <div className="library-card-meta">
                <span>Source: {doc.source}</span>
                <span>Updated on: {doc.updatedOn}</span>
              </div>
            </div>
          </div>
          <span className="library-card-badge">{doc.badge}</span>
          <div className="library-card-actions">
            <button className="library-btn-action" title="View">
              <i className="ri-eye-line"></i>
            </button>
            <button className="library-btn-action" title="Download">
              <i className="ri-download-line"></i>
            </button>
            <button className="library-btn-action library-btn-delete" title="Delete">
              <i className="ri-delete-bin-line"></i>
            </button>
          </div>
        </div>
      ))}
    </div>
    </div>
  );
}
