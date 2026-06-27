import { useState } from "react";
import PersonalLibrary from "./PersonalLibrary";
import SharedLibrary from "./SharedLibrary";
import Templates from "./Templates";
import "../../styles/library.css";

export default function Library() {
  const [activeTab, setActiveTab] = useState("personal");
  const [showUploadForm, setShowUploadForm] = useState(false);

  const renderTabContent = () => {
    switch (activeTab) {
      case "shared":
        return <SharedLibrary />;
      case "templates":
        return <Templates />;
      default:
        return <PersonalLibrary showUploadForm={showUploadForm} setShowUploadForm={setShowUploadForm} />;
    }
  };

  return (
    <>
      <div className="dv-page">
        <div className="main">
          <div className="quarter-bar">
            Q2 FY2024 · Week 10 of 12 ·{" "}
            <span className="closure-badge">Closure Phase</span>
          </div>

          <div className="page-header">
            <div className="page-header-row">
              <span className="topbar-title">Library</span>
              <button
                onClick={() => { setActiveTab("personal"); setShowUploadForm(!showUploadForm); }}
                id="library-upload-btn"
                style={{
                  width: 36, height: 36, border: "none", background: "#1D4ED8", borderRadius: 8,
                  display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", color: "#fff", fontSize: 18, marginLeft: "auto",
                }}
                title="Upload"
              >
                <i className="ri-upload-2-line"></i>
              </button>
            </div>
            <div className="tabs-outer">
              <div className="tabs">
                <button
                  className={`tab ${activeTab === "personal" ? "active" : ""}`}
                  onClick={() => setActiveTab("personal")}
                >
                  Personal Library
                </button>
                <button
                  className={`tab ${activeTab === "shared" ? "active" : ""}`}
                  onClick={() => setActiveTab("shared")}
                >
                  Shared Library
                </button>
                <button
                  className={`tab ${activeTab === "templates" ? "active" : ""}`}
                  onClick={() => setActiveTab("templates")}
                >
                  Templates
                </button>
              </div>
            </div>
          </div>

          <div className="content">
            <div className="col-main">
              {renderTabContent()}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

