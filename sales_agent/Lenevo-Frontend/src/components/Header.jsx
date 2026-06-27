import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/header.css";
import lenovoLogo from "../assets/xlqLogo.png";
import { fetchFiscalPeriod } from "../api/client";

const getDayGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return "Morning";
  if (hour < 17) return "Afternoon";
  return "Evening";
};

function Header({ sidebarWidth }) {
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const [dayGreeting, setDayGreeting] = useState(getDayGreeting());
  const [fiscalPeriod, setFiscalPeriod] = useState(null);

  const inputRef = useRef(null);
  const containerRef = useRef(null);

  const handleSearchClick = () => {
    setIsOpen(true);

    setTimeout(() => {
      inputRef.current?.focus();
    }, 200);
  };

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  useEffect(() => {
    const tick = () => setDayGreeting(getDayGreeting());
    tick();
    const id = setInterval(tick, 60_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    fetchFiscalPeriod()
      .then(setFiscalPeriod)
      .catch(() => {});
  }, []);

  return (
    <header
      className="app-header"
      // marginLeft is no longer applied here — header is now full-width and the
      // sidebar starts below it (top: 64px in sidebar.css).
      // style={{ marginLeft: `${sidebarWidth}px` }}
    >
      <div className="container-fluid">
        <div className="row align-items-center h-100">
          <div className="col d-flex align-items-center gap-3 p-0">
            <div className="logo-text">
              <img
                src={lenovoLogo}
                alt="Lenovo Logo"
                className="logo-image"
              />
            </div>
            <p className="subtitle-text text-muted mb-0" style={{ marginLeft: '30px' }}>
              <span className="subtitle-greeting d-block" style={{ color: 'black' }}>
                {`Good ${dayGreeting}, Amit`}
              </span>
              <span className="subtitle-line mt-1">
                {fiscalPeriod ? (
                  <>
                    {fiscalPeriod.label} • Week {fiscalPeriod.weekOfQuarter} of {fiscalPeriod.totalWeeks} •{" "}
                    <span className="sd-priority-due-today">
                      <b>{fiscalPeriod.phase}</b>
                    </span>
                  </>
                ) : (
                  <span style={{ opacity: 0.4 }}>Loading…</span>
                )}
              </span>
            </p>
          </div>

          <div className="col-auto">
            <div className="d-flex align-items-center gap-2">

              {/* SEARCH */}
              <div className="search-container" ref={containerRef}>
                <input
                  ref={inputRef}
                  type="text"
                  placeholder="Search..."
                  className={`search-input ${isOpen ? "open" : ""}`}
                />

                <button
                  className="btn btn-icon"
                  onClick={handleSearchClick}
                  aria-label="Search"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="22"
                    height="22"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    viewBox="0 0 24 24"
                  >
                    <circle cx="11" cy="11" r="7" />
                    <line x1="16.5" y1="16.5" x2="21" y2="21" />
                  </svg>
                </button>
              </div>

              <button className="btn btn-icon">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  viewBox="0 0 24 24"
                >
                  <path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9" />
                  <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
                </svg>
              </button>

              <button className="btn btn-icon" onClick={() => navigate("/admin-settings")}>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="18"
                  height="18"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  viewBox="0 0 24 24"
                >
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
              </button>

      

              <div className="user-avatar">
                <span>A</span>
              </div>

            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;