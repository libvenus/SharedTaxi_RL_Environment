import { useState } from "react";
import "../../styles/login.css";
import logo from "../../assets/xlqLogo.png";

const DUMMY_USER = "admin";
const DUMMY_PASS = "admin123";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username === DUMMY_USER && password === DUMMY_PASS) {
      setError("");
      sessionStorage.setItem("ownerId", "81aadfdb-1817-425c-a5b1-45f383f230ce");
      onLogin();
    } else {
      setError("Invalid username or password");
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        {/* Logo */}
        <div className="login-logo-block">
          <img src={logo} alt="Lenovo xIQ" className="login-logo-img" />
          <div className="login-logo-subtitle">Lenovo Sales Central</div>
        </div>

        {/* Form */}
        <div className="login-form-title">Sign In</div>

        {error && <div style={{ color: "#ef4444", fontSize: "13px", marginBottom: "10px", textAlign: "center" }}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="login-field">
            <label htmlFor="username">Username</label>
            <div className="login-input-wrap">
              <span className="login-input-icon">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="#475569" strokeWidth="1.2">
                  <circle cx="7.5" cy="4.5" r="2.5" />
                  <path d="M2.5 13c0-2.5 2.2-4 5-4s5 1.5 5 4" strokeLinecap="round" />
                </svg>
              </span>
              <input
                type="text"
                id="username"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="login-field">
            <label htmlFor="password">Password</label>
            <div className="login-input-wrap">
              <span className="login-input-icon">
                <i className="bi bi-key" style={{ fontSize: 16, transform: "rotate(135deg)" }}></i>
              </span>
              <input
                type="password"
                id="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="login-actions">
            <button type="submit" className="login-btn-signin">
              Sign In
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path
                  d="M6 3H3a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h3M10 11l3-3-3-3M13 8H6"
                  stroke="#fff"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
            <a href="#" className="login-reset-link">
              Reset Password
            </a>
          </div>
        </form>
      </div>
    </div>
  );
}
