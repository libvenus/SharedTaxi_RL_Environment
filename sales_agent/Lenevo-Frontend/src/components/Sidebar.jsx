import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { menuItems } from '../data/MenuData';
import '../styles/sidebar.css';
import '../styles/header.css';
import lenovoLogo from "../assets/xlqLogo.png";

function Sidebar({ collapsed, toggleSidebar }) {
  const [executeExpanded, setExecuteExpanded] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (location.pathname.startsWith('/execute')) {
      setExecuteExpanded(true);
    }
  }, [location.pathname]);

  return (
    <aside className={`app-sidebar ${collapsed ? 'collapsed' : 'expanded'}`}>
      {/* Sidebar header (Lenovo logo + subtitle) was moved into <Header /> alongside
          the greeting. Kept here as a reference in case we want to bring it back. */}
      {/* <div className="sidebar-header">
        <div className="sidebar-logo">
          {!collapsed && (
            <>
              <div className="logo-text">
  <img
        src={lenovoLogo}
        alt="Lenovo Logo"
        className="logo-image"
      />              </div>
              <div className="logo-subtitle">Lenovo Sales Central</div>
            </>
          )}
          {collapsed && <div className="logo-icon">L</div>}
        </div>
      </div> */}

      <nav className="sidebar-nav">
        <ul className="nav flex-column">
          {menuItems.map((item) => (
            <li key={item.id} className="nav-item">
              {item.title === 'Execute' && item.children && !collapsed ? (
                <>
                  <button
                    type="button"
                    className={`nav-link nav-link-toggle ${
                      location.pathname.startsWith('/execute') ? 'active' : ''
                    }`}
                    onClick={() => setExecuteExpanded((prev) => !prev)}
                  >
                    {typeof item.icon === "string" ? (
                      <i className={`bi ${item.icon}`} />
                    ) : (
                      item.icon
                    )}
                    <span className="sidebar-text">{item.title}</span>
                    <i
                      className={`bi bi-chevron-down sidebar-chevron ${
                        executeExpanded ? 'expanded' : ''
                      }`}
                    ></i>
                  </button>

                  {executeExpanded && (
                    <ul className="sidebar-submenu">
                      {item.children.map((child) => (
                        <li key={child.id} className="nav-item">
                          <NavLink
                            to={child.path}
                            className={({ isActive }) => `nav-link sidebar-submenu-link ${isActive ? 'active' : ''}`}
                          >
                            <span className="sidebar-text">{child.title}</span>
                          </NavLink>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              ) : (
                <NavLink
                  to={item.path}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                  title={collapsed ? item.title : ''}
                >
                  {typeof item.icon === "string" ? (
                    <i className={`bi ${item.icon}`} />
                  ) : (
                    item.icon
                  )}
                  {!collapsed && <span className="sidebar-text">{item.title}</span>}
                </NavLink>
              )}
            </li>
          ))}

        </ul>
      </nav>

      <div className="sidebar-footer">
        <NavLink
          to="/event-spline"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          title={collapsed ? 'Admin' : ''}
        >
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
            className="sidebar-icon"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
          {!collapsed && <span className="sidebar-text">Admin</span>}
        </NavLink>
      </div>
    </aside>
  );
}

export default Sidebar;