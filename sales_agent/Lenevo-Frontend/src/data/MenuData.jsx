const iconProps = {
  xmlns: "http://www.w3.org/2000/svg",
  width: 18,
  height: 18,
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  viewBox: "0 0 16 16",
  className: "sidebar-icon",
};

export const menuItems = [
  {
    id: 1,
    title: "Dashboard",
    icon: (
      <svg {...iconProps}>
        <rect x="1" y="1" width="5" height="3" rx="1" />
        <rect x="1" y="7" width="5" height="8" rx="1" />
        <rect x="10" y="1" width="5" height="8" rx="1" />
        <rect x="10" y="12" width="5" height="3" rx="1" />
      </svg>
    ),
    path: "/dashboard",
    active: true,
  },
  {
    id: 2,
    title: "Opportunities",
    icon: (
      <svg {...iconProps}>
        <rect x="1.5" y="4.5" width="13" height="9" rx="1.5" />
        <path d="M5.5 4.5V3a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v1.5" />
        <line x1="1.5" y1="8.5" x2="14.5" y2="8.5" />
      </svg>
    ),
    path: "/opportunities",
    active: false,
  },
  {
    id: 5,
    title: "Accounts",
    icon: (
      <svg {...iconProps}>
        <rect x="3" y="1.5" width="8" height="13" rx="1" />
        <path d="M11 5h2.5a1 1 0 0 1 1 1v7.5a1 1 0 0 1-1 1H11" />
        <line x1="5.5" y1="4" x2="8.5" y2="4" />
        <line x1="5.5" y1="6.5" x2="8.5" y2="6.5" />
        <line x1="5.5" y1="9" x2="8.5" y2="9" />
      </svg>
    ),
    path: "/accounts",
    active: false,
  },
  {
    id: 3,
    title: "Calendar",
    icon: (
      <svg {...iconProps}>
        <rect x="2" y="3" width="12" height="11" rx="1.5" />
        <line x1="2" y1="6.5" x2="14" y2="6.5" />
        <line x1="5" y1="1.5" x2="5" y2="4" />
        <line x1="11" y1="1.5" x2="11" y2="4" />
      </svg>
    ),
    path: "/execute",
    active: false,
  },
  {
    id: 4,
    title: "To Do List",
    icon: (
      <svg {...iconProps}>
        <rect x="2" y="2" width="12" height="12" rx="2" />
        <path d="M5 8l2 2 4-4" />
      </svg>
    ),
    path: "/todo",
    active: false,
  },
  {
    id: 7,
    title: "Activity Timeline",
    icon: (
      <svg {...iconProps}>
        <circle cx="8" cy="8" r="6.5" />
        <path d="M8 4.5V8l2.5 1.5" />
      </svg>
    ),
    path: "/activities",
    active: false,
  },
  {
    id: 6,
    title: "Commercial",
    icon: (
      <svg {...iconProps}>
        <line x1="8" y1="1.5" x2="8" y2="14.5" />
        <path d="M11 4.2a3 3 0 0 0-3-1.2c-1.8 0-3 1-3 2.4 0 3.4 6 1.6 6 5 0 1.6-1.4 2.6-3 2.6a3.2 3.2 0 0 1-3-1.4" />
      </svg>
    ),
    path: "/commercial",
    active: false,
  },
  {
    id: 8,
    title: "Library",
    icon: (
      <svg {...iconProps}>
        <line x1="4.5" y1="2.5" x2="11.5" y2="2.5" />
        <line x1="3" y1="4.5" x2="13" y2="4.5" />
        <rect x="1.5" y="6.5" width="13" height="8" rx="1.5" />
      </svg>
    ),
    path: "/library",
    active: false,
  },
];

// export const adminMenuItem = {
//   id: 99,
//   title: "Admin",
//   icon: "bi-gear",
//   path: "/event-spline",
//   active: false,
// };
