import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import MainLayout from "./layouts/MainLayout";
import Dashboard from "./pages/Dashboard";
import Opportunities from "./pages/Opportunities";
import Execute from "./pages/Execute";
import Detailedview from "./pages/Detailedview";
import Account from "./pages/Account";
import AccountDetails from "./pages/Account/AccountDetails";
import MeetingDetails from "./pages/Execute/MeetingDetails";
import KeyPoints from "./pages/Execute/PostMeetingDetails/KeyPoints";
import ActivityTimeline from "./pages/ActivityTimeline";
import AddMeeting from "./pages/Execute/AddMeeting";
import Settings from "./pages/Settings";
import EventSpline from "./pages/Admin/EventSpline";
import SalesOperatingModel from "./pages/Admin/SalesOperatingModel";
import Login from "./pages/Login";
import OutreachEmail from "./pages/Todo/OutreachEmail";
import Todo from "./pages/Todo/index";
import Library from "./pages/Library";
import AdminSettings from "./pages/Settings/AdminSettings";
import QuarterPulse from "./pages/Settings/QuarterPulse";
import Commercial from "./pages/Commercial";
import Activities from "./pages/Activities";
function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => sessionStorage.getItem("isAuthenticated") === "true",
  );

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  const handleLogin = () => {
    sessionStorage.setItem("isAuthenticated", "true");
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    sessionStorage.removeItem("isAuthenticated");
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return (
      <Router>
        <Routes>
          <Route path="*" element={<Login onLogin={handleLogin} />} />
        </Routes>
      </Router>
    );
  }

  return (
    <Router>
      <MainLayout
        sidebarCollapsed={sidebarCollapsed}
        toggleSidebar={toggleSidebar}
      >
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/opportunities" element={<Opportunities />} />
          <Route path="/opportunities/:id" element={<Detailedview />} />
          <Route path="/detailedview" element={<Detailedview />} />
          <Route path="/detailedviewpage" element={<Detailedview />} />
          <Route path="/execute" element={<Execute />} />
          <Route path="/execute/plan" element={<Execute />} />
          <Route path="/execute/approvals" element={<Execute />} />
          <Route path="/execute/milestones" element={<Execute />} />
          <Route path="/execute/reports" element={<Execute />} />
          <Route path="/execute/meeting/:id" element={<MeetingDetails />} />
          <Route path="/execute/postmeeting/:id" element={<KeyPoints />} />
          <Route path="/tasks" element={<Dashboard />} />
          <Route path="/accounts" element={<Account />} />
          <Route path="/accounts/:id" element={<AccountDetails />} />
          {/* <Route path="/accounts" element={<AccountDetails />} /> */}
          <Route path="/calendar" element={<Dashboard />} />
          <Route path="/activity" element={<Dashboard />} />
          <Route path="/insights" element={<Dashboard />} />
          <Route path="/admin" element={<Dashboard />} />
          <Route path="/activity-timeline" element={<ActivityTimeline />} />
          <Route path="/execute/add-meeting" element={<AddMeeting />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/event-spline" element={<EventSpline />} />
          <Route path="/library" element={<Library />} />
          <Route path="/admin-settings" element={<AdminSettings />} />
          <Route path="/quarter-pulse" element={<QuarterPulse />} />
          <Route path="/activities" element={<Activities />} />
          <Route
            path="/sales-operating-model"
            element={<SalesOperatingModel />}
          />
          <Route path="/commercial" element={<Commercial />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
          <Route path="/todo" element={<Todo />} />
          <Route path="/todo/:id" element={<OutreachEmail />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
