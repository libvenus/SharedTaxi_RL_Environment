import { useEffect } from 'react';
import Header from '../components/Header';
import Sidebar from '../components/Sidebar';
import ChatWidget from '../components/ChatWidget';

function MainLayout({ children, sidebarCollapsed, toggleSidebar }) {
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 992 && !sidebarCollapsed) {
        // Auto-collapse on tablet/mobile
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarCollapsed]);

  const sidebarWidth = sidebarCollapsed ? 70 : 240;

  return (
    <div className="dashboard-wrapper">
      <Header 
        sidebarWidth={sidebarWidth} 
        toggleSidebar={toggleSidebar}
        sidebarCollapsed={sidebarCollapsed}
      />
      <Sidebar 
        collapsed={sidebarCollapsed} 
        toggleSidebar={toggleSidebar}
      />
      <main 
        className="main-content"
        style={{ marginLeft: `${sidebarWidth}px` }}
      >
        {children}
      </main>
      <ChatWidget />
    </div>
  );
}

export default MainLayout;
