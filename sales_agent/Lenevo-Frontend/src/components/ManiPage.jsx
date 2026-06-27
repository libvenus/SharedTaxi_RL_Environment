import PageTitle from './PageTitle';
import DashboardCard from '../components/DashboardCard';
import StatWidget from '../components/StatWidget';
import DataTable from '../components/DataTable';
import '../styles/dashboard.css';

function Dashboard() {
  const actionCards = [
    {
      company: 'TechWave Solutions',
      badge: 'Due Today',
      badgeClass: 'badge-due-today',
      description: 'Re-engage Arjun Shah — 18 days without response'
    },
    {
      company: 'Infosys Limited',
      badge: 'Due Soon',
      badgeClass: 'badge-due-soon',
      description: 'Confirm staggered 3-month delivery feasibility with Lenovo logistics'
    }
  ];

  const accounts = [
    { name: 'Acme Corp Global', status: 'At Risk', statusColor: 'danger' },
    { name: 'TechStart Health', status: 'Needs Attention', statusColor: 'warning' },
    { name: 'Modern Logistics', status: 'Opportunity', statusColor: 'primary' }
  ];

  return (


//     <div class="sd-wrapper">
//   <div class="row g-4">
 
//     <div class="col-12 col-lg-7">
 
//       <div class="sd-priority-panel">
//         <div class="sd-priority-header">
//           <h2 class="sd-priority-title">
//             Priority Actions
//             <span class="sd-priority-meta">27 total actions · <span class="sd-priority-due-today">3 due today</span></span>
//           </h2>
//           <a href="#" class="sd-view-all-link">
//             View All
//             <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
//               <path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
//             </svg>
//           </a>
//         </div>
 
//         <div class="sd-actions-grid">
 
//           <div class="sd-action-card">
//             <div class="sd-action-card-header">
//               <span class="sd-action-company">TechWave Solutions</span>
//               <span class="sd-badge-due-today">
//                 <span class="sd-badge-icon">🔴</span> Due Today
//               </span>
//             </div>
//             <p class="sd-action-desc">Re-engage Arjun Shah — 18 days without response</p>
//             <button class="sd-action-btn">Action</button>
//           </div>
 
//           <div class="sd-action-card">
//             <div class="sd-action-card-header">
//               <span class="sd-action-company">Infosys Limited</span>
//               <span class="sd-badge-due-soon">
//                 <span class="sd-badge-icon">⚠️</span> Due Soon
//               </span>
//             </div>
//             <p class="sd-action-desc">Confirm staggered 3-month delivery feasibility with Lenovo logistics</p>
//             <button class="sd-action-btn">Action</button>
//           </div>
 
//           <div class="sd-action-card">
//             <div class="sd-action-card-header">
//               <span class="sd-action-company">TechWave Solutions</span>
//               <span class="sd-badge-due-today">
//                 <span class="sd-badge-icon">🔴</span> Due Today
//               </span>
//             </div>
//             <p class="sd-action-desc">Re-engage Arjun Shah — 18 days without response</p>
//             <button class="sd-action-btn">Action</button>
//           </div>
 
//           <div class="sd-action-card">
//             <div class="sd-action-card-header">
//               <span class="sd-action-company">Infosys Limited</span>
//               <span class="sd-badge-due-soon">
//                 <span class="sd-badge-icon">⚠️</span> Due Soon
//               </span>
//             </div>
//             <p class="sd-action-desc">Confirm staggered 3-month delivery feasibility with Lenovo logistics</p>
//             <button class="sd-action-btn">Action</button>
//           </div>
 
//         </div>
//       </div>
 
//       <div class="sd-accounts-panel">
//         <div class="sd-accounts-header">
//           <h2 class="sd-accounts-title">
//             Accounts Needing Attention
//             <span class="sd-accounts-meta">11 total accounts · <span class="sd-accounts-risk">1 at risk</span></span>
//           </h2>
//           <a href="#" class="sd-view-all-link">
//             View All
//             <svg class="sd-chevron-svg" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
//               <path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
//             </svg>
//           </a>
//         </div>
 
//         <div class="sd-account-row">
//           <div class="sd-account-row-header">
//             <span class="sd-account-name">Acme Corp Global</span>
//             <span class="sd-status-at-risk">
//               At Risk <span class="sd-status-dot-red"></span>
//             </span>
//           </div>
//           <div class="sd-account-task-row">
//             <div class="sd-account-task-body">
//               <div class="sd-account-task-title">ThinkPad Fleet Refresh</div>
//               <div class="sd-account-task-desc">Proposal revision needed by Jun 17 — scope grew to 1,400 units + DaaS</div>
//               <div class="sd-account-task-ellipsis">...</div>
//             </div>
//             <svg class="sd-account-chevron sd-chevron-svg" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
//               <path d="M6 3l5 5-5 5" stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
//             </svg>
//           </div>
//         </div>
 
//         <div class="sd-account-row">
//           <div class="sd-account-row-header">
//             <span class="sd-account-name">TechStart Health</span>
//             <span class="sd-status-needs-attention">
//               Needs Attention <span class="sd-status-dot-amber"></span>
//             </span>
//           </div>
//           <div class="sd-account-task-row">
//             <div class="sd-account-task-body">
//               <div class="sd-account-task-title">ThinkPad Fleet Refresh</div>
//               <div class="sd-account-task-desc">Proposal revision needed by Jun 17 — scope grew to 1,400 units + DaaS</div>
//               <div class="sd-account-task-ellipsis">...</div>
//             </div>
//             <svg class="sd-account-chevron sd-chevron-svg" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
//               <path d="M6 3l5 5-5 5" stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
//             </svg>
//           </div>
//         </div>
 
//         <div class="sd-account-row">
//           <div class="sd-account-row-header">
//             <span class="sd-account-name">Modern Logistics</span>
//             <span class="sd-status-opportunity">
//               Opportunity <span class="sd-status-dot-blue"></span>
//             </span>
//           </div>
//         </div>
 
//       </div>
 
//     </div>
 
//     <div class="col-12 col-lg-5">
 
//       <div class="sd-q3-panel">
//         <div class="sd-q3-header">
//           <h2 class="sd-q3-title">Q3 Pulse</h2>
//           <span class="sd-q3-days-left">27 days left</span>
//         </div>
 
//         <div class="sd-q3-metric">
//           <div class="sd-q3-metric-header">
//             <span class="sd-q3-metric-label">Quota Attainment</span>
//             <span class="sd-q3-metric-value">60%</span>
//           </div>
//           <div class="sd-progress-track">
//             <div class="sd-progress-fill-blue" style="width: 60%;"></div>
//           </div>
//         </div>
 
//         <div class="sd-q3-metric">
//           <div class="sd-q3-metric-header">
//             <span class="sd-q3-metric-label">Pipeline Coverage</span>
//             <span class="sd-q3-metric-value">2.4x</span>
//           </div>
//           <div class="sd-progress-track">
//             <div class="sd-progress-fill-yellow" style="width: 75%;"></div>
//           </div>
//         </div>
//       </div>
 
//       <div class="sd-meetings-panel">
//         <div class="sd-meetings-header">
//           <h2 class="sd-meetings-title">Today's Meetings</h2>
//           <a href="#" class="sd-view-all-link">
//             View All
//             <svg class="sd-chevron-svg" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
//               <path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
//             </svg>
//           </a>
//         </div>
 
//         <div class="sd-meeting-card">
//           <div class="sd-meeting-card-header">
//             <div class="sd-meeting-platform">
//               <div class="sd-platform-icon sd-platform-webex">
//                 <svg class="sd-webex-svg" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
//                   <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/>
//                 </svg>
//               </div>
//               Internal · Webex
//             </div>
//             <span class="sd-meeting-time">14:00 - 15:00</span>
//           </div>
//           <div class="sd-meeting-title">Demo: Acor Integration</div>
//           <div class="sd-meeting-avatars">
//             <div class="sd-avatar-group">
//               <div class="sd-avatar sd-avatar-initials-ra">RA</div>
//               <div class="sd-avatar-face"></div>
//               <div class="sd-avatar-face" style="background: linear-gradient(135deg,#d1a87a 40%,#c08040 100%);"></div>
//             </div>
//           </div>
//           <div class="sd-meeting-actions">
//             <button class="sd-btn-prep">
//               <svg class="sd-prep-icon" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
//                 <path d="M10 2a6 6 0 00-3.9 10.6l.9.9V15a1 1 0 001 1h4a1 1 0 001-1v-1.5l.9-.9A6 6 0 0010 2zm2 13h-4v1h4v-1z"/>
//               </svg>
//               Prep
//             </button>
//             <button class="sd-btn-join-primary">Join</button>
//           </div>
//         </div>
 
//         <div class="sd-meeting-card">
//           <div class="sd-meeting-card-header">
//             <div class="sd-meeting-platform">
//               <div class="sd-platform-icon sd-platform-meet">
//                 <svg class="sd-meet-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
//                   <path d="M17 10.5V7a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h12a1 1 0 001-1v-3.5l4 4v-11l-4 4z" fill="#34A853"/>
//                   <rect x="3" y="7" width="14" height="10" rx="1" fill="none" stroke="#4285F4" stroke-width="1.5"/>
//                   <path d="M3 7h14v10H3z" fill="#4285F4" opacity=".1"/>
//                 </svg>
//               </div>
//               External · Meet
//             </div>
//             <span class="sd-meeting-time">15:45 - 16:15</span>
//           </div>
//           <div class="sd-meeting-title">Q3 Financial Review</div>
//           <div class="sd-meeting-avatars">
//             <div class="sd-avatar-group">
//               <div class="sd-avatar sd-avatar-initials-lm">LM</div>
//               <div class="sd-avatar-face" style="background: linear-gradient(135deg,#c9a884 40%,#a07040 100%);"></div>
//               <div class="sd-avatar-face" style="background: linear-gradient(135deg,#f0d0b0 40%,#d4a070 100%);"></div>
//             </div>
//           </div>
//           <div class="sd-meeting-actions">
//             <button class="sd-btn-prep">
//               <svg class="sd-prep-icon" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
//                 <path d="M10 2a6 6 0 00-3.9 10.6l.9.9V15a1 1 0 001 1h4a1 1 0 001-1v-1.5l.9-.9A6 6 0 0010 2zm2 13h-4v1h4v-1z"/>
//               </svg>
//               Prep
//             </button>
//             <button class="sd-btn-join-ghost">Join</button>
//           </div>
//         </div>
 
//         <div class="sd-meeting-card">
//           <div class="sd-meeting-card-header">
//             <div class="sd-meeting-platform">
//               <div class="sd-platform-icon sd-platform-meet">
//                 <svg class="sd-meet-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
//                   <path d="M17 10.5V7a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h12a1 1 0 001-1v-3.5l4 4v-11l-4 4z" fill="#34A853"/>
//                   <rect x="3" y="7" width="14" height="10" rx="1" fill="none" stroke="#4285F4" stroke-width="1.5"/>
//                   <path d="M3 7h14v10H3z" fill="#4285F4" opacity=".1"/>
//                 </svg>
//               </div>
//               External · Meet
//             </div>
//             <span class="sd-meeting-time">16:30 - 17:15</span>
//           </div>
//           <div class="sd-meeting-title">Contract Check-In</div>
//           <div class="sd-meeting-avatars">
//             <div class="sd-avatar-group">
//               <div class="sd-avatar sd-avatar-initials-mp">MP</div>
//               <div class="sd-avatar-face" style="background: linear-gradient(135deg,#e8c4a0 40%,#c89060 100%);"></div>
//               <div class="sd-avatar-face" style="background: linear-gradient(135deg,#d4b896 40%,#b88860 100%);"></div>
//             </div>
//           </div>
//           <div class="sd-meeting-actions">
//             <button class="sd-btn-prep">
//               <svg class="sd-prep-icon" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
//                 <path d="M10 2a6 6 0 00-3.9 10.6l.9.9V15a1 1 0 001 1h4a1 1 0 001-1v-1.5l.9-.9A6 6 0 0010 2zm2 13h-4v1h4v-1z"/>
//               </svg>
//               Prep
//             </button>
//             <button class="sd-btn-join-ghost">Join</button>
//           </div>
//         </div>
 
//       </div>
 
//     </div>
 
//   </div>
// </div>
    <div className="container-fluid py-4">
    

      <div className="row g-4">
        <div className="col-lg-8">
          <DashboardCard className="mb-4">
            <div className="d-flex justify-content-between align-items-center mb-4">
              <div>
                <h5 className="section-title mb-1">Priority Actions</h5>
                <span className="text-muted small">
                  27 total actions · <span className="text-danger fw-semibold">3 due today</span>
                </span>
              </div>
              <a href="#" className="view-all-link">
                View All <i className="bi bi-chevron-right"></i>
              </a>
            </div>

            <div className="row g-3">
              {actionCards.map((card, index) => (
                <div key={index} className="col-md-6">
                  <div className="action-card">
                    <div className="d-flex justify-content-between align-items-start mb-2">
                      <h6 className="action-company mb-0">{card.company}</h6>
                      <span className={`badge ${card.badgeClass}`}>
                        <i className="bi bi-exclamation-circle me-1"></i>
                        {card.badge}
                      </span>
                    </div>
                    <p className="action-description">{card.description}</p>
                    <button className="btn btn-primary btn-sm w-100">Action</button>
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>

          {/* Accounts Needing Attention */}
          <DashboardCard>
            <div className="d-flex justify-content-between align-items-center mb-4">
              <div>
                <h5 className="section-title mb-1">Accounts Needing Attention</h5>
                <span className="text-muted small">
                  11 total accounts - <span className="text-danger fw-semibold">1 at risk</span>
                </span>
              </div>
              <a href="#" className="view-all-link">
                View All <i className="bi bi-chevron-right"></i>
              </a>
            </div>

            {accounts.map((account, index) => (
              <div key={index} className="account-card mb-3">
                <div className="account-header">
                  <h6 className="mb-0">{account.name}</h6>
                  <span className={`badge bg-${account.statusColor}`}>
                    {account.status}
                  </span>
                </div>
                <div className="account-body">
                  <div className="account-detail">
                    <strong>ThinkPad Fleet Refresh</strong>
                    <p className="mb-0 text-muted small">
                      Proposal revision needed by Jun 17 — scope grew to 1,400 units + DaaS
                    </p>
                  </div>
                  <button className="btn btn-link p-0">
                    <i className="bi bi-chevron-right"></i>
                  </button>
                </div>
              </div>
            ))}
          </DashboardCard>
        </div>

        {/* Right Column */}
        <div className="col-lg-4">
          {/* Q3 Pulse */}
          <DashboardCard className="mb-4">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h5 className="section-title mb-0">Q3 Pulse</h5>
              <span className="text-muted small">27 days left</span>
            </div>

            <div className="metric-item mb-3">
              <div className="d-flex justify-content-between mb-2">
                <span className="metric-label">Quota Attainment</span>
                <span className="metric-value">60%</span>
              </div>
              <div className="progress" style={{ height: '8px' }}>
                <div 
                  className="progress-bar bg-primary" 
                  style={{ width: '60%' }}
                  role="progressbar" 
                  aria-valuenow="60" 
                  aria-valuemin="0" 
                  aria-valuemax="100"
                ></div>
              </div>
            </div>

            <div className="metric-item">
              <div className="d-flex justify-content-between mb-2">
                <span className="metric-label">Pipeline Coverage</span>
                <span className="metric-value">2.4x</span>
              </div>
              <div className="progress" style={{ height: '8px' }}>
                <div 
                  className="progress-bar bg-warning" 
                  style={{ width: '80%' }}
                  role="progressbar" 
                  aria-valuenow="80" 
                  aria-valuemin="0" 
                  aria-valuemax="100"
                ></div>
              </div>
            </div>
          </DashboardCard>

          <DashboardCard>
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h5 className="section-title mb-0">Today's Meetings</h5>
              <a href="#" className="view-all-link small">View All</a>
            </div>

            <div className="meeting-card mb-3">
              <div className="d-flex align-items-start gap-2 mb-2">
                <i className="bi bi-camera-video-fill text-primary"></i>
                <div className="flex-grow-1">
                  <div className="d-flex justify-content-between">
                    <span className="meeting-type">Internal - Webex</span>
                    <span className="meeting-time">14:00 - 15:00</span>
                  </div>
                </div>
              </div>
              <h6 className="meeting-title">Demo: Acor Integration</h6>
              <div className="d-flex gap-2 mt-2">
                <button className="btn btn-outline-secondary btn-sm">
                  <i className="bi bi-lightbulb me-1"></i> Prep
                </button>
                <button className="btn btn-primary btn-sm">Join</button>
              </div>
            </div>

            <div className="meeting-card mb-3">
              <div className="d-flex align-items-start gap-2 mb-2">
                <i className="bi bi-google text-success"></i>
                <div className="flex-grow-1">
                  <div className="d-flex justify-content-between">
                    <span className="meeting-type">External - Meet</span>
                    <span className="meeting-time">15:45 - 16:15</span>
                  </div>
                </div>
              </div>
              <h6 className="meeting-title">Q3 Financial Review</h6>
              <div className="d-flex gap-2 mt-2">
                <button className="btn btn-outline-secondary btn-sm">
                  <i className="bi bi-lightbulb me-1"></i> Prep
                </button>
                <button className="btn btn-primary btn-sm">Join</button>
              </div>
            </div>

            <div className="meeting-card">
              <div className="d-flex align-items-start gap-2 mb-2">
                <i className="bi bi-google text-success"></i>
                <div className="flex-grow-1">
                  <div className="d-flex justify-content-between">
                    <span className="meeting-type">External - Meet</span>
                    <span className="meeting-time">16:30 - 17:15</span>
                  </div>
                </div>
              </div>
              <h6 className="meeting-title">Contract Check-In</h6>
              <div className="d-flex gap-2 mt-2">
                <button className="btn btn-outline-secondary btn-sm">
                  <i className="bi bi-lightbulb me-1"></i> Prep
                </button>
                <button className="btn btn-primary btn-sm">Join</button>
              </div>
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;