import { useState } from 'react';
import PageTitle from '../components/PageTitle';
import DashboardCard from '../components/DashboardCard';

function Execute() {
  const [activeTab, setActiveTab] = useState('pending');

  const tasks = {
    pending: [
      {
        id: 1,
        title: 'Prepare Q3 sales presentation',
        description: 'Create comprehensive slides for quarterly review meeting',
        priority: 'High',
        dueDate: 'Jun 15, 2024',
        assignee: 'John Smith',
        tags: ['Presentation', 'Q3']
      },
      {
        id: 2,
        title: 'Follow up with Acme Corp',
        description: 'Call regarding proposal feedback and next steps',
        priority: 'Critical',
        dueDate: 'Jun 10, 2024',
        assignee: 'Sarah Johnson',
        tags: ['Follow-up', 'Acme Corp']
      },
      {
        id: 3,
        title: 'Update CRM records',
        description: 'Ensure all contact information is current',
        priority: 'Medium',
        dueDate: 'Jun 20, 2024',
        assignee: 'Mike Davis',
        tags: ['CRM', 'Data']
      },
      {
        id: 4,
        title: 'Schedule team training',
        description: 'Organize product training for new ThinkPad models',
        priority: 'Low',
        dueDate: 'Jun 25, 2024',
        assignee: 'Lisa Chen',
        tags: ['Training', 'Team']
      }
    ],
    inProgress: [
      {
        id: 5,
        title: 'Review contract terms',
        description: 'Legal review of Infrastructure Upgrade contract',
        priority: 'High',
        dueDate: 'Jun 12, 2024',
        assignee: 'Tom Wilson',
        tags: ['Legal', 'Contract']
      },
      {
        id: 6,
        title: 'Prepare demo environment',
        description: 'Set up cloud environment for client demo',
        priority: 'High',
        dueDate: 'Jun 14, 2024',
        assignee: 'Alex Brown',
        tags: ['Demo', 'Technical']
      }
    ],
    completed: [
      {
        id: 7,
        title: 'Submit expense report',
        description: 'Q2 business travel expenses',
        priority: 'Medium',
        dueDate: 'Jun 05, 2024',
        assignee: 'John Smith',
        tags: ['Finance', 'Expenses']
      },
      {
        id: 8,
        title: 'Client meeting notes',
        description: 'Document action items from TechWave meeting',
        priority: 'Medium',
        dueDate: 'Jun 08, 2024',
        assignee: 'Sarah Johnson',
        tags: ['Meeting', 'Notes']
      }
    ]
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'Critical': return 'danger';
      case 'High': return 'warning';
      case 'Medium': return 'info';
      case 'Low': return 'secondary';
      default: return 'secondary';
    }
  };

  const stats = [
    { label: 'Pending Tasks', value: tasks.pending.length, icon: 'bi-hourglass-split', color: 'warning' },
    { label: 'In Progress', value: tasks.inProgress.length, icon: 'bi-arrow-repeat', color: 'primary' },
    { label: 'Completed', value: tasks.completed.length, icon: 'bi-check-circle', color: 'success' },
    { label: 'Overdue', value: '2', icon: 'bi-exclamation-triangle', color: 'danger' }
  ];

  return (
    <div className="container-fluid py-4">
      <PageTitle 
        title="Execute" 
        subtitle="Track and manage your tasks and action items"
      />

      {/* Stats Row */}
      <div className="row g-4 mb-4">
        {stats.map((stat, index) => (
          <div key={index} className="col-lg-3 col-md-6">
            <DashboardCard>
              <div className="d-flex justify-content-between align-items-center">
                <div>
                  <div className="text-muted small mb-1">{stat.label}</div>
                  <h3 className="mb-0 fw-bold">{stat.value}</h3>
                </div>
                <div className={`stat-icon-circle bg-${stat.color} bg-opacity-10`}>
                  <i className={`bi ${stat.icon} text-${stat.color} fs-3`}></i>
                </div>
              </div>
            </DashboardCard>
          </div>
        ))}
      </div>

      {/* Tasks Section */}
      <DashboardCard>
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h5 className="section-title mb-0">Action Items</h5>
          <button className="btn btn-primary btn-sm">
            <i className="bi bi-plus-circle me-1"></i> New Task
          </button>
        </div>

        {/* Tabs */}
        <ul className="nav nav-tabs mb-4">
          <li className="nav-item">
            <button 
              className={`nav-link ${activeTab === 'pending' ? 'active' : ''}`}
              onClick={() => setActiveTab('pending')}
            >
              Pending ({tasks.pending.length})
            </button>
          </li>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeTab === 'inProgress' ? 'active' : ''}`}
              onClick={() => setActiveTab('inProgress')}
            >
              In Progress ({tasks.inProgress.length})
            </button>
          </li>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeTab === 'completed' ? 'active' : ''}`}
              onClick={() => setActiveTab('completed')}
            >
              Completed ({tasks.completed.length})
            </button>
          </li>
        </ul>

        {/* Task List */}
        <div className="task-list">
          {tasks[activeTab].map((task) => (
            <div key={task.id} className="task-item">
              <div className="d-flex gap-3">
                <div className="form-check">
                  <input 
                    className="form-check-input" 
                    type="checkbox" 
                    defaultChecked={activeTab === 'completed'}
                  />
                </div>
                <div className="flex-grow-1">
                  <div className="d-flex justify-content-between align-items-start mb-2">
                    <div>
                      <h6 className="mb-1">{task.title}</h6>
                      <p className="text-muted small mb-2">{task.description}</p>
                      <div className="d-flex gap-2 flex-wrap">
                        <span className={`badge bg-${getPriorityColor(task.priority)}`}>
                          {task.priority}
                        </span>
                        {task.tags.map((tag, index) => (
                          <span key={index} className="badge bg-light text-dark">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="text-end">
                      <button className="btn btn-sm btn-outline-secondary me-2">
                        <i className="bi bi-pencil"></i>
                      </button>
                      <button className="btn btn-sm btn-outline-danger">
                        <i className="bi bi-trash"></i>
                      </button>
                    </div>
                  </div>
                  <div className="d-flex gap-3 text-muted small">
                    <span>
                      <i className="bi bi-person me-1"></i>
                      {task.assignee}
                    </span>
                    <span>
                      <i className="bi bi-calendar me-1"></i>
                      {task.dueDate}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </DashboardCard>

      {/* Activity Feed */}
      <div className="row g-4 mt-2">
        <div className="col-lg-6">
          <DashboardCard>
            <h5 className="section-title mb-4">Recent Updates</h5>
            <div className="activity-feed">
              {[
                { user: 'John Smith', action: 'completed', task: 'Submit expense report', time: '1 hour ago' },
                { user: 'Sarah Johnson', action: 'started', task: 'Review contract terms', time: '3 hours ago' },
                { user: 'Mike Davis', action: 'commented on', task: 'Update CRM records', time: '5 hours ago' },
                { user: 'Lisa Chen', action: 'created', task: 'Schedule team training', time: '1 day ago' }
              ].map((activity, index) => (
                <div key={index} className="activity-item mb-3">
                  <div className="d-flex align-items-start gap-3">
                    <div className="user-avatar-sm">
                      {activity.user.charAt(0)}
                    </div>
                    <div className="flex-grow-1">
                      <div>
                        <strong>{activity.user}</strong> {activity.action}{' '}
                        <span className="text-primary">{activity.task}</span>
                      </div>
                      <small className="text-muted">{activity.time}</small>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>
        </div>

        <div className="col-lg-6">
          <DashboardCard>
            <h5 className="section-title mb-4">Team Workload</h5>
            <div className="workload-list">
              {[
                { name: 'John Smith', tasks: 5, progress: 80 },
                { name: 'Sarah Johnson', tasks: 8, progress: 60 },
                { name: 'Mike Davis', tasks: 3, progress: 90 },
                { name: 'Lisa Chen', tasks: 6, progress: 70 },
                { name: 'Tom Wilson', tasks: 4, progress: 75 }
              ].map((member, index) => (
                <div key={index} className="workload-item mb-3">
                  <div className="d-flex justify-content-between mb-2">
                    <div className="d-flex align-items-center gap-2">
                      <div className="user-avatar-sm">
                        {member.name.charAt(0)}
                      </div>
                      <strong>{member.name}</strong>
                    </div>
                    <span className="text-muted small">{member.tasks} tasks</span>
                  </div>
                  <div className="progress" style={{ height: '6px' }}>
                    <div 
                      className="progress-bar bg-success" 
                      style={{ width: `${member.progress}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  );
}

export default Execute;