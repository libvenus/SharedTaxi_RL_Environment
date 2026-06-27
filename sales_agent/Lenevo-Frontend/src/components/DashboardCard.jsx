function DashboardCard({ title, children, className = '' }) {
  return (
    <div className={`card dashboard-card ${className}`}>
      <div className="card-body">
        {title && <h5 className="card-title">{title}</h5>}
        {children}
      </div>
    </div>
  );
}

export default DashboardCard;