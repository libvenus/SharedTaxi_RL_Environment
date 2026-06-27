import { useState } from 'react';
import { menuItems, adminMenuItem } from '../data/menuData';
import '../styles/sidebar.css';


function StatWidget({ label, value, icon, color = 'primary' }) {
  return (
    <div className={`stat-widget stat-${color}`}>
      <div className="stat-content">
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
      </div>
      {icon && (
        <div className="stat-icon">
          <i className={`bi ${icon}`}></i>
        </div>
      )}
    </div>
  );
}

export default StatWidget;