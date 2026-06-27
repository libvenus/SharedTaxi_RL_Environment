import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchActivityTimeline } from "../api/client";
import "../styles/dashboard.css";
import "bootstrap-icons/font/bootstrap-icons.css";

function getIconForType(type) {
	switch (type) {
		case "meeting":
			return { icon: "bi-calendar-event", iconBg: "#DCFDFF", iconColor: "#475569" };
		case "email":
			return { icon: "bi-envelope", iconBg: "#E2E8F0", iconColor: "#475569" };
		case "crm_update":
			return { icon: "bi-pencil-square", iconBg: "#FEF9C3", iconColor: "#92400E" };
		case "call":
			return { icon: "bi-telephone", iconBg: "#DCFCE7", iconColor: "#475569" };
		default:
			return { icon: "bi-bell", iconBg: "#E2E8F0", iconColor: "#475569" };
	}
}

function getBadgeClass(type) {
	switch (type) {
		case "meeting":
			return "at-badge-meeting";
		case "email":
			return "at-badge-outreach";
		case "crm_update":
			return "at-badge-crm";
		default:
			return "at-badge-outreach";
	}
}

function formatTimelineDate(dateStr) {
	const date = new Date(dateStr);
	const today = new Date();
	const yesterday = new Date();
	yesterday.setDate(today.getDate() - 1);

	if (date.toDateString() === today.toDateString()) return "Today";
	if (date.toDateString() === yesterday.toDateString()) return "Yesterday";

	return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatTime(dateStr) {
	return new Date(dateStr).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true });
}

function groupByDate(items) {
	const groups = {};
	items.forEach((item) => {
		const label = formatTimelineDate(item.eventAt);
		if (!groups[label]) groups[label] = [];
		groups[label].push(item);
	});
	return Object.entries(groups).map(([date, items]) => ({ date, items }));
}

export default function ActivityTimeline() {
	const navigate = useNavigate();
	const [notifications, setNotifications] = useState([]);

	useEffect(() => {
		fetchActivityTimeline("055DAFE7-9840-451D-8328-5F70A6326C03", { page: 1, pageSize: 25 })
			.then((res) => setNotifications(res?.items || []))
			.catch((err) => console.error("Failed to fetch activity timeline:", err));
	}, []);

	const timelineSections = groupByDate(notifications);

	return (
		<div className="sd-wrapper">
			<div className="at-header">
				<button className="at-back-btn" onClick={() => navigate(-1)}>
					‹
				</button>
				<div className="at-header-text">
					<h1 className="at-title">Activity Timeline</h1>
					<span className="at-subtitle">All interactions across your accounts</span>
				</div>
			</div>

			<div className="at-timeline">
				{timelineSections.map((section, si) => (
					<div key={si} className="at-section">
						<div className="at-date-label">{section.date}</div>
						{section.items.map((item, i) => {
							const { icon, iconBg, iconColor } = getIconForType(item.activityType);
							return (
								<div key={item.id || i} className="at-item">
									<div className="at-card">
										<div className="at-card-header">
											<div className="sd-summary-label-row">
												<div className="sd-summary-label" style={{ marginBottom: 0 }}>
													<span
														className="at-icon-badge"
														style={{ background: iconBg, color: iconColor }}
													>
														<i className={`bi ${icon}`}></i>
													</span>
													{item.title}
													<span className={`at-badge ${getBadgeClass(item.activityType)}`}>{item.categoryLabel}</span>
												</div>
												<div className="at-time-row">
													<span className={`at-direction ${item.direction === "inbound" ? "at-direction-inbound" : "at-direction-outbound"}`}>
														{item.direction === "inbound" ? "Inbound" : "Outbound"}
													</span>
													<span className="sd-summary-time">{formatTime(item.eventAt)}</span>
												</div>
											</div>
											<div className="at-card-subtitle">
												{item.opportunityName}
												{item.actorName && (
													<>
														<span className="at-card-dot">·</span>
														<span className="at-card-person">{item.actorName}</span>
													</>
												)}
											</div>
										</div>
										<div className="at-card-body">
											<div className="sd-summary-desc">
												{item.summary}
											</div>
										</div>
									</div>
								</div>
							);
						})}
					</div>
				))}
			</div>
		</div>
	);
}
