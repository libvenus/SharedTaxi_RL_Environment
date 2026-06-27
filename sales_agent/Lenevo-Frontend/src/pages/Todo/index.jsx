import React, { useMemo, useState, useEffect } from "react";
import accountIcon from "../../assets/icons/account_icon.png";
import "../../styles/execute.css";
import {
    fetchTodos,
    createTodo,
    updateTodo,
    updateTodoStatus,
    fetchOutreach,
    lookupAccounts,
    lookupOpportunities,
} from "../../api/client";
import { Modal } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import { formatCurrencyShort } from "../../utils/format";
const Todo = () => {
    const navigate = useNavigate();
    const initialFormData = {
        title: "",
        type: "",
        priority: "",
        linkedAccount: "",
        linkedOpportunity: "",
        dueDate: "",
        addText: "",
    };

    const [activeTopTab, setActiveTopTab] = useState("pending");
    const [searchText, setSearchText] = useState("");
    const [accountFilter, setAccountFilter] = useState("all");
    const [typeFilter, setTypeFilter] = useState("all");
    const [dueDateFilter, setDueDateFilter] = useState("all");
    const [priorityFilter, setPriorityFilter] = useState("all");
    const [showAddForm, setShowAddForm] = useState(false);
    const [formData, setFormData] = useState(initialFormData);
    const [todoData, setTodoData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    const [successMessage, setSuccessMessage] = useState("");
    const [editingTask, setEditingTask] = useState(null);
    const [accountsList, setAccountsList] = useState([]);
    const [opportunitiesList, setOpportunitiesList] = useState([]);
    const [accountSearch, setAccountSearch] = useState("");
    const [opportunitySearch, setOpportunitySearch] = useState("");
    const [showAccountDropdown, setShowAccountDropdown] = useState(false);
    const [showOpportunityDropdown, setShowOpportunityDropdown] = useState(false);
    const [accountLoading, setAccountLoading] = useState(false);
    const [opportunityLoading, setOpportunityLoading] = useState(false);

    const statusFilter = activeTopTab;

    const loadTodos = () => {
        const controller = new AbortController();
        setLoading(true);
        fetchTodos({ filter_type: typeFilter, show_completed: activeTopTab === "completed" }, controller.signal)
            .then((data) => {
                setTodoData(data);
                setLoading(false);
            })
            .catch((err) => {
                if (err.name !== "AbortError") {
                    console.error("Failed to fetch todos:", err);
                    setLoading(false);
                }
            });
        return controller;
    };

    useEffect(() => {
        const controller = loadTodos();
        return () => controller.abort();
    }, [typeFilter, activeTopTab]);

    useEffect(() => {
        if (!showAccountDropdown) return;
        const controller = new AbortController();
        const timer = setTimeout(() => {
            setAccountLoading(true);
            lookupAccounts(accountSearch, controller.signal)
                .then((data) => {
                    setAccountsList(data?.items || []);
                    setAccountLoading(false);
                })
                .catch((err) => {
                    if (err.name !== "AbortError") {
                        console.error("Failed to lookup accounts:", err);
                        setAccountLoading(false);
                    }
                });
        }, 300);
        return () => { clearTimeout(timer); controller.abort(); };
    }, [accountSearch, showAccountDropdown]);

    useEffect(() => {
        if (!showOpportunityDropdown) return;
        const controller = new AbortController();
        const timer = setTimeout(() => {
            setOpportunityLoading(true);
            lookupOpportunities(opportunitySearch, controller.signal)
                .then((data) => {
                    setOpportunitiesList(data?.items || []);
                    setOpportunityLoading(false);
                })
                .catch((err) => {
                    if (err.name !== "AbortError") {
                        console.error("Failed to lookup opportunities:", err);
                        setOpportunityLoading(false);
                    }
                });
        }, 300);
        return () => { clearTimeout(timer); controller.abort(); };
    }, [opportunitySearch, showOpportunityDropdown]);

    useEffect(() => {
        if (activeTopTab !== "completed") return;
        setShowAddForm(false);
        setEditingTask(null);
        setSearchText("");
        setAccountFilter("all");
        setTypeFilter("all");
        setDueDateFilter("all");
        setPriorityFilter("all");
    }, [activeTopTab]);

    const today = new Date().toISOString().split("T")[0];

    const filteredTasks = useMemo(() => {
        let tasks = Array.isArray(todoData?.tasks) ? todoData.tasks : [];
        if (statusFilter === "pending") {
            tasks = tasks.filter((t) => t.status?.toLowerCase() !== "completed");
        } else {
            tasks = tasks.filter((t) => t.status?.toLowerCase() === "completed");
        }
        if (searchText.trim()) {
            const q = searchText.toLowerCase();
            tasks = tasks.filter(
                (t) =>
                    (t.task_title || "").toLowerCase().includes(q) ||
                    (t.notes || "").toLowerCase().includes(q) ||
                    (t.account_name || "").toLowerCase().includes(q),
            );
        }
        if (accountFilter !== "all") {
            tasks = tasks.filter((t) =>
                (t.account_name || "").toLowerCase().includes(accountFilter.toLowerCase()),
            );
        }
        if (priorityFilter !== "all") {
            tasks = tasks.filter((t) => (t.priority || "").toLowerCase() === priorityFilter.toLowerCase());
        }
        if (dueDateFilter === "today") {
            tasks = tasks.filter((t) => t.due_date === today);
        } else if (dueDateFilter === "this-week") {
            const weekEnd = new Date();
            weekEnd.setDate(weekEnd.getDate() + 7);
            const weekEndStr = weekEnd.toISOString().split("T")[0];
            tasks = tasks.filter((t) => t.due_date && t.due_date >= today && t.due_date <= weekEndStr);
        } else if (dueDateFilter === "overdue") {
            tasks = tasks.filter((t) => t.due_date && t.due_date < today);
        }
        return tasks;
    }, [statusFilter, todoData?.tasks, searchText, accountFilter, priorityFilter, dueDateFilter, today]);

    const pendingDueTodayTasks = filteredTasks.filter(
        (task) => task.status?.toLowerCase() !== "completed" && task.due_date === today,
    );
    const completedTasks = filteredTasks.filter(
        (task) => task.status?.toLowerCase() === "completed",
    );
    const summaryTitle =
        statusFilter === "completed"
            ? `${completedTasks.length} completed actions`
            : `${pendingDueTodayTasks.length} actions due today`;
    const summaryItems =
        statusFilter === "completed" ? completedTasks : pendingDueTodayTasks;

    const handleFormChange = (e) => {
        const { name, value } = e.target;
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    const handleAddClick = () => {
        setEditingTask(null);
        setFormData(initialFormData);
        setAccountSearch("");
        setOpportunitySearch("");
        setShowAccountDropdown(false);
        setShowOpportunityDropdown(false);
        setShowAddForm(!showAddForm);
    };

    const handleFormSave = async () => {
        try {
            await createTodo({
                title: formData.title,
                type_tag: formData.type,
                priority: formData.priority,
                linked_account_id: formData.linkedAccount || null,
                linked_opportunity_id: formData.linkedOpportunity || null,
                due_date: formData.dueDate || null,
                notes: formData.addText || null,
            });
            setShowAddForm(false);
            setFormData(initialFormData);
            setSuccessMessage("Todo created successfully");
            setShowSuccessModal(true);
        } catch (err) {
            console.error("Failed to create todo:", err);
        }
    };

    const handleEditClick = (task) => {
        setEditingTask(task);
        setAccountSearch(task.linked_account_name || task.linked_account_id || "");
        setOpportunitySearch(task.linked_opportunity_name || task.linked_opportunity_id || "");
        setFormData({
            title: task.task_title || "",
            type: task.type_tag || "",
            priority: task.priority || "",
            linkedAccount: task.linked_account_id || "",
            linkedOpportunity: task.linked_opportunity_id || "",
            dueDate: task.due_date || "",
            addText: task.notes || "",
        });
        setShowAddForm(true);
    };

    const handleFormUpdate = async () => {
        try {
            await updateTodo(editingTask.id, {
                task_title: formData.title,
                type_tag: formData.type,
                priority: formData.priority,
                source_label: editingTask.source_label || "",
                due_date: formData.dueDate || null,
            });
            setShowAddForm(false);
            setEditingTask(null);
            setFormData(initialFormData);
            setSuccessMessage("Todo updated successfully");
            setShowSuccessModal(true);
        } catch (err) {
            console.error("Failed to update todo:", err);
        }
    };

    return (
        <div className="dv-page">
            <style>{`
                .todo-context-card-title {
                    font-size: 16px;
                    font-weight: 700;
                    color: #0f172a;
                    margin: 0 0 12px;
                }
                .todo-context-line {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 10px;
                    padding: 8px 0;
                    border-bottom: 1px solid #e2e8f0;
                    font-size: 13px;
                    color: #64748b;
                }
                .todo-context-line:last-child {
                    border-bottom: none;
                    padding-bottom: 0;
                }
                .todo-context-line strong {
                    color: #0f172a;
                    font-weight: 700;
                }
                .todo-proposal-badge {
                    background: #eff6ff;
                    color: #1d4ed8;
                    border-radius: 999px;
                    padding: 2px 10px;
                    font-size: 12px;
                    font-weight: 600;
                    white-space: nowrap;
                }
            `}</style>
            <div className="main">
                <div className="quarter-bar">
                    Q2 FY2024 · Week 10 of 12 · <span className="closure-badge">Closure Phase</span>
                </div>

                <div className="page-header">
                    <div className="page-header-row">
                        <span className="topbar-title">
                            {activeTopTab === "pending"
                                ? "Todo List : Pending Tasks"
                                : "Todo List : Completed Tasks"}
                        </span>
                    </div>

                    <div className="tabs-outer">
                        <div className="tabs">
                            <button
                                className={`tab ${activeTopTab === "pending" ? "active" : ""}`}
                                onClick={() => setActiveTopTab("pending")}
                            >
                                Pending
                            </button>
                            <button
                                className={`tab ${activeTopTab === "completed" ? "active" : ""}`}
                                onClick={() => setActiveTopTab("completed")}
                            >
                                Completed
                            </button>
                        </div>
                    </div>
                </div>

                <div className="content">
                    <div className="col-main tdl-root">


                        {statusFilter !== "completed" && (
                            <>
                                <div className="tdl-actions-wrap" style={{ display: "flex", gap: 16, alignItems: "stretch" }}>
                                    <div className="tdl-actions-head-card" style={{ flex: "0 0 70%", maxWidth: "70%" }}>
                                        <h3 className="tdl-actions-title">{summaryTitle}</h3>
                                        <div className="tdl-actions-card-rainbow">
                                            <div className="tdl-actions-card-inner">
                                                <div className="tdl-actions-icon-col">
                                                    <img src={accountIcon} alt="Actions icon" className="tdl-actions-icon" />
                                                </div>
                                                <div className="tdl-actions-text-grid">
                                                    {summaryItems.map((t, i) => (
                                                        <p className="tdl-action-item" key={t.id}>
                                                            <span className="tdl-action-num">{i + 1}.</span> {t.task_title}
                                                        </p>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="tdl-actions-head-card" style={{ flex: "0 0 30%", maxWidth: "30%" }}>
                                        <h3 className="sd-draft-card-title todo-context-card-title">Outreach Context</h3>

                                        <div className="sd-context-line todo-context-line">
                                            <span>High Priority</span>
                                            <strong>{Array.isArray(todoData?.tasks) ? todoData.tasks.filter((t) => t.priority?.toLowerCase() === "high").length : 0}</strong>
                                        </div>
                                        <div className="sd-context-line todo-context-line">
                                            <span>Pending</span>
                                            <strong>{todoData?.filters?.all ?? 0}</strong>
                                        </div>
                                        <div className="sd-context-line todo-context-line">
                                            <span>Due Today</span>
                                            <strong>{todoData?.summary?.today ?? 0}</strong>
                                        </div>
                                        <div className="sd-context-line todo-context-line">
                                            <span>Overdue</span>
                                            <strong>{todoData?.summary?.overdue ?? 0}</strong>
                                        </div>
                                    </div>
                                </div>

                                <div className="tdl-filter-bar">
                                    <div className="tdl-search-box">
                                        <input
                                            type="text"
                                            placeholder="Search"
                                            value={searchText}
                                            onChange={(e) => setSearchText(e.target.value)}
                                        />
                                    </div>

                                    <div className="tdl-dd-wrap">
                                        <select className="tdl-dd-select" value={accountFilter} onChange={(e) => setAccountFilter(e.target.value)}>
                                            <option value="all">Account</option>
                                            {[...new Set((todoData?.tasks || []).map((t) => t.account_name).filter(Boolean))].sort().map((name) => (
                                                <option key={name} value={name}>{name}</option>
                                            ))}
                                        </select>
                                        <span className="tdl-dd-icon"><i className="bi bi-caret-down-fill"></i></span>
                                    </div>

                                    <div className="tdl-dd-wrap">
                                        <select className="tdl-dd-select" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                                            <option value="all">All Types</option>
                                            <option value="outreach">Outreach</option>
                                            <option value="document">Document</option>
                                            <option value="action">Action</option>
                                        </select>
                                        <span className="tdl-dd-icon"><i className="bi bi-caret-down-fill"></i></span>
                                    </div>

                                    <div className="tdl-dd-wrap">
                                        <select className="tdl-dd-select" value={dueDateFilter} onChange={(e) => setDueDateFilter(e.target.value)}>
                                            <option value="all">Due Date</option>
                                            <option value="today">Today</option>
                                            <option value="this-week">This Week</option>
                                            <option value="overdue">Overdue</option>
                                        </select>
                                        <span className="tdl-dd-icon"><i className="bi bi-caret-down-fill"></i></span>
                                    </div>

                                    <div className="tdl-dd-wrap">
                                        <select className="tdl-dd-select" value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
                                            <option value="all">Priority</option>
                                            <option value="high">High</option>
                                            <option value="medium">Medium</option>
                                            <option value="low">Low</option>
                                        </select>
                                        <span className="tdl-dd-icon"><i className="bi bi-caret-down-fill"></i></span>
                                    </div>

                                    <button className="tdl-btn-dl" type="button" aria-label="Download">
                                        <i className="bi bi-download"></i>
                                    </button>
                                </div>
                            </>
                        )}

                        <div className="sd-meetings-panel">
                            <div className="sd-meetings-header" style={{ display: "flex", alignItems: "center", gap: 12 }}>
                                <h2 className="sd-meetings-title" style={{ margin: 0 }}>To Do List</h2>
                                <span className="sd-priority-meta" style={{ margin: 0 }}>
                                    {filteredTasks.length} total action &middot;{" "}
                                    <span className="sd-priority-due-today">
                                        {pendingDueTodayTasks.length} due today
                                    </span>
                                </span>
                                <button
                                    className="dv-exec-add-btn"
                                    type="button"
                                    aria-label="Add Action"
                                    onClick={handleAddClick}
                                    style={{ marginLeft: "auto" }}
                                >
                                    <i className="bi bi-plus-lg"></i>
                                </button>
                            </div>

                            {showAddForm && (
                                <div className="tdl-add-action-form">
                                    <div className="ct-new-form">
                                        <div className="ct-form-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                            <div className="ct-form-title">{editingTask ? "Edit Action" : "Create New Task"}</div>
                                            <button
                                                type="button"
                                                style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b", fontSize: 20 }}
                                                onClick={() => {
                                                    setShowAddForm(false);
                                                    setEditingTask(null);
                                                    setFormData(initialFormData);
                                                }}
                                            >
                                                <i className="ri-close-line"></i>
                                            </button>
                                        </div>

                                        <div className="ct-fields">
                                            <div className="ct-field">
                                                <label>Title</label>
                                                <input type="text" name="title" value={formData.title} onChange={handleFormChange} placeholder="Enter action title" />
                                            </div>

                                            <div className="ct-field">
                                                <label>Type</label>
                                                <div className="ct-select-wrap">
                                                    <select name="type" value={formData.type} onChange={handleFormChange}>
                                                        <option value="">Select Type</option>
                                                        <option value="outreach">Outreach</option>
                                                        <option value="document">Document</option>
                                                        <option value="action">Action</option>
                                                    </select>
                                                    <span className="tdl-dd-icon"><i className="bi bi-caret-down-fill"></i></span>
                                                </div>
                                            </div>

                                            <div className="ct-field">
                                                <label>Priority</label>
                                                <div className="ct-select-wrap">
                                                    <select name="priority" value={formData.priority} onChange={handleFormChange}>
                                                        <option value="">Select Priority</option>
                                                        <option value="High">High</option>
                                                        <option value="Medium">Medium</option>
                                                        <option value="Low">Low</option>
                                                    </select>
                                                    <span className="tdl-dd-icon"><i className="bi bi-caret-down-fill"></i></span>
                                                </div>
                                            </div>

                                            <div className="ct-field" style={{ position: "relative" }}>
                                                <label>Linked Account</label>
                                                <input
                                                    type="text"
                                                    autoComplete="off"
                                                    value={accountSearch}
                                                    onChange={(e) => {
                                                        setAccountSearch(e.target.value);
                                                        setFormData((prev) => ({ ...prev, linkedAccount: "" }));
                                                        setShowAccountDropdown(true);
                                                    }}
                                                    onFocus={() => setShowAccountDropdown(true)}
                                                    onBlur={() => setTimeout(() => setShowAccountDropdown(false), 200)}
                                                    placeholder="Search account"
                                                />
                                                {showAccountDropdown && (
                                                    <ul style={{ position: "absolute", top: "100%", left: 0, right: 0, zIndex: 10, background: "#fff", border: "1px solid #cbd5e1", borderRadius: 6, maxHeight: 180, overflowY: "auto", margin: 0, padding: 0, listStyle: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}>
                                                        {accountLoading && <li style={{ padding: "8px 12px", fontSize: 13, color: "#64748b" }}>Searching...</li>}
                                                        {!accountLoading && accountsList.length === 0 && <li style={{ padding: "8px 12px", fontSize: 13, color: "#64748b" }}>No accounts found</li>}
                                                        {!accountLoading && accountsList.map((acc) => (
                                                            <li
                                                                key={acc.id}
                                                                style={{ padding: "8px 12px", cursor: "pointer", fontSize: 13, color: "#0f172a", borderBottom: "1px solid #f1f5f9" }}
                                                                onMouseDown={() => {
                                                                    setFormData((prev) => ({ ...prev, linkedAccount: acc.id }));
                                                                    setAccountSearch(acc.name);
                                                                    setShowAccountDropdown(false);
                                                                }}
                                                                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                                                                onMouseLeave={(e) => (e.currentTarget.style.background = "#fff")}
                                                            >
                                                                {acc.name}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                )}
                                            </div>

                                            <div className="ct-field" style={{ position: "relative" }}>
                                                <label>Linked Opportunity</label>
                                                <input
                                                    type="text"
                                                    autoComplete="off"
                                                    value={opportunitySearch}
                                                    onChange={(e) => {
                                                        setOpportunitySearch(e.target.value);
                                                        setFormData((prev) => ({ ...prev, linkedOpportunity: "" }));
                                                        setShowOpportunityDropdown(true);
                                                    }}
                                                    onFocus={() => setShowOpportunityDropdown(true)}
                                                    onBlur={() => setTimeout(() => setShowOpportunityDropdown(false), 200)}
                                                    placeholder="Search opportunity"
                                                />
                                                {showOpportunityDropdown && (
                                                    <ul style={{ position: "absolute", top: "100%", left: 0, right: 0, zIndex: 10, background: "#fff", border: "1px solid #cbd5e1", borderRadius: 6, maxHeight: 180, overflowY: "auto", margin: 0, padding: 0, listStyle: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}>
                                                        {opportunityLoading && <li style={{ padding: "8px 12px", fontSize: 13, color: "#64748b" }}>Searching...</li>}
                                                        {!opportunityLoading && opportunitiesList.length === 0 && <li style={{ padding: "8px 12px", fontSize: 13, color: "#64748b" }}>No opportunities found</li>}
                                                        {!opportunityLoading && opportunitiesList.map((opp) => (
                                                            <li
                                                                key={opp.id}
                                                                style={{ padding: "8px 12px", cursor: "pointer", fontSize: 13, color: "#0f172a", borderBottom: "1px solid #f1f5f9" }}
                                                                onMouseDown={() => {
                                                                    setFormData((prev) => ({ ...prev, linkedOpportunity: opp.id }));
                                                                    setOpportunitySearch(opp.name);
                                                                    setShowOpportunityDropdown(false);
                                                                }}
                                                                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                                                                onMouseLeave={(e) => (e.currentTarget.style.background = "#fff")}
                                                            >
                                                                {opp.name}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                )}
                                            </div>

                                            <div className="ct-field">
                                                <label>Due Date</label>
                                                <div className="ct-date-input-wrap">
                                                    <input type="date" name="dueDate" value={formData.dueDate} onChange={handleFormChange} />
                                                    <span className="ct-date-icon"><i className="bi bi-calendar"></i></span>
                                                </div>
                                            </div>

                                            <div className="ct-field ct-field-full">
                                                <label>Add</label>
                                                <div className="ct-textarea-wrap">
                                                    <textarea name="addText" value={formData.addText} onChange={handleFormChange} placeholder="Type here" maxLength={1000} />
                                                    <div className="ct-text-count">{formData.addText.length}/1000</div>
                                                </div>
                                            </div>

                                            <button
                                                className="ct-btn-cancel"
                                                type="button"
                                                onClick={() => {
                                                    if (editingTask) {
                                                        setShowAddForm(false);
                                                        setEditingTask(null);
                                                    }
                                                    setFormData(initialFormData);
                                                }}
                                            >
                                                {editingTask ? "Cancel" : "Clear"}
                                            </button>

                                            <button className="ct-btn-save" type="button" onClick={editingTask ? handleFormUpdate : handleFormSave}>
                                                {editingTask ? "Update" : "Add"}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                                {loading && <p style={{ padding: 16, color: "#64748b" }}>Loading...</p>}

                                {!loading && filteredTasks.length === 0 && (
                                    <p style={{ padding: 16, color: "#64748b" }}>No data available</p>
                                )}

                                {!loading &&
                                    filteredTasks.map((task) => {
                                        const isCompleted = task.status?.toLowerCase() === "completed";
                                        const isDueToday = !isCompleted && task.due_date === today;
                                        const isDueSoon = !isCompleted && !isDueToday && task.due_date > today;
                                        const isOverdue = !isCompleted && task.due_date < today;

                                        return (
                                            <div className="ex-meeting-card" key={task.id}>
                                                <div className="ex-meeting-layout">
                                                    <div style={{ display: "flex", alignItems: "flex-start", gap: 14, flex: 1 }}>
                                                        <div className="tdl-card-icon-box">
                                                            <i className="ri-send-plane-fill"></i>
                                                        </div>
                                                        <div style={{ flex: 1 }}>
                                                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                                <span className="sd-meeting-title" style={{ margin: 0 }}>{task.task_title}</span>
                                                            </div>

                                                            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginTop: 4 }}>
                                                                {task.account_name && (
                                                                    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                                                                        <i className="bi bi-building" style={{ fontSize: 13, color: "#0F172A" }} aria-hidden="true"></i>
                                                                        <span style={{ fontSize: 13, color: "#0f172a", textDecoration: "underline", whiteSpace: "nowrap" }}>
                                                                            {task.account_name}
                                                                        </span>
                                                                    </div>
                                                                )}
                                                                {task.opportunity_name && (
                                                                    <span style={{ fontSize: 12, color: "#475569" }}>{task.opportunity_name}</span>
                                                                )}
                                                                {task.stage_name && (
                                                                    <span style={{ fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 12, background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" }}>
                                                                        {task.stage_name}
                                                                    </span>
                                                                )}
                                                                {task.deal_value != null && (
                                                                    <strong style={{ fontSize: 13, color: "#0f172a" }}>{formatCurrencyShort(task.deal_value)}</strong>
                                                                )}
                                                            </div>

                                                            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginTop: 6 }}>
                                                                {task.due_date && (
                                                                    <span style={{ fontSize: 12, color: "#64748b" }}>
                                                                        <i className="bi bi-calendar3" style={{ marginRight: 3 }}></i>
                                                                        {new Date(task.due_date + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                                                                    </span>
                                                                )}
                                                                {task.priority && (() => {
                                                                    const p = task.priority.toLowerCase();
                                                                    const colors = { high: { bg: "#fef2f2", color: "#b91c1c", border: "#fecaca" }, medium: { bg: "#fffbeb", color: "#92400e", border: "#fde68a" }, low: { bg: "#f0fdf4", color: "#15803d", border: "#bbf7d0" } };
                                                                    const c = colors[p] || colors.medium;
                                                                    return (
                                                                        <span style={{ fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 12, background: c.bg, color: c.color, border: `1px solid ${c.border}` }}>
                                                                            {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)} Priority
                                                                        </span>
                                                                    );
                                                                })()}
                                                                {task.source_label && (
                                                                    <span style={{ fontSize: 11, color: "#94a3b8" }}>
                                                                        <i className="bi bi-link-45deg" style={{ marginRight: 2 }}></i>{task.source_label}
                                                                    </span>
                                                                )}
                                                            </div>

                                                            <p style={{ fontSize: 13, color: "#475569", marginTop: 8, marginBottom: 0, lineHeight: 1.5 }}>
                                                                {task.notes}
                                                            </p>



                                                            {!isCompleted && (
                                                                <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                                                                    <button
                                                                        type="button"
                                                                        className="kp-btn-cancel"
                                                                        onClick={async () => {
                                                                            try {
                                                                                await updateTodoStatus(task.id, "completed");
                                                                                loadTodos();
                                                                            } catch (err) {
                                                                                console.error("Failed to complete task:", err);
                                                                            }
                                                                        }}
                                                                    >
                                                                        <span>Complete Task</span>
                                                                    </button>

                                                                    <button
                                                                        type="button"
                                                                        className="sd-draft-btn"
                                                                        onClick={() =>
                                                                            navigate(`/todo/${task.id}`, {
                                                                                state: { selectedTask: task },
                                                                            })
                                                                        }
                                                                    >
                                                                        <img src={accountIcon} alt="Draft Email" style={{ width: 16, height: 16, marginRight: 6 }} />
                                                                        <span>Draft Email</span>
                                                                    </button>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>

                                                    <div style={{ display: "flex", alignItems: "flex-start", gap: 8, flexShrink: 0 }}>
                                                        {isCompleted && (
                                                            <span className="sd-badge-due-soon" style={{ background: "#d1fae5", color: "#059669" }}>
                                                                <i className="bi bi-check-circle-fill" style={{ marginRight: 4 }}></i> Completed
                                                            </span>
                                                        )}

                                                        {isDueToday && (
                                                            <span className="sd-badge-due-today">
                                                                <span className="sd-badge-icon-red"><i className="bi bi-exclamation-circle-fill"></i></span> Due Today
                                                            </span>
                                                        )}

                                                        {isDueSoon && (
                                                            <span className="sd-badge-due-soon">
                                                                <span className="sd-badge-icon-warning"><i className="bi bi-exclamation-triangle-fill"></i></span> Due Soon
                                                            </span>
                                                        )}

                                                        {isOverdue && (
                                                            <span className="sd-badge-due-today">
                                                                <span className="sd-badge-icon-red"><i className="bi bi-exclamation-circle-fill"></i></span> Overdue
                                                            </span>
                                                        )}

                                                        {!isCompleted && (
                                                            <button className="sd-badge-edit-btn" type="button" aria-label="Edit action" onClick={() => handleEditClick(task)}>
                                                                <i className="bi bi-pencil"></i>
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                            </div>
                        </div>

                        <Modal show={showSuccessModal} backdrop="static" keyboard={false} centered>
                            <Modal.Body style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                <i className="ri-checkbox-circle-fill" style={{ color: "#047857", fontSize: "20px" }}></i>
                                <span>{successMessage}</span>
                            </Modal.Body>
                            <Modal.Footer style={{ border: "none" }}>
                                <button
                                    className="kp-btn-update"
                                    onClick={() => {
                                        setShowSuccessModal(false);
                                        loadTodos();
                                    }}
                                >
                                    OK
                                </button>
                            </Modal.Footer>
                        </Modal>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Todo;