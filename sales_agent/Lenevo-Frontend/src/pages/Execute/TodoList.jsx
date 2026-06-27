import React, { useState, useEffect } from "react";
import accountIcon from "../../assets/icons/account_icon.png";
import "../../styles/execute.css";
import { fetchTodos, createTodo, updateTodo, updateTodoStatus } from "../../api/client";
import { Modal } from "react-bootstrap";

const TodoList = () => {
  const initialFormData = {
    title: "",
    type: "",
    priority: "",
    linkedAccount: "",
    linkedOpportunity: "",
    dueDate: "",
    addText: "",
  };

  const [activeTopTab, setActiveTopTab] = useState("todolist");
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

  const loadTodos = () => {
    const controller = new AbortController();
    setLoading(true);
    fetchTodos({ filter_type: typeFilter }, controller.signal)
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
  }, [typeFilter]);

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleAddClick = () => {
    setEditingTask(null);
    setFormData(initialFormData);
    setShowAddForm(!showAddForm);
  };

  const handleFormCancel = () => {
    setShowAddForm(false);
    setFormData(initialFormData);
  };

  const handleFormClear = () => {
    setFormData(initialFormData);
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
  const tabs = [
    { key: "outreachstudio", label: "Outreach Studio" },
    { key: "todolist", label: "To Do List" },
  ];

  return (
    <div className="col-main tdl-root">
      <div className="tdl-summary-row">
        <div className="tdl-summary-col">
          <div className="tdl-scard-simple">
            <span className="tdl-label">Today</span>
            <span className="tdl-value">{todoData?.summary?.today ?? 0}</span>
          </div>
        </div>
        <div className="tdl-summary-col">
          <div className="tdl-scard-simple">
            <span className="tdl-label">Upcoming</span>
            <span className="tdl-value">
              {todoData?.summary?.upcoming ?? 0}
            </span>
          </div>
        </div>
        <div className="tdl-summary-col">
          <div className="tdl-scard-simple">
            <span className="tdl-label">Completed Today</span>
            <span className="tdl-value">
              {todoData?.summary?.completed_today ?? 0}
            </span>
          </div>
        </div>
        <div className="tdl-summary-col">
          <div className="tdl-scard-simple">
            <span className="tdl-label">Completed Upcoming</span>
            <span className="tdl-value">
              {todoData?.summary?.completed_upcoming ?? 0}
            </span>
          </div>
        </div>
      </div>

      <div className="tdl-actions-wrap">
        <div className="tdl-actions-head-card">
          <h3 className="tdl-actions-title">
            {todoData?.summary?.today ?? 0} actions due today
          </h3>
          <div className="tdl-actions-card-rainbow">
            <div className="tdl-actions-card-inner">
              <div className="tdl-actions-icon-col">
                <img
                  src={accountIcon}
                  alt="Actions icon"
                  className="tdl-actions-icon"
                />
              </div>
              <div className="tdl-actions-text-grid">
                {(todoData?.tasks || [])
                  .filter(
                    (t) =>
                      t.due_date === new Date().toISOString().split("T")[0],
                  )
                  .map((t, i) => (
                    <p className="tdl-action-item" key={t.id}>
                      <span className="tdl-action-num">{i + 1}.</span>{" "}
                      {t.task_title}
                    </p>
                  ))}
              </div>
            </div>
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
          <select
            className="tdl-dd-select"
            value={accountFilter}
            onChange={(e) => setAccountFilter(e.target.value)}
          >
            <option value="all">Account</option>
            <option value="acme">Acme Corp Global</option>
            <option value="techstart">TechStart Health</option>
            <option value="modern">Modern Logistics</option>
          </select>
          <span className="tdl-dd-icon">
            <i className="bi bi-caret-down-fill"></i>
          </span>
        </div>

        <div className="tdl-dd-wrap">
          <select
            className="tdl-dd-select"
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
          >
            <option value="all">All Types</option>
            <option value="outreach">Outreach</option>
            <option value="document">Document</option>
            <option value="action">Action</option>
          </select>
          <span className="tdl-dd-icon">
            <i className="bi bi-caret-down-fill"></i>
          </span>
        </div>

        <div className="tdl-dd-wrap">
          <select
            className="tdl-dd-select"
            value={dueDateFilter}
            onChange={(e) => setDueDateFilter(e.target.value)}
          >
            <option value="all">Due Date</option>
            <option value="today">Today</option>
            <option value="this-week">This Week</option>
            <option value="overdue">Overdue</option>
          </select>
          <span className="tdl-dd-icon">
            <i className="bi bi-caret-down-fill"></i>
          </span>
        </div>

        <div className="tdl-dd-wrap">
          <select
            className="tdl-dd-select"
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
          >
            <option value="all">Priority</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <span className="tdl-dd-icon">
            <i className="bi bi-caret-down-fill"></i>
          </span>
        </div>

        <button className="tdl-btn-dl" type="button" aria-label="Download">
          <i className="bi bi-download"></i>
        </button>
      </div>

      <div className="sd-meetings-panel">
        <div className="sd-meetings-header" style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <h2 className="sd-meetings-title" style={{ margin: 0 }}>To Do List</h2>
          <span className="sd-priority-meta" style={{ margin: 0 }}>
            {todoData?.tasks?.length ?? 0} total action &middot;{" "}
            <span className="sd-priority-due-today">
              {todoData?.summary?.today ?? 0} due today
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

        {showAddForm ? (
          <div className="tdl-add-action-form">
            <div className="ct-new-form">
              <div className="ct-form-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div className="ct-form-title">{editingTask ? "Edit Action" : "Add Action"}</div>
                <button type="button" style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b", fontSize: 20 }} onClick={() => { setShowAddForm(false); setEditingTask(null); setFormData(initialFormData); }}>
                  <i className="ri-close-line"></i>
                </button>
              </div>

              <div className="ct-fields">
                <div className="ct-field">
                  <label>Title</label>
                  <input
                    type="text"
                    name="title"
                    value={formData.title}
                    onChange={handleFormChange}
                    placeholder="Enter action title"
                  />
                </div>

                <div className="ct-field">
                  <label>Type</label>
                  <div className="ct-select-wrap">
                    <select
                      name="type"
                      value={formData.type}
                      onChange={handleFormChange}
                    >
                      <option value="">Select Type</option>
                      <option value="outreach">Outreach</option>
                      <option value="document">Document</option>
                      <option value="action">Action</option>
                    </select>
                    <span className="tdl-dd-icon">
                      <i className="bi bi-caret-down-fill"></i>
                    </span>
                  </div>
                </div>
                <div className="ct-field">
                  <label>Priority</label>
                  <div className="ct-select-wrap">
                    <select
                      name="priority"
                      value={formData.priority}
                      onChange={handleFormChange}
                    >
                      <option value="">Select Priority</option>
                      <option value="High">High</option>
                      <option value="Medium">Medium</option>
                      <option value="Low">Low</option>
                    </select>
                    <span className="tdl-dd-icon">
                      <i className="bi bi-caret-down-fill"></i>
                    </span>
                  </div>
                </div>
                <div className="ct-field">
                  <label>Linked Account</label>
                  <input
                    type="text"
                    name="linkedAccount"
                    value={formData.linkedAccount}
                    onChange={handleFormChange}
                    placeholder="Search account"
                  />
                </div>
                <div className="ct-field">
                  <label>Linked Opportunity</label>
                  <input
                    type="text"
                    name="linkedOpportunity"
                    value={formData.linkedOpportunity}
                    onChange={handleFormChange}
                    placeholder="Search opportunity"
                  />
                </div>
                <div className="ct-field">
                  <label>Due Date</label>
                  <div className="ct-date-input-wrap">
                    <input
                      type="text"
                      name="dueDate"
                      value={formData.dueDate}
                      onChange={handleFormChange}
                      placeholder="MM/DD/YYYY"
                    />
                    <span className="ct-date-icon">
                      <i className="bi bi-calendar"></i>
                    </span>
                  </div>
                </div>

                <div className="ct-field ct-field-full">
                  <label>Add</label>
                  <div className="ct-textarea-wrap">
                    <textarea
                      name="addText"
                      value={formData.addText}
                      onChange={handleFormChange}
                      placeholder="Type here"
                      maxLength={1000}
                    />
                    <div className="ct-text-count">
                      {formData.addText.length}/1000
                    </div>
                  </div>
                </div>

                <button
                  className="ct-btn-cancel"
                  type="button"
                  onClick={() => { setShowAddForm(false); setEditingTask(null); setFormData(initialFormData); }}
                >
                  Cancel
                </button>
                <button
                  className="ct-btn-save"
                  type="button"
                  onClick={editingTask ? handleFormUpdate : handleFormSave}
                >
                  {editingTask ? "Update" : "Add"}
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {loading && (
              <p style={{ padding: 16, color: "#64748b" }}>Loading...</p>
            )}
            {!loading && (!todoData?.tasks || todoData.tasks.length === 0) && (
              <p style={{ padding: 16, color: "#64748b" }}>No data available</p>
            )}
            {!loading &&
              (todoData?.tasks || []).map((task) => {
                const today = new Date().toISOString().split("T")[0];
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
                          <div style={{ fontSize: 13, color: "#64748b", marginTop: 4 }}>
                            {task.source_label || "—"}
                          </div>
                          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                            {task.type_tag && (
                              <span className="sd-content-badge sd-content-badge-outreach">
                                {task.type_tag.charAt(0).toUpperCase() + task.type_tag.slice(1)}
                              </span>
                            )}
                            {task.priority && (
                              <span className={`sd-content-badge sd-content-badge-priority-${task.priority}`}>
                                {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
                              </span>
                            )}
                            {task.due_date && (
                              <span style={{ fontSize: 12, color: "#64748b", alignSelf: "center" }}>Due: {task.due_date}</span>
                            )}
                          </div>
                          {!isCompleted && (
                          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                            <button type="button" className="kp-btn-cancel" onClick={async () => {
                              try {
                                await updateTodoStatus(task.id, "Completed");
                                loadTodos();
                              } catch (err) {
                                console.error("Failed to complete task:", err);
                              }
                            }}>
                              <i className="ri-checkbox-circle-line" style={{ fontSize: 16, marginRight: 6 }}></i>
                              <span>Complete Task</span>
                            </button>
                            <button type="button" className="sd-draft-btn">
                              <i className="ri-mail-send-line" style={{ fontSize: 16, marginRight: 6 }}></i>
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
                        <button
                          className="sd-badge-edit-btn"
                          type="button"
                          aria-label="Edit action"
                          onClick={() => handleEditClick(task)}
                        >
                          <i className="bi bi-pencil"></i>
                        </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </div>

      <Modal show={showSuccessModal} backdrop="static" keyboard={false} centered>
        <Modal.Body style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <i className="ri-checkbox-circle-fill" style={{ color: "#047857", fontSize: "20px" }}></i>
          <span>{successMessage}</span>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button className="kp-btn-update" onClick={() => { setShowSuccessModal(false); loadTodos(); }}>OK</button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

export default TodoList;
