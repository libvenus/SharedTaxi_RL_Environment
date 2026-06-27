import { useState } from "react";
import { useParams } from "react-router-dom";
import { Modal } from "react-bootstrap";
import { updateNextStep, deleteNextStep } from "../../../api/client";

export default function NextSteps({ nextSteps = [], viewOnly = false, onUpdated }) {
  const { id: meetingId } = useParams();
  const [editingStep, setEditingStep] = useState(null);
  const [formData, setFormData] = useState({ task: "", owner: "", dueDate: "", status: "Pending", confidence: "High", transcriptRefs: [] });
  const [saving, setSaving] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deletingStepId, setDeletingStepId] = useState(null);

  const handleEdit = (step) => {
    setEditingStep(step.id);
    setFormData({
      task: step.task || "",
      owner: step.owner || "",
      dueDate: step.dueDate || "",
      status: step.status || "Pending",
      confidence: step.confidence || "High",
      transcriptRefs: step.transcriptRefs || [],
    });
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const res = await updateNextStep(meetingId, editingStep, formData);
      setEditingStep(null);
      setSuccessMessage(res.message || "Next step updated successfully");
      setShowSuccessModal(true);
      if (onUpdated) onUpdated(res);
    } catch (err) {
      console.error("Failed to update next step:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteStep = async (stepId) => {
    try {
      const res = await deleteNextStep(meetingId, stepId);
      setShowDeleteModal(false);
      setConfirmDelete(false);
      setDeletingStepId(null);
      setSuccessMessage(res?.message || "Next step deleted successfully");
      setShowSuccessModal(true);
      if (onUpdated) onUpdated(res);
    } catch (err) {
      console.error("Failed to delete next step:", err);
    }
  };

  return (
    <>
    <div className="">
      <div className="kp-panel-title">Confirmed Next Steps</div>

      {editingStep && (
        <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10, padding: "20px 24px", marginBottom: 16, display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>Edit Next Step</span>
            <button style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b" }} onClick={() => setEditingStep(null)}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 100%" }}>
              <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Task</label>
              <textarea style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, padding: 10, fontSize: 13, color: "#475569", width: "100%", outline: "none", resize: "vertical" }} rows={3} value={formData.task} onChange={(e) => setFormData({ ...formData, task: e.target.value })} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
              <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Owner</label>
              <input type="text" style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={formData.owner} onChange={(e) => setFormData({ ...formData, owner: e.target.value })} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
              <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Due Date</label>
              <input type="text" style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={formData.dueDate} onChange={(e) => setFormData({ ...formData, dueDate: e.target.value })} placeholder="e.g. After the call" />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
              <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Status</label>
              <select style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={formData.status} onChange={(e) => setFormData({ ...formData, status: e.target.value })}>
                <option value="Pending">Pending</option>
                <option value="In Progress">In Progress</option>
                <option value="Completed">Completed</option>
              </select>
            </div>
            {/* <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
              <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Confidence</label>
              <select style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={formData.confidence} onChange={(e) => setFormData({ ...formData, confidence: e.target.value })}>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
            </div> */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
              <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Transcript Refs (comma separated)</label>
              <input type="text" style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={formData.transcriptRefs.join(", ")} onChange={(e) => setFormData({ ...formData, transcriptRefs: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} placeholder="e.g. T001, T002" />
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="kp-btn-cancel" onClick={() => setEditingStep(null)}>Cancel</button>
            <button className="kp-btn-update" onClick={handleSave} disabled={saving || !formData.task}>{saving ? "Saving..." : "Update"}</button>
          </div>
        </div>
      )}

      {!editingStep && (
      <div className="steps-wrapper">
        {nextSteps.map((step, index) => (
          <div className="step-row" key={step.id}>
            <div className="timeline-col">
              <div className={`step-circle${index === 0 ? " active" : ""}`}></div>
            </div>
            <div className="step-content" style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
              <div className="step-body">
                <span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%", fontWeight: 700, fontSize: 15 }} title={step.task_title || step.task}>{step.task_title || step.task}</span>
                <div className="step-meta">
                  {step.dueDate ? `Due: ${step.dueDate}` : ""} &nbsp; {step.owner ? `Owner: ${step.owner}` : ""}
                </div>
              </div>
              <div className="step-actions" style={{ width: "100%", justifyContent: "flex-start", gap: 8 }}>
                {/* <span className={`priority priority-${(step.confidence || "").toLowerCase()}`}>{step.confidence}</span> */}
                <span className={`status-${(step.status || "pending").toLowerCase()}`}>{step.status || "Pending"}</span>
                <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
                {!viewOnly && (
                <>
                <button className="kp-icon-btn" title="Edit" onClick={() => handleEdit(step)}>
                  <i className="ri-pencil-line" style={{ fontSize: 18 }}></i>
                </button>
                <button className="kp-icon-btn delete" title="Delete" onClick={() => { setDeletingStepId(step.id); setShowDeleteModal(true); }}>
                  <i className="ri-delete-bin-line" style={{ fontSize: 18 }}></i>
                </button>
                </>
                )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      )}
    </div>

      <Modal show={showSuccessModal} backdrop="static" keyboard={false} centered>
        <Modal.Body style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <i className="ri-checkbox-circle-fill" style={{ color: "#047857", fontSize: "20px" }}></i>
          <span>{successMessage}</span>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button className="kp-btn-update" onClick={() => setShowSuccessModal(false)}>OK</button>
        </Modal.Footer>
      </Modal>

      <Modal show={showDeleteModal} onHide={() => { setShowDeleteModal(false); setConfirmDelete(false); }} centered>
        <Modal.Header>
          <Modal.Title style={{ fontSize: "18px" }}>
            <i className="ri-information-fill" style={{ color: "#B91C1C", fontSize: "20px", marginRight: "10px" }}></i>
            Delete Next Step
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to delete this next step? This action cannot be undone.</p>
          <div className="form-check mt-3">
            <input className="form-check-input" type="checkbox" id="confirmDeleteNS" checked={confirmDelete} onChange={(e) => setConfirmDelete(e.target.checked)} />
            <label className="form-check-label" htmlFor="confirmDeleteNS" style={{ fontWeight: 600 }}>I want to delete this next step.</label>
          </div>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button className="kp-btn-cancel" onClick={() => { setShowDeleteModal(false); setConfirmDelete(false); }}>Cancel</button>
          <button className="kp-btn-update" onClick={() => handleDeleteStep(deletingStepId)} disabled={!confirmDelete} style={{ opacity: !confirmDelete ? 0.5 : 1, cursor: !confirmDelete ? "not-allowed" : "pointer" }}>Delete</button>
        </Modal.Footer>
      </Modal>
    </>
  );
}