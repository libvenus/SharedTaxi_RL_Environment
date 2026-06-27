import { useState, useEffect } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import "../../../styles/execute.css";
import webex from "../../../assets/image 3.png";
import googleMeetLogo from "../../../assets/logos_google-meet.png";
import avatar1 from "../../../assets/avatar1.jpg";
import accounticon from "../../../assets/icons/account_icon.png";
import NextSteps from "./NextSteps";
import CrmUpdates from "./CrmUpdates";
import { fetchPostMeetingDetail, addKeyPoint, updateKeyPoint, deleteKeyPoint, completeKeyPoints, completeNextSteps } from "../../../api/client";
import { Modal } from "react-bootstrap";

export default function KeyPoints() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [searchParams] = useSearchParams();
  const viewOnly = searchParams.get("viewOnly") === "true";
  const [meeting, setMeeting] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [editingKeyPointId, setEditingKeyPointId] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deletingKeyPointId, setDeletingKeyPointId] = useState(null);
  const [newKeyPoint, setNewKeyPoint] = useState({
    point: "",
    confidence: "High",
    isAmbiguous: true,
  });

  useEffect(() => {
    if (!id) return;
    fetchPostMeetingDetail(id)
      .then((data) => setMeeting(data))
      .catch((err) => console.error("Failed to fetch meeting detail:", err));
  }, [id]);

  const handleAddKeyPoint = async () => {
    try {
      let res;
      if (editingKeyPointId) {
        res = await updateKeyPoint(id, editingKeyPointId, newKeyPoint);
      } else {
        res = await addKeyPoint(id, newKeyPoint);
      }
      setMeeting((prev) => ({ ...prev, key_points: res.key_points }));
      setSuccessMessage(res.message || (editingKeyPointId ? "Key point updated successfully" : "Key point added successfully"));
      setShowAddModal(false);
      setEditingKeyPointId(null);
      setNewKeyPoint({ id: "", point: "", confidence: "High", transcriptRefs: [], isAmbiguous: false });
      setShowSuccessModal(true);
    } catch (err) {
      console.error("Failed to save key point:", err);
    }
  };

  const handleEditKeyPoint = (kp) => {
    setEditingKeyPointId(kp.id);
    setNewKeyPoint({
      id: kp.id,
      point: kp.point,
      confidence: kp.confidence || "High",
      transcriptRefs: kp.transcriptRefs || [],
      isAmbiguous: kp.isAmbiguous || false,
    });
    setShowAddModal(true);
  };

  const handleDeleteKeyPoint = async (keyPointId) => {
    try {
      const res = await deleteKeyPoint(id, keyPointId);
      setMeeting((prev) => ({
        ...prev,
        key_points: (prev.key_points || []).filter((kp) => kp.id !== keyPointId),
      }));
      setShowDeleteModal(false);
      setConfirmDelete(false);
      setDeletingKeyPointId(null);
      setSuccessMessage(res?.message || "Key point deleted successfully");
      setShowSuccessModal(true);
    } catch (err) {
      console.error("Failed to delete key point:", err);
    }
  };

  const platform = meeting?.meeting_platform || "";
  const isGoogle = platform.toLowerCase().includes("google");
  const platformIcon = isGoogle ? googleMeetLogo : webex;
  const platformLabel = platform.replace(/^(Internal|External)\s*/i, "").trim() || "Webex";
  const typeLabel = platform.toLowerCase().includes("external") ? "External" : "Internal";

  return (
    <>
      <div className="exec-page-header">
        <div className="exec-page-header-row p-2" style={{ borderBottom: "1px solid #E2E8F0" }}>
          <span className="exec-topbar-title">Execute: Post-Meeting Review</span>
          <div className="exec-topbar-actions">
            <button className="exec-btn-prep">Settings</button>
          </div>
        </div>

        <div className="exec-page-header-row" style={{ paddingTop: 0 }}>
          <button className="exec-back-btn" onClick={() => navigate("/execute")}>
            ‹
          </button>
          <span style={{ fontSize: 18, fontWeight: 700, color: "#0F172A" }}>{meeting?.title || ""}</span>
          <div className="exec-topbar-actions">
            <div className="mr-meeting-meta">
              <img alt={platformLabel} className="sd-platform-icon sd-platform-meet" src={platformIcon} />
              <span className="mr-meta-text">{typeLabel} · {platformLabel} &nbsp; {meeting?.meeting_duration || ""}</span>
              <div className="sd-meeting-avatars mb-0">
                <div className="sd-avatar-group">
                  {(meeting?.attendees || []).map((a, i) => (
                    <div key={i} className="sd-avatar sd-avatar-initials-ra">
                      {a.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
                    </div>
                  ))}
                </div>
              </div>
              {meeting?.meeting_type && (
                <span className={`mr-badge mr-badge-${meeting.meeting_type.toLowerCase()}`}>
                  {meeting.meeting_type}
                </span>
              )}
            </div>
          </div>
        </div>

      </div>

      {/* Two Column Layout */}
      <div className="kp-main-content">
        {/* Left Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Key Points From The Call */}
          <div className="kp-panel">
            <>
              <div className="kp-panel-title">Key Points From The Call</div>

              {/* Inline Add/Edit Key Point Form */}
              {showAddModal && (
                <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10, padding: "20px 24px", marginBottom: 16, display: "flex", flexDirection: "column", gap: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{editingKeyPointId ? "Edit Key Point" : "Add Key Point"}</span>
                    <button style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b" }} onClick={() => { setShowAddModal(false); setEditingKeyPointId(null); }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
                      <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Confidence</label>
                      <select style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={newKeyPoint.confidence} onChange={(e) => setNewKeyPoint({ ...newKeyPoint, confidence: e.target.value })}>
                        <option value="High">High</option>
                        <option value="Medium">Medium</option>
                        <option value="Low">Low</option>
                      </select>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: "0 0 calc(50% - 8px)", minWidth: 160 }}>
                      <label style={{ fontWeight: 600, fontSize: 13, color: "#0f172a" }}>Key Point Description</label>
                      <input type="text" style={{ background: "#f8fafc", border: "2px solid #cbd5e1", borderRadius: 4, height: 38, padding: "0 10px", fontSize: 13, color: "#475569", width: "100%", outline: "none" }} value={newKeyPoint.point} onChange={(e) => setNewKeyPoint({ ...newKeyPoint, point: e.target.value })} placeholder="Enter key point" />
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button className="kp-btn-cancel" onClick={() => { setShowAddModal(false); setEditingKeyPointId(null); }}>Cancel</button>
                    <button className="kp-btn-update" onClick={handleAddKeyPoint} disabled={!newKeyPoint.point}>{editingKeyPointId ? "Update" : "Add"}</button>
                  </div>
                </div>
              )}

              {!showAddModal && (
              <>
              {(meeting?.key_points || []).map((kp, index) => (
                <div className="key-point" key={kp.id}>
                  <div className="kp-number">{index + 1}</div>
                  <div className="kp-text">{kp.point}</div>
                  <span className={`priority priority-${(kp.confidence || "").toLowerCase()}`}>{kp.confidence}</span>
                  {!viewOnly && (
                  <div className="kp-actions">
                    <button className="kp-icon-btn" title="Edit" onClick={() => handleEditKeyPoint(kp)}><i className="ri-pencil-line" style={{ fontSize: 18 }}></i></button>
                    <button className="kp-icon-btn delete" title="Delete" onClick={() => { setDeletingKeyPointId(kp.id); setShowDeleteModal(true); }}><i className="ri-delete-bin-line" style={{ fontSize: 18 }}></i></button>
                  </div>
                  )}
                </div>
              ))}
              {!viewOnly && (
              <div className="add-point" onClick={() => { setEditingKeyPointId(null); setNewKeyPoint({ id: "", point: "", confidence: "High", transcriptRefs: [], isAmbiguous: false }); setShowAddModal(true); }} style={{ cursor: "pointer" }}>
                <div className="add-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg></div>
                Add a key point
              </div>
              )}
              </>
              )}
              {!viewOnly && (
              <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
                <button className="kp-btn-cancel" onClick={() => navigate("/execute")}>Cancel</button>
                <button className="kp-btn-update" onClick={async () => {
                  try {
                    await completeKeyPoints(id);
                    setSuccessMessage("Key points completed successfully");
                    setShowSuccessModal(true);
                  } catch (err) {
                    console.error("Failed to complete:", err);
                  }
                }}>Update</button>
              </div>
              )}
            </>
          </div>

          {/* CRM Updates From The Call */}
          <div className="kp-panel">
            <CrmUpdates viewOnly={viewOnly} crmUpdates={meeting?.crm_updates} meetingId={id} summary={meeting?.summary} />
          </div>

          {/* AI-Generated Meeting Notes */}
          {/* {meeting?.summary && (
            <div className="kp-panel">
              <div className="kp-panel-title">AI-Generated Meeting Notes</div>
              <div style={{ fontSize: 13, color: "#334155", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
                {meeting.summary}
              </div>
            </div>
          )} */}

        </div>

        {/* Right Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Call Transcript */}
          <div className="kp-panel">
            <div className="kp-panel-title">Call Transcript</div>
            <div className="transcript-box">
              <div className="transcript-header">
                <span className="ai-icon">
                  <img src={accounticon} alt="AI" style={{ width: 18, height: 18 }} />
                </span>
                <span className="transcript-label">Transcribed by Ai</span>
              </div>
              <div className="transcript-body" style={{ whiteSpace: "pre-wrap" }}>
                {meeting?.call_transcript?.transcriptSegments
                  ? meeting.call_transcript.transcriptSegments.map((seg) => (
                      <div key={seg.id} style={{ marginBottom: 12 }}>
                        <strong>{seg.speakerName}</strong> <span style={{ color: "#64748b", fontSize: 12 }}>({seg.startTimestamp} - {seg.endTimestamp})</span>
                        <div>{seg.text}</div>
                      </div>
                    ))
                  : (
                    <>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:20 PM] Amit Sharma (Lenovo):</strong>
                        <div>Good evening, everyone. Thanks for joining. Brian, Sarah, Tom — appreciate your time for this final technical review. We're excited to support Ford's manufacturing expansion. We understand the scope is now adjusted to 1,400 units, and today we'll align on technical fit, delivery sequencing, and next steps.</div>
                      </div>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:22 PM] Brian Novak (Ford):</strong>
                        <div>Thanks, Amit. The urgency is high with the production line launch on August 1. We need confidence in delivery timelines and overall execution. I want to ensure alignment on support, delivery, and commercial model.</div>
                      </div>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:24 PM] Sarah Chen (Ford):</strong>
                        <div>From procurement and finance perspective, the revised scope helps. Finance is strongly considering an opex model instead of upfront CAPEX, so we need clarity on that structure.</div>
                      </div>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:25 PM] Tom Rodriguez (Ford):</strong>
                        <div>My concern is operational readiness. Any delay in delivery impacts the plant rollout. I want clarity on the staggered delivery plan and deployment feasibility.</div>
                      </div>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:27 PM] Amit Sharma (Lenovo):</strong>
                        <div>We recommend Lenovo DaaS — no upfront CAPEX, predictable monthly cost, and a 36-month refresh cycle. This aligns well with finance expectations.</div>
                      </div>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:29 PM] Sarah Chen (Ford):</strong>
                        <div>That works conceptually. We'll need transparency on total cost and included services, no hidden costs.</div>
                      </div>
                      <div style={{ marginBottom: 12 }}>
                        <strong>[5:30 PM] Amit Sharma (Lenovo):</strong>
                        <div>Understood. Support is included via ProSupport+, which offers:{"\n"}1-business-day onsite response{"\n"}Accidental damage protection{"\n"}99.9% uptime SLA{"\n"}Dell does not match this at the same price tier.</div>
                      </div>
                    </>
                  )}
              </div>
            </div>
          </div>

          {/* Next Steps From The Call */}
          <div className="kp-panel">
            <NextSteps nextSteps={meeting?.next_steps || []} viewOnly={viewOnly} onUpdated={(res) => { if (res?.key_points) setMeeting((prev) => ({ ...prev, next_steps: res.key_points })); }} />
            {!viewOnly && (
            <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
              <button className="kp-btn-cancel" onClick={() => navigate("/execute")}>Cancel</button>
              <button className="kp-btn-update" onClick={async () => {
                try {
                  await completeNextSteps(id);
                  setSuccessMessage("Next steps completed successfully");
                  setShowSuccessModal(true);
                } catch (err) {
                  console.error("Failed to complete:", err);
                }
              }}>Update</button>
            </div>
            )}
          </div>
        </div>
      </div>


      {/* Delete Confirmation Modal */}
      <Modal show={showDeleteModal} onHide={() => { setShowDeleteModal(false); setConfirmDelete(false); }} centered>
        <Modal.Header>
          <Modal.Title style={{ fontSize: "18px" }}>
            <i className="ri-information-fill" style={{ color: "#B91C1C", fontSize: "20px", marginRight: "10px" }}></i>
            Delete Key Point
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to delete this key point? This action cannot be undone.</p>
          <div className="form-check mt-3">
            <input
              className="form-check-input"
              type="checkbox"
              id="confirmDeleteKP"
              checked={confirmDelete}
              onChange={(e) => setConfirmDelete(e.target.checked)}
            />
            <label className="form-check-label" htmlFor="confirmDeleteKP" style={{ fontWeight: 600 }}>
              I want to delete this key point.
            </label>
          </div>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button className="kp-btn-cancel" onClick={() => { setShowDeleteModal(false); setConfirmDelete(false); }}>Cancel</button>
          <button
            className="kp-btn-update"
            onClick={() => handleDeleteKeyPoint(deletingKeyPointId)}
            disabled={!confirmDelete}
            style={{ opacity: !confirmDelete ? 0.5 : 1, cursor: !confirmDelete ? "not-allowed" : "pointer" }}
          >
            Delete
          </button>
        </Modal.Footer>
      </Modal>

      {/* Success Modal */}
      <Modal show={showSuccessModal} backdrop="static" keyboard={false} centered>
        <Modal.Body style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <i className="ri-checkbox-circle-fill" style={{ color: "#047857", fontSize: "20px" }}></i>
          <span>{successMessage}</span>
        </Modal.Body>
        <Modal.Footer style={{ border: "none" }}>
          <button className="kp-btn-update" onClick={() => setShowSuccessModal(false)}>OK</button>
        </Modal.Footer>
      </Modal>
    </>
  );
}