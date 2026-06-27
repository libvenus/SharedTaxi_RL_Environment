import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchQuarterPulseQuota, updateQuarterPulseQuota } from "../../api/client";
import { Modal } from "react-bootstrap";
import "../../styles/execute.css";
import "../../styles/admin.css";

const SELLER_ID = "055DAFE7-9840-451D-8328-5F70A6326C03";

export default function QuarterPulse() {
  const navigate = useNavigate();
  const [quotaAmount, setQuotaAmount] = useState("");
  const [currencyCode, setCurrencyCode] = useState("USD");
  const [fiscalYear, setFiscalYear] = useState(0);
  const [fiscalQuarter, setFiscalQuarter] = useState(1);
  const [loading, setLoading] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    window.scrollTo(0, 0);
    fetchQuarterPulseQuota(SELLER_ID)
      .then((res) => {
        setQuotaAmount(res.quotaAmount ?? "");
        setCurrencyCode(res.currencyCode || "USD");
        setFiscalYear(res.fiscalYear ?? 0);
        setFiscalQuarter(res.fiscalQuarter ?? 1);
      })
      .catch((err) => console.error("Failed to fetch quota:", err));
  }, []);

  const handleSave = async () => {
    try {
      setLoading(true);
      const payload = {
        quotaAmount: Number(quotaAmount),
        fiscalYear,
        fiscalQuarter,
        currencyCode,
      };
      const res = await updateQuarterPulseQuota(SELLER_ID, payload);
      if (res) {
        setMessage("Quota updated successfully");
        setShowSuccessModal(true);
        if (res.quotaAmount !== undefined) setQuotaAmount(res.quotaAmount);
      }
    } catch (error) {
      console.error("Update error:", error);
      setMessage("Failed to update quota");
      setShowSuccessModal(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dv-page">
      <div className="main">
        <div className="quarter-bar">
          Q2 FY2024 · Week 10 of 12 ·{" "}
          <span className="closure-badge">Closure Phase</span>
        </div>

        <div className="page-header">
          <div className="page-header-row">
            <span className="topbar-title">Admin Settings</span>
          </div>
          <div className="tabs-outer">
            <div className="tabs">
              <button
                className="tab"
                onClick={() => navigate("/admin-settings")}
              >
                Dead Letters
              </button>
              <button className="tab active">
                Quarter Pulse
              </button>
            </div>
          </div>
        </div>

        <div className="content">
          <div className="admin-col-left" style={{ width: "100%" }}>
            <div className="row w-100 g-3">
              <div className="col-12 col-lg-4">
                <div className="admin-right-panel">
                  <div className="admin-right-header">
                    <h2 className="admin-right-title">Quarter Pulse</h2>
                  </div>
                  <p className="admin-panel-subtitle">Configure the metrics for the quarter</p>

                  <div className="admin-form-inner">
                    <div className="admin-form-field">
                      <label>Quota for the Quarter</label>
                      <input
                        type="number"
                        placeholder="e.g. 2500000"
                        className="admin-form-input"
                        value={quotaAmount}
                        onChange={(e) => setQuotaAmount(e.target.value)}
                      />
                    </div>
                    <div className="admin-form-field">
                      <label>Pipeline Coverage Ratio expectation</label>
                      <input
                        type="text"
                        placeholder=""
                        className="admin-form-input"
                        value={currencyCode}
                        onChange={(e) => setCurrencyCode(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="admin-form-actions" style={{ marginTop: 18 }}>
                    <button className="admin-btn-cancel" onClick={() => navigate("/admin-settings")}>Cancel</button>
                    <button className="admin-btn-save" onClick={handleSave} disabled={loading}>
                      {loading ? "Saving..." : "Save"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <Modal
          show={showSuccessModal}
          backdrop="static"
          keyboard={false}
          centered
        >
          <Modal.Body
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <i
              className="ri-checkbox-circle-fill"
              style={{ color: "#047857", fontSize: "20px" }}
            ></i>
            <span>{message}</span>
          </Modal.Body>
          <Modal.Footer style={{ border: "none" }}>
            <button
              className="ct-btn-save"
              onClick={() => setShowSuccessModal(false)}
            >
              OK
            </button>
          </Modal.Footer>
        </Modal>
      </div>
    </div>
  );
}
