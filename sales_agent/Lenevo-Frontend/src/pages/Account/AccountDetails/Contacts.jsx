import { useState, useEffect } from "react";
import {
  fetchAccountContactDetails,
  addAccountContact,
  updateAccountContact,
  deleteAccountContact,
} from "../../../api/client";
import { useParams, useNavigate } from "react-router-dom";
import "../../../styles/account.css";
import { Modal } from "react-bootstrap";
import downloadIcon from "../../../assets/download.png";
export default function Contacts() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [showNewContact, setShowNewContact] = useState(false);
  const [editingContact, setEditingContact] = useState(null);
  const [contacts, setContacts] = useState([]);
  const [viewMode, setViewMode] = useState("account");
  const [selectedContact, setSelectedContact] = useState(null);
  const [isPrimary, setIsPrimary] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [role, setRole] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [phone, setPhone] = useState("");
  const [email, setEmail] = useState("");
  const [popup, setPopup] = useState({
    open: false,
    message: "",
    success: true,
  });

  const [loading, setLoading] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [originalContact, setOriginalContact] = useState(null);

  const [message, setMessage] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [searchContact, setSearchContact] = useState("");
  const [roleFilter, setRoleFilter] = useState("");
  const [selectedRows, setSelectedRows] = useState([]);
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const phoneRegex = /^\+91\d{10}$/;

  const isEmailValid = emailRegex.test(email.trim());
  const isPhoneValid = phoneRegex.test(phone.trim());
  const isFormValid =
    firstName.trim() &&
    lastName.trim() &&
    role.trim() &&
    email.trim() &&
    phone.trim() &&
    isEmailValid &&
    isPhoneValid;

  const handleCardClick = (contact) => {
    setEditingContact(contact);
    setOriginalContact(contact);
    setFirstName(contact.firstName || "");
    setLastName(contact.lastName || "");
    setEmail(contact.email || "");
    setPhone(contact.phone || "");
    setJobTitle(contact.jobTitle || "");
    setRole(contact.role || "");
    setIsPrimary(contact.isPrimary || false);
    setShowNewContact(false);
  };
  const hasChanges =
    firstName !== (originalContact?.firstName || "") ||
    lastName !== (originalContact?.lastName || "") ||
    role !== (originalContact?.role || "") ||
    phone !== (originalContact?.phone || "") ||
    email !== (originalContact?.email || "") ||
    isPrimary !== (originalContact?.isPrimary || false);

  const isUpdateDisabled =
    loading || !isFormValid || !isEmailValid || !isPhoneValid || !hasChanges;
  const resetForm = () => {
    setFirstName("");
    setLastName("");
    setRole("");
    setPhone("");
    setEmail("");
    setJobTitle("");
    setIsPrimary(false);
  };

  const handleCloseEdit = () => {
    setShowSuccessModal(false);
    setEditingContact(null);
    resetForm();
  };

  const handleNewContact = () => {
    setShowNewContact(true);
    setEditingContact(null);
    resetForm();
  };
  const loadContactsdata = async () => {
    try {
      const response = await fetchAccountContactDetails(id);

      const allContacts = [
        ...(response?.primary ? [response.primary] : []),
        ...(response?.others || []),
      ];

      setContacts(allContacts);

      // Optional: select primary contact by default
      if (response?.primary) {
        setSelectedContact(response.primary);
      }
    } catch (error) {
      console.error("Failed to fetch contacts:", error);
      setContacts([]);
    }
  };
  useEffect(() => {
    const loadContacts = async () => {
      try {
        const response = await fetchAccountContactDetails(id);

        const allContacts = [
          ...(response?.primary ? [response.primary] : []),
          ...(response?.others || []),
        ];

        setContacts(allContacts);

        // Optional: select primary contact by default
        if (response?.primary) {
          setSelectedContact(response.primary);
        }
      } catch (error) {
        console.error("Failed to fetch contacts:", error);
        setContacts([]);
      }
    };

    if (id) {
      loadContacts();
    }
  }, [id]);

  const getDisplayName = (contact) => {
    const fallbackName = [contact?.firstName, contact?.lastName]
      .filter(Boolean)
      .join(" ")
      .trim();
    return (contact?.name || fallbackName || "-").trim();
  };

  const getInitials = (contact) => {
    const displayName = getDisplayName(contact);
    const parts = displayName.split(/\s+/).filter(Boolean);
    if (!parts.length || displayName === "-") return "?";
    return parts.slice(0, 2).map((part) => part[0].toUpperCase()).join("");
  };

  const getAvatarUrl = (contact) =>
    contact?.image ||
    contact?.avatar ||
    contact?.profileImage ||
    contact?.photoUrl ||
    "";

  const getRoleText = (contact) => contact?.role || contact?.jobTitle || "-";

  const getRowId = (contact) =>
    String(contact?.id ?? `${getDisplayName(contact)}-${contact?.email || ""}`);

  const normalizedSearch = searchContact.trim().toLowerCase();
  const filteredContacts = contacts.filter((contact) => {
    const roleText = getRoleText(contact);
    const matchesRole = !roleFilter || roleText === roleFilter;
    const matchesSearch =
      !normalizedSearch ||
      getDisplayName(contact).toLowerCase().includes(normalizedSearch) ||
      (contact?.email || "").toLowerCase().includes(normalizedSearch) ||
      String(contact?.phone || "").toLowerCase().includes(normalizedSearch);
    return matchesRole && matchesSearch;
  });

  const roleOptions = [...new Set(contacts.map((contact) => getRoleText(contact)))].filter(
    (roleText) => roleText && roleText !== "-",
  );

  const areAllVisibleSelected =
    filteredContacts.length > 0 &&
    filteredContacts.every((contact) => selectedRows.includes(getRowId(contact)));

  const handleToggleSelectAll = (checked) => {
    if (checked) {
      setSelectedRows(filteredContacts.map((contact) => getRowId(contact)));
      return;
    }
    setSelectedRows([]);
  };

  const handleToggleRow = (contact, checked) => {
    const rowId = getRowId(contact);
    setSelectedRows((prev) => {
      if (checked) {
        return prev.includes(rowId) ? prev : [...prev, rowId];
      }
      return prev.filter((idValue) => idValue !== rowId);
    });
  };

  const handleDownloadContacts = () => {
    const csvHeader = ["Name", "Role", "Email", "Phone", "Primary"];
    const csvRows = filteredContacts.map((contact) => [
      getDisplayName(contact),
      getRoleText(contact),
      contact?.email || "",
      contact?.phone || "",
      contact?.isPrimary ? "Yes" : "No",
    ]);

    const escapeCell = (value) => `"${String(value).replace(/"/g, '""')}"`;
    const csvText = [csvHeader, ...csvRows].map((row) => row.map(escapeCell).join(",")).join("\n");

    const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "contacts.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const buildPayload = () => ({
    firstName,
    lastName,
    email,
    phone,
    jobTitle,
    role,
    isPrimary,
  });
  const handleSaveContact = async () => {
    const payload = buildPayload();

    if (!firstName || !email) {
      setPopup({
        open: true,
        message: "First name and email are required",
        success: false,
      });
      return;
    }

    try {
      setLoading(true);

      const response = await addAccountContact(id, payload);

      if (response) {
        setMessage("Contact added successfully");
        setShowSuccessModal(true);
        setShowNewContact(false);
        resetForm();
        await loadContactsdata();
      } else {
        setMessage("Contact not Added Successfully");
        setShowSuccessModal(true);
      }
    } catch (error) {
      console.error("Update contact error:", error);

      setPopup({
        open: true,
        message: "Something went wrong. Please try again.",
        success: false,
      });
    } finally {
      setLoading(false);
    }
  };
  const handleUpdateContact = async () => {
    const payload = buildPayload();

    if (!firstName || !email) {
      setPopup({
        open: true,
        message: "First name and email are required",
        success: false,
      });
      return;
    }

    try {
      setLoading(true);

      const response = await updateAccountContact(
        id,
        editingContact.id, // 🔥 REQUIRED FOR UPDATE
        payload,
      );

      if (response) {
        await loadContactsdata();
        setMessage("Contact Updated Successfully");
        setShowSuccessModal(true);
        setShowNewContact(false);
        setEditingContact(null);
        resetForm();
      } else {
        setMessage("Contact not Updated Successfully");
        setShowSuccessModal(true);
      }
    } catch (error) {
      console.error("Update error:", error);

      setPopup({
        open: true,
        message: "Something went wrong while updating",
        success: false,
      });
    } finally {
      setLoading(false);
    }
  };
  const handleDeleteContact = async () => {
    try {
      setLoading(true);

      const response = await deleteAccountContact(id, editingContact.id);
      if (response?.message) {
        setMessage("Deleted Successfully");
        setShowDeleteModal(false);
        setShowSuccessModal(true);
        await loadContactsdata();
      } else {
        setMessage("Deleted unSuccessfully");
      }
    } catch (error) {
      console.error("Delete error:", error);

      setPopup({
        open: true,
        message: "Something went wrong while deleting",
        success: false,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
      :root {
  --opd-crm:        #1dcb8a;
  --opd-email:      #1e3a5f;
  --opd-meeting:    #5bb8f5;
  --opd-multiple:   #1e5fa8;
  --opd-h-high:     #22c55e;
  --opd-h-mid:      #f59e0b;
  --opd-h-low:      #ef4444;
  --opd-border:     #CBD5E1;
  --opd-muted:      #9ca3af;
  --opd-accent:     #1a56db;
  --opd-bg:         #f4f6f9;
  --opd-card:       #ffffff;
  --opd-text:       #111827;
  --opd-text2:      #374151;
  --opd-text3:      #6b7280;
}
        .ct-wrap { background: #fff; border: 1px solid var(--opd-border); border-radius: 12px; padding: 24px; }
        .ct-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
        .ct-header h2 { font-size: 20px; font-weight: 700; color: #0f172a; margin: 0; }
        .ct-btn { display: flex; align-items: center; gap: 8px; background: #1D4ED8; color: #fff; border: none; border-radius: 50px; padding: 8px 20px; font-size: 13px; font-weight: 600; cursor: pointer; }
        .ct-btn:hover { background: #1F3A8A; }
        .ct-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
        .ct-card { border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; cursor: default; transition: border-color 0.15s, box-shadow 0.15s; }
        .ct-card.ct-card-primary { background: #EFF6FF; }
        .ct-card:hover { box-shadow: 0 4px 6px rgba(0,0,0,.08); border-color: #cbd5e1; }
        .ct-card.ct-card-active { border-color: #1D4ED8; box-shadow: 0 0 0 2px rgba(29,78,216,.15); }
        .ct-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
        .ct-card-top-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
        .ct-avatar-wrap { width: 32px; height: 32px; border-radius: 50%; overflow: hidden; flex-shrink: 0; background: #E2E8F0; display: flex; align-items: center; justify-content: center; }
        .ct-avatar-img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .ct-avatar-fallback { color: #0F172A; font-size: 13px; font-weight: 700; letter-spacing: 0.5px; }
        .ct-name-row { display: flex; align-items: center; gap: 8px; min-width: 0; }
        .ct-name { font-size: 16px; font-weight: 700; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px; }
        .ct-div { width: 1px; height: 14px; background: #cbd5e1; }
        .ct-role { font-size: 13px; color: #475569; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 170px; }
        .ct-card-top-right { display: flex; align-items: center; gap: 8px; }
        .ct-badge { display: inline-flex; align-items: center; gap: 6px; background: #DBEAFE; color: #1D4ED8; border: none; border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 600; }
        .ct-badge i { color: #1D4ED8; }
        .ct-edit-btn { border: none; background: transparent; color: #475569; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; width: auto; height: auto; }
        .ct-edit-btn:hover { color: #1d4ed8; }
        .ct-card-divider { height: 1px; width: 100%; background: #e2e8f0; margin: 2px 0; }
        .ct-detail-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
        .ct-detail-left { display: flex; align-items: flex-start; gap: 10px; min-width: 0; }
        .ct-detail-icon { width: 18px; color: #0F172A; font-size: 16px; margin-top: 2px; }
   .ct-detail-icon-bg { width: 32px; height: 32px; border-radius: 999px; background: #E2E8F0; display: inline-flex; align-items: center; justify-content: center; font-size: 14px; margin-top: 0; }
        .ct-card-primary .ct-detail-icon-bg { background: #DBEAFE; }

        .ct-detail-text { min-width: 0; }
       .ct-detail-label { font-size: 10px; color: #64748b; line-height: 1.2; }
        .ct-detail-value { font-size: 14px; color: #0f172a; line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 220px; }
        .ct-action-btn { display: inline-flex; align-items: center; gap: 6px; background: transparent; color: #0F172A; border: none; font-size: 14px; font-weight: 600; padding: 0; border-radius: 0; cursor: pointer; white-space: nowrap; }
       
        .ct-action-btn:hover { color: #0F172A; }
        .ct-action-btn i { font-size: 14px; }

        /* Avatar */
        .ct-avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #0F172A; background: #E2E8F0; flex-shrink: 0; }

        /* New/Edit Contact Form */
        .ct-new-form { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 20px; }
        .ct-form-header { display: flex; align-items: center; justify-content: space-between; }
        .ct-form-title { font-size: 16px; font-weight: 700; color: #0f172a; }
        .ct-form-close { background: none; border: none; cursor: pointer; color: #64748b; padding: 4px; }
        .ct-form-close:hover { color: #0f172a; }
        .ct-fields { display: flex; flex-wrap: wrap; gap: 16px; }
        .ct-field { display: flex; flex-direction: column; gap: 4px; flex: 0 0 calc(25% - 12px); min-width: 160px; }
        .ct-field.half { flex: 0 0 calc(50% - 8px); }
        .ct-field.full { flex: 0 0 100%; }
        .ct-field label { font-weight: 600; font-size: 13px; color: #0f172a; line-height: 20px; }
        .ct-field input, .ct-field select {
          background: #f8fafc; border: 2px solid #E2E8F0; border-radius: 4px;
          height: 38px; padding: 0 10px; font-size: 13px; color: #475569;
          width: 100%; outline: none; transition: border-color 0.15s;
        }
        .ct-field input:focus, .ct-field select:focus { border-color: #94a3b8; }
        .ct-field input::placeholder { color: #94a3b8; }
        .ct-field select { cursor: pointer; appearance: none; padding-right: 30px; }
        .ct-select-wrap { position: relative; width: 100%; }
        .ct-select-wrap select { width: 100%; }
        .ct-select-wrap::after { content: ''; position: absolute; right: 12px; top: 50%; transform: translateY(-50%); border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #64748b; pointer-events: none; }
        .ct-form-actions { display: flex; gap: 8px; justify-content: flex-start; }
        .ct-btn-cancel { display: inline-flex; align-items: center; background: #E2E8F0; color: #374151; border: none; font-size: 13px; font-weight: 600; padding: 8px 20px; border-radius: 50px; cursor: pointer; }
        .ct-btn-cancel:hover { background: #CBD5E1; }
        .ct-btn-save { display: inline-flex; align-items: center; background: #1D4ED8; color: #fff; border: none; font-size: 13px; font-weight: 600; padding: 8px 20px; border-radius: 50px; cursor: pointer; }
        .ct-btn-save:hover { background: #1F3A8A; }




        .ct-btn-delete { display: inline-flex; align-items: center; background: #EF4444; color: #fff; border: none; font-size: 13px; font-weight: 600; padding: 8px 20px; border-radius: 50px; cursor: pointer; }
        .ct-btn-delete:hover { background: #DC2626; }
        .ct-btn-update { display: inline-flex; align-items: center; background: #16A34A; color: #fff; border: none; font-size: 13px; font-weight: 600; padding: 8px 20px; border-radius: 50px; cursor: pointer; }
        .ct-btn-update:hover { background: #15803D; }

        /* Toggle */
        .ct-toggle-row { display: flex; align-items: center; height: 38px; }
        .ct-toggle { position: relative; width: 40px; height: 24px; display: inline-block; }
        .ct-toggle input { opacity: 0; width: 0; height: 0; }
        .ct-toggle-slider { position: absolute; inset: 0; background: #cbd5e1; border-radius: 100px; cursor: pointer; transition: background .2s; }
        .ct-toggle-slider::before { content: ''; position: absolute; width: 16px; height: 16px; background: #fff; border-radius: 50%; top: 4px; left: 4px; transition: transform .2s; }
        .ct-toggle input:checked + .ct-toggle-slider { background: #1D4ED8; }
        .ct-toggle input:checked + .ct-toggle-slider::before { transform: translateX(16px); }
        .ct-toggle input:disabled + .ct-toggle-slider { cursor: not-allowed; opacity: 0.7; }
        .ct-primary-label { font-size: 12px; color: #64748b; margin-top: 4px; font-weight: 500; }
        .ctl-contact-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .ctl-contact-sub { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 11px; color: #888; margin-top: 2px; }
        .ctl-filter-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
        .ctl-filter-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .ctl-search-wrap { position: relative; min-width: 240px; }
        .ctl-search-wrap i { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: #64748b; font-size: 14px; }
        .ctl-search-input { height: 34px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0 12px 0 30px; font-size: 13px; color: #0f172a; width: 100%; background: #fff; }
        .ctl-search-input:focus { outline: none; border-color: #1d4ed8; }
        .ctl-role-select { height: 34px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0 10px; font-size: 13px; color: #0f172a; min-width: 160px; background: #fff; }
        .ctl-role-select:focus { outline: none; border-color: #1d4ed8; }
        .ctl-download-btn { width: 40px; height: 40px; border-radius: 50px; border: none; background: #E2E8F0; cursor: pointer; display: flex; align-items: center; justify-content: center; color: #475569; font-size: 16px; flex-shrink: 0; }
        .ctl-download-btn:hover { background: #cbd5e1; }
        .ctl-download-btn img { width: 18px; height: 18px; }
        .ctl-check-th, .ctl-check-td { width: 44px; text-align: center; }
        .ctl-check-input { width: 15px; height: 15px; cursor: pointer; appearance: none; -webkit-appearance: none; background: #CBD5E1; border-radius: 3px; border: none; position: relative; flex-shrink: 0; }
        .ctl-check-input:checked { background: #1d4ed8; }
        .ctl-check-input:checked::after { content: '✓'; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #fff; font-size: 11px; font-weight: 700; line-height: 1; }
      `}</style>

      <div className="ct-wrap">
        <div className="ct-header">
          <h2>Contact List</h2>
          {contacts.length > 0 && !editingContact && (
            <div className="ct-view-toggle">
              <button
                type="button"
                className={viewMode === "account" ? "active" : ""}
                onClick={() => setViewMode("account")}
              >
                Card View
              </button>
              <button
                type="button"
                className={viewMode === "list" ? "active" : ""}
                onClick={() => setViewMode("list")}
              >
                List View
              </button>
            </div>
          )}

          {!editingContact && (
          <button className="ct-btn" onClick={handleNewContact}>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <line x1="19" y1="8" x2="19" y2="14" />
              <line x1="22" y1="11" x2="16" y2="11" />
            </svg>
            New Contact
          </button>
          )}
        </div>

        {/* New Contact Form */}
        {showNewContact && (
          <div className="ct-new-form">
            <div className="ct-form-header">
              <div className="ct-form-title">New Contact</div>
              <button
                className="ct-form-close"
                onClick={() => {
                  setShowNewContact(false);
                  resetForm();
                }}
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <div className="ct-fields">
              <div className="ct-field">
                <label>First Name</label>
                <input
                  type="text"
                  value={firstName}
                  placeholder="Input"
                  onChange={(e) => setFirstName(e.target.value)}
                />
              </div>
              <div className="ct-field">
                <label>Last Name</label>
                <input
                  type="text"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                />
              </div>
              <div className="ct-field">
                <label>Role</label>
                <input
                  type="text"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                />
              </div>
              <div className="ct-field">
                <label>Phone</label>
                <input
                  type="tel"
                  value={phone}
                  placeholder="+919876543210"
                  onChange={(e) => setPhone(e.target.value)}
                />
                {phone && !isPhoneValid && (
                  <span className="ct-error">
                    Phone must be +91 followed by 10 digits
                  </span>
                )}
              </div>
              <div className="ct-field">
                <label>Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
                {email && !isEmailValid && (
                  <span className="ct-error">Invalid email format</span>
                )}
              </div>

              <div className="ct-field">
                <label>Primary</label>
                <div className="ct-toggle-row">
                  <label className="ct-toggle">
                    <input
                      type="checkbox"
                      checked={isPrimary}
                      onChange={(e) => setIsPrimary(e.target.checked)}
                    />
                    <span className="ct-toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
            <div className="ct-form-actions">
              <button
                className="ct-btn-cancel"
                onClick={resetForm}
              >
                Clear
              </button>
              <button
                className="ct-btn-save"
                onClick={handleSaveContact}
                disabled={!isFormValid || loading}
              >
                {loading ? "Adding..." : "Add"}
              </button>
            </div>
          </div>
        )}

        {/* Edit Contact Form */}
        {editingContact && (
          <div className="ct-new-form">
            <div className="ct-form-header">
              <div className="ct-form-title">Edit Contact</div>
              <button className="ct-form-close" onClick={handleCloseEdit}>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <div className="ct-fields">
              <div className="ct-field">
                <label>First Name</label>
                <input
                  type="text"
                  value={firstName}
                  placeholder="Input"
                  onChange={(e) => setFirstName(e.target.value)}
                />
              </div>
              <div className="ct-field">
                <label>Last Name</label>
                <input
                  type="text"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                />
              </div>
              <div className="ct-field">
                <label>Role</label>
                <input
                  type="text"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                />
              </div>
              <div className="ct-field">
                <label>Phone</label>
                <input
                  type="tel"
                  value={phone}
                  placeholder="+919876543210"
                  onChange={(e) => setPhone(e.target.value)}
                />
                {phone && !isPhoneValid && (
                  <span className="ct-error">
                    Phone must be +91 followed by 10 digits
                  </span>
                )}
              </div>
              <div className="ct-field">
                <label>Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
                {email && !isEmailValid && (
                  <span className="ct-error">Invalid email format</span>
                )}
              </div>

              <div className="ct-field">
                <label>Primary</label>
                <div className="ct-toggle-row">
                  <label className="ct-toggle">
                    <input
                      type="checkbox"
                      checked={isPrimary}
                      disabled={isPrimary} // Prevent unchecking if already primary
                      onChange={(e) => setIsPrimary(e.target.checked)}
                    />
                    <span className="ct-toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
            <div className="ct-bottom-bar">
              <button className="ct-btn-cancel" onClick={handleCloseEdit}>
                Cancel
              </button>
              <button
                className="ct-btn-delete"
                onClick={() => setShowDeleteModal(true)}
              >
                Delete
              </button>
              <button
                className="ct-btn-save ct-btn-update"
                onClick={handleUpdateContact}
                disabled={loading}
              >
                {loading ? "Updating..." : "Update"}
              </button>
            </div>
          </div>
        )}
        {/* <div className="ct-view-toggle">
          <button
            type="button"
            className={viewMode === "account" ? "active" : ""}
            onClick={() => setViewMode("account")}
          >
            Account View
          </button>

          <button
            type="button"
            className={viewMode === "list" ? "active" : ""}
            onClick={() => setViewMode("list")}
          >
            List View
          </button>
        </div> */}
        {/* Only show grid when not editing */}
        {!editingContact && (
          <>
            {viewMode === "account" ? (
              <div className="ct-grid">
                {contacts.length === 0 && (
                  <p style={{ color: "#64748b", fontSize: 14, padding: "20px 0" }}>No contacts available</p>
                )}
                {contacts.map((c) => (
                  <div
                    className={`ct-card${c.isPrimary ? " ct-card-primary" : ""}${editingContact?.id === c.id ? " ct-card-active" : ""
                      }`}
                    key={c.id}
                  >
                    <div className="ct-card-top">
                      <div className="ct-card-top-left">
                        <div className="ct-avatar-wrap">
                          {getAvatarUrl(c) ? (
                            <img
                              src={getAvatarUrl(c)}
                              alt={getDisplayName(c)}
                              className="ct-avatar-img"
                            />
                          ) : (
                            <span className="ct-avatar-fallback">{getInitials(c)}</span>
                          )}
                        </div>
                        <div className="ct-name-row">
                          <span className="ct-name">{getDisplayName(c)}</span>
                          <div className="ct-div"></div>
                          <span className="ct-role">{getRoleText(c)}</span>
                        </div>
                      </div>

                      <div
                        className="ct-card-top-right"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {c.isPrimary && (
                          <span className="ct-badge">
                            <i className="ri-star-fill"></i>
                            Primary
                          </span>
                        )}
                        <button
                          type="button"
                          className="ct-edit-btn"
                          onClick={() => handleCardClick(c)}
                        >
                          <i className="ri-pencil-line" style={{ fontSize: 18 }}></i>
                        </button>
                      </div>
                    </div>

                 

                    <div className="ct-detail-row">
                      <div className="ct-detail-left">
 <i className="ri-mail-line ct-detail-icon ct-detail-icon-bg"></i>                        <div className="ct-detail-text">
                          <div className="ct-detail-label">Email</div>
                          <div className="ct-detail-value">{c.email || "-"}</div>
                        </div>
                      </div>

                      <div
                        className="ct-actions"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button type="button" className="ct-action-btn">
                          Compose
                          <i className="ri-arrow-right-s-line"></i>
                        </button>
                      </div>
                    </div>

                    <div className="ct-detail-row">
                      <div className="ct-detail-left">
         <i className="ri-phone-line ct-detail-icon ct-detail-icon-bg"></i>
                                 <div className="ct-detail-text">
                          <div className="ct-detail-label">Phone</div>
                          <div className="ct-detail-value">{c.phone || "-"}</div>
                        </div>
                      </div>

                      <div
                        className="ct-actions"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button type="button" className="ct-action-btn">
                          Call
                          <i className="ri-arrow-right-s-line"></i>
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <>
                <div className="ctl-filter-card">
                  <div className="ctl-filter-left">
                    <div className="ctl-search-wrap">
                      <i className="ri-search-line"></i>
                      <input
                        type="text"
                        className="ctl-search-input"
                        placeholder="Search Contact"
                        value={searchContact}
                        onChange={(e) => setSearchContact(e.target.value)}
                      />
                    </div>

                    <select
                      className="ctl-role-select"
                      value={roleFilter}
                      onChange={(e) => setRoleFilter(e.target.value)}
                    >
                      <option value="">All Roles</option>
                      {roleOptions.map((roleValue) => (
                        <option key={roleValue} value={roleValue}>
                          {roleValue}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    type="button"
                    className="ctl-download-btn"
                    onClick={handleDownloadContacts}
                    aria-label="Download contacts"
                    title="Download contacts"
                  >
                    <i className="bi bi-download"></i>
                  </button>
                </div>

                <div className="ctl-table-card">
                  <div className="ctl-table-outer">
                    {/* Frozen Left */}
                    <div className="ctl-frozen-panel">
                      <table>
                        <colgroup>
                          <col style={{ width: 44 }} />
                          <col style={{ width: 250 }} />
                        </colgroup>

                        <thead>
                          <tr>
                            <th className="ctl-th ctl-check-th">
                              <input
                                type="checkbox"
                                className="ctl-check-input"
                                checked={areAllVisibleSelected}
                                onChange={(e) => handleToggleSelectAll(e.target.checked)}
                              />
                            </th>
                            <th className="ctl-th">Name</th>
                          </tr>
                        </thead>

                        <tbody>
                          {filteredContacts.length === 0 ? (
                            <tr>
                              <td className="ctl-no-results" colSpan={2}>
                                No contacts found
                              </td>
                            </tr>
                          ) : (
                            filteredContacts.map((c, i) => {
                              const rowId = getRowId(c);
                              return (
                                <tr
                                  key={rowId}
                                  className={
                                    i % 2 === 0 ? "ctl-row-even" : "ctl-row-odd"
                                  }
                                  style={{ cursor: "pointer" }}
                                  onClick={() => handleCardClick(c)}
                                >
                                  <td
                                    className="ctl-td ctl-check-td"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    <input
                                      type="checkbox"
                                      className="ctl-check-input"
                                      checked={selectedRows.includes(rowId)}
                                      onChange={(e) => handleToggleRow(c, e.target.checked)}
                                    />
                                  </td>
                                  <td className="ctl-td">
                                    <div className="ctl-contact-name">{getDisplayName(c)}</div>
                                    <div className="ctl-contact-sub">
                                      {c.contactId}
                                    </div>
                                  </td>
                                </tr>
                              );
                            })
                          )}
                        </tbody>
                      </table>
                    </div>

                    {/* Scrollable Right */}
                    <div className="ctl-scroll-panel">
                      <table>
                        <thead>
                          <tr>
                            <th className="ctl-th">Role</th>
                            <th className="ctl-th">Email Address</th>
                            <th className="ctl-th">Phone Number</th>
                            <th className="ctl-th">Primary Contact</th>
                          </tr>
                        </thead>

                        <tbody>
                          {filteredContacts.length === 0 ? (
                            <tr>
                              <td colSpan={4} className="ctl-no-results">
                                No contacts found
                              </td>
                            </tr>
                          ) : (
                            filteredContacts.map((c, i) => (
                              <tr
                                key={getRowId(c)}
                                className={
                                  i % 2 === 0 ? "ctl-row-even" : "ctl-row-odd"
                                }
                                style={{ cursor: "pointer" }}
                                onClick={() => handleCardClick(c)}
                              >
                                <td className="ctl-td">
                                  <span
                                    className={
                                      c.isPrimary
                                        ? "ctl-role-badge ctl-role-primary"
                                        : "ctl-role-badge ctl-role-secondary"
                                    }
                                  >
                                    {getRoleText(c)}
                                  </span>
                                </td>
                                <td className="ctl-td">{c.email || "-"}</td>
                                <td className="ctl-td">{c.phone || "-"}</td>

                                <td className="ctl-td">
                                  {c.isPrimary ? (
                                    <span className="">Yes</span>
                                  ) : (
                                    <span className="">No</span>
                                  )}
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </>
            )}
          </>
        )}
        <Modal
          show={showDeleteModal}
          onHide={() => {
            setShowDeleteModal(false);
            setConfirmDelete(false);
          }}
          centered
        >
          <Modal.Header>
            <Modal.Title style={{ fontSize: "18px" }}>
              <i
                className=" ri-information-fill"
                style={{
                  color: "#B91C1C",
                  fontSize: "20px",
                  marginRight: "10px",
                }}
              ></i>
              Delete Contact
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <p className="delete-text">
              Are you sure you want to delete this contact? This may affect
              associated deals linked to this contact.
            </p>

            <div className="form-check mt-3">
              <input
                className="form-check-input"
                type="checkbox"
                id="confirmDelete"
                checked={confirmDelete}
                onChange={(e) => setConfirmDelete(e.target.checked)}
              />

              <label
                className="form-check-label bold-text"
                htmlFor="confirmDelete"
              >
                I want to delete the contact.
              </label>
            </div>
          </Modal.Body>

          <Modal.Footer style={{ border: "none" }}>
            <button
              className="ct-btn-save"
              onClick={() => setShowDeleteModal(false)}
            >
              Cancel
            </button>

            <button
              className="ct-btn-save"
              onClick={handleDeleteContact}
              disabled={!confirmDelete || loading}
              style={{
                opacity: !confirmDelete || loading ? 0.5 : 1,
                cursor: !confirmDelete || loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "Deleting..." : "Delete"}
            </button>
          </Modal.Footer>
        </Modal>
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
              onClick={handleCloseEdit} // your contacts list route
            >
              OK
            </button>
          </Modal.Footer>
        </Modal>
      </div>
    </>
  );
}
