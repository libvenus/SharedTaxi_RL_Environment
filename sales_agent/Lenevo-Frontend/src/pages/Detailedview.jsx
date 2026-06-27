import { useState, useEffect, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import leftArrowIcon from "../assets/icons/left.png";
import rightArrowIcon from "../assets/icons/right1.png";
import accounticon from "../assets/icons/account_icon.png";
import buildingicon from "../assets/icons/building_icon.png";
import {
  fetchOpportunityDetails,
  updateDeal,
  fetchOpportunityContactDetails,
  fetchAccountContactDetails,
  editAccountContact,
  deleteOpportunity,
  deleteopportunityContact,
  fetchActivities,
  createOpportunityNote,
  fetchOpportunityNotes,
  updateOpportunity,
} from "../api/client";
import { formatCurrencyShort, formatDateMMDDYY } from "../utils/format";
import "../styles/account.css";
import { Modal } from "react-bootstrap";
export default function Detailedview() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTopTab, setActiveTopTab] = useState("overview");
  const [opportunityDetails, setOpportunityDetails] = useState({});
  const [formData, setFormData] = useState(opportunityDetails || {});
  const [editingContact, setEditingContact] = useState(null);
  const [isPrimarys, setIsPrimarys] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [role, setRole] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [phone, setPhone] = useState("");
  const [email, setEmail] = useState("");
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showOpportunityDeleteModal, setShowOpportunityDeleteModal] =
    useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [message, setMessage] = useState("");
  const [contacts, setContacts] = useState([]);
  const [selectedContact, setSelectedContact] = useState(null);
  const [isDecisionMaker, setIsDecisionMaker] = useState(false);
  const [availableContacts, setAvailableContacts] = useState([]);
  const [showContactPicker, setShowContactPicker] = useState(false);
  const [contactSearch, setContactSearch] = useState("");
  const pickerRef = useRef(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmOpportunityDelete, setConfirmOpportunityDelete] =
    useState(false);
  const [selectedTempContacts, setSelectedTempContacts] = useState([]);
  const [noteText, setNoteText] = useState("");
  const [noteError, setNoteError] = useState("");

  const toBool = (value) => {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value === 1;
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      return (
        normalized === "true" || normalized === "1" || normalized === "yes"
      );
    }
    return false;
  };
  // Close contact picker on outside click
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target)) {
        setShowContactPicker(false);
      }
    };
    if (showContactPicker) {
      document.addEventListener("mousedown", handleOutsideClick);
    }
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [showContactPicker]);

  const [nextActionsList, setNextActionsList] = useState([
    { nextAction: "", nextActionDueDate: "" },
  ]);
  const [competitorsList, setCompetitorsList] = useState([
    { competitor: "", competitorType: "", resellingPartner: "" },
  ]);
  const execCards = [
    {
      title: "Lid Hinge Replacement",
      po: "1324356789",
      sellThroughDate: "13/06/26",
      week: "24",
      orderNumber: "1324356789",
      shippingDate: "13/06/26",
    },
    {
      title: "Laptops Parts",
      po: "1324356789",
      sellThroughDate: "13/06/26",
      week: "24",
      orderNumber: "1324356789",
      shippingDate: "13/06/26",
    },
    {
      title: "Lid Hinge Replacement",
      po: "1324356789",
      sellThroughDate: "13/06/26",
      week: "24",
      orderNumber: "1324356789",
      shippingDate: "13/06/26",
    },
    {
      title: "Laptops Parts",
      po: "1324356789",
      sellThroughDate: "13/06/26",
      week: "24",
      orderNumber: "1324356789",
      shippingDate: "13/06/26",
    },
  ];
  useEffect(() => {
    fetchOpportunityDetails(id).then((response) => {
      setOpportunityDetails(response);
      if (response) {
        setFormData({
          ...response,
          accountName: response.account?.name || "",
          region: response.account?.territory || "",
          totalAccountValue: response.account?.totalAccountValue || "",
          openDealsCount: response.account?.openDealsCount || "",
          accountSegment: response.account?.segment || "",
          accountIndustry: response.account?.industry || "",
          accountEmployeeCount: response.account?.employeeCount || "",
          estimatedDealValue: response.value || "",
          estimatedCloseDate: response.closeDate || "",
        });
        // Map competitors to competitorsList
        if (response.competitors?.length > 0) {
          setCompetitorsList(
            response.competitors.map((c) => ({
              id: c.id || null,
              competitor: c.competitorName || "",
              competitorType: c.competitorType || "",
              resellingPartner: c.resellingPartnerId || "",
            })),
          );
        }
      }
    });
  }, [id]);
  console.log(opportunityDetails);

  const loadActivities = async () => {
    if (!id) return;
    try {
      setLoading(true);
      const res = await fetchActivities(id);
      setActivities(res || []);
    } catch (err) {
      console.error("Failed to fetch activities:", err);
      setActivities([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const loadContacts = async () => {
      try {
        const response = await fetchOpportunityContactDetails(id);

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
  const loadContactsData = async () => {
    try {
      const response = await fetchOpportunityContactDetails(id);

      const allContacts = [
        ...(response?.decisionMaker ? [response.decisionMaker] : []),
        ...(response?.additionalContacts || []),
      ];

      setContacts(allContacts);

      if (response?.decisionMaker) {
        setSelectedContact(response.decisionMaker);
      }
    } catch (error) {
      console.error("Failed to fetch contacts:", error);
      setContacts([]);
    }
  };
  const loadAvailableContacts = async () => {
    try {
      const response = await fetchAccountContactDetails(
        opportunityDetails?.accountId,
      );

      const list = [
        ...(response?.primary ? [response.primary] : []),
        ...(response?.others || []),
      ];

      setAvailableContacts(list);
      setShowContactPicker(true);
    } catch (err) {
      console.error(err);
    }
  };
  const handleCardClick = (contact) => {
    console.log(contact);
    setEditingContact(contact);
    setFirstName(contact.firstName || "");
    setLastName(contact.lastName || "");
    setEmail(contact.email || "");
    setPhone(contact.phone || "");
    setJobTitle(contact.jobTitle || "");
    setRole(contact.role || "");
    setIsDecisionMaker(
      toBool(
        contact?.isDecisionMaker ??
          contact?.isdecisionmaker ??
          contact?.is_primary,
      ),
    );
    // setShowNewContact(false);
  };
  const buildPayload = {
    opportunityid: id,
    name: formData.name || "",
    accountid: formData.accountId || "",
    contactid: formData.contactId || "",
    stagename: formData.stage?.raw || formData.stage || "",
    estimatedvalue: Number(formData.estimatedDealValue) || 0,
    estimatedclosedate:
      formData.estimatedCloseDate?.slice(0, 10) ||
      new Date().toISOString().slice(0, 10),
    lvo_forecastcategory: formData.forecastCategory || "",
    lvo_salesmotion: formData.saleMotion?.raw || formData.saleMotion || "",
    lvo_dealtempo: formData.tempoClass || "",
    lvo_solutionarea: formData.solutionArea || "",
    lvo_solutioncertificationid: formData.solutionCertifications || "",
    lvo_dealqualificationreviewid: formData.dealQualificationReview
      ? "true"
      : "",
    lvo_rfpid: formData.rfpReference || "",
    lvo_geoid: formData.region || "",
    lvo_country: formData.country || "",
    lvo_businessgroup: formData.businessGroup || "",
    lvo_dealdeskanalystid: formData.dealDeskAnalyst || "",
    lvo_dealengagementmanagerid: formData.dealEngagementManager || "",
    lvo_solutionarchitectid: formData.solutionServiceDomainSpecialist || "",
    lvo_leasingvendor: formData.leasingVendor || "",
    lvo_primary_partner: formData.partnerCommercialModel || "",
    owninguser: formData.ownerName || "",
    msdyn_accountmanagerid: formData.accountManagerId || "",
    closeprobability: Number(formData.closeProbability) || 0,
    statecode: 0,
    lvo_dealhealthscore: formData.health?.score || 0,
    lvo_riskscore: formData.riskScore || 0,
    lvo_riskreason: formData.riskReason || "",
    lvo_tempoclass: formData.tempoClass || "",
    lvo_stageentrydate: formData.stageEntryDate || new Date().toISOString(),
    lvo_createdat: formData.createdAt || new Date().toISOString(),
    lvo_dealhealthupdatedat:
      formData.health?.updatedAt || new Date().toISOString(),
    lvo_summary: formData.summary || "",
    lvo_priority: formData.priority || "",
    lvo_leadorigin: formData.leadOrigin || "",
    lvo_partnerinvolved: formData.partnerInvolved || false,
    lvo_parentopportunityid: formData.parentOpportunityName || "",
    createdby: formData.createdBy || "",
    modifiedon: formData.modifiedAt || new Date().toISOString(),
    modifiedby: formData.modifiedBy || "",
    actual_revenue: Number(formData.actualRevenue) || 0,
    actual_close_date:
      formData.actualCloseDate?.slice(0, 10) ||
      new Date().toISOString().slice(0, 10),
    close_reason: formData.closeReason || "",
    sales_order_reference: formData.salesOrderReference || "",
    won_solution_category: formData.wonSolutionCategory || "",
    win_notes_commentary: formData.winNotesCommentary || "",
    invoice_number: formData.invoiceNumber || "",
    loss_reason: formData.lossReason || "",
    competitor_won: formData.competitorWon || "",
    lost_solution_category: formData.lostSolutionCategory || "",
    lost_revenue_value: Number(formData.lostRevenueValue) || 0,
    loss_notes_commentary: formData.lossNotesCommentary || "",
    deal_appeal: formData.dealAppeal || "",
    re_engagement_date:
      formData.reEngagementDate?.slice(0, 10) ||
      new Date().toISOString().slice(0, 10),
    solution_area: formData.solutionArea || "",
    sub_solution_area: formData.subSolutionArea || "",
    solution_certifications: formData.solutionCertifications || "",
    solution_offerings: formData.solutionOfferings || "",
    leasing_vendor: formData.leasingVendor || "",
    sales_model: formData.salesModel || "",
    service_model: formData.serviceModel || "",
    budget_confirmed: formData.budgetConfirmed || false,
    quote_reference: formData.quoteReference || "",
    partner_commercial_model: formData.partnerCommercialModel || "",
    actual_confirmed_revenue: Number(formData.actualConfirmedRevenue) || 0,
    reseller_channel_account: formData.resellerChannelAccount || "",
    deal_protection_status: formData.dealProtectionStatus || "",
    deal_registration_ref: formData.dealRegistrationRef || "",
    number_of_countries: Number(formData.numberOfCountries) || 0,
    sow_required: formData.sowRequired || false,
    multi_country_solution_required: formData.multiCountrySolution || false,
    deal_qualification_review: formData.dealQualificationReview ? "true" : "",
    solution_handover_artefacts: formData.solutionHandoverArtefacts || "",
    solution_service_executive: formData.solutionServiceExecutive || "",
    solution_service_domain_specialist:
      formData.solutionServiceDomainSpecialist || "",
    lgfs_sales_representatives: formData.lgfsSalesReps || "",
    lgfs_sales_support: formData.lgfsSalesSupport || "",
    deal_desk_analyst: formData.dealDeskAnalyst || "",
    deal_engagement_manager: formData.dealEngagementManager || "",
    ssds_channel: formData.ssdsChannel || "",
    sell_through_week_auto: Number(formData.sellThroughWeek) || 0,
    competitor_type: formData.orderCompetitorType || "",
    order_date:
      formData.orderDate?.slice(0, 10) || new Date().toISOString().slice(0, 10),
    shipping_date:
      formData.shippingDate?.slice(0, 10) ||
      new Date().toISOString().slice(0, 10),
    sales_order_reference_po: formData.salesOrderRef || "",
    created_date: formData.createdAt || new Date().toISOString(),
    order_number: formData.orderNumber || "",
    days_in_stage: Number(formData.daysInStage) || 0,
  };
  const editPayload = () => ({
    firstName,
    lastName,
    email,
    phone,
    jobTitle,
    role,
    isDecisionMaker,
  });
  const handleUpdateContact = async () => {
    const payload = buildPayload;

    try {
      setLoading(true);

      const response = await updateOpportunity(id, payload);

      if (response) {
        setMessage("Updated Successfully");
        setShowSuccessModal(true);
      } else {
        setMessage("Update Failed");
        setShowSuccessModal(true);
      }
    } catch (error) {
      console.error("Update error:", error);
      setMessage("Update Failed");
      setShowSuccessModal(true);
    } finally {
      setLoading(false);
    }
  };
  const handleEditContact = async () => {
    const payload = editPayload();

    try {
      setLoading(true);
      const response = await editAccountContact(
        id,
        editingContact.id, // 🔥 REQUIRED FOR UPDATE
        payload,
      );

      if (response) {
        setMessage("Contact Updated Successfully");
        setShowSuccessModal(true);

        // resetForm();
      } else {
        setMessage("Contact not Updated Successfully");
        setShowSuccessModal(true);
      }
    } catch (error) {
      console.error("Update error:", error);
    } finally {
      setLoading(false);
    }
  };
  const handleDeleteContact = async () => {
    try {
      setLoading(true);

      const response = await deleteopportunityContact(id, editingContact.id);
      if (response?.message) {
        setMessage("Deleted Successfully");
        setShowDeleteModal(false);
        setShowSuccessModal(true);
      } else {
        setMessage("Deleted unSuccessfully");
      }
    } catch (error) {
      console.error("Delete error:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteOpportunity = async () => {
    try {
      setLoading(true);
      await deleteOpportunity(id);
      setShowOpportunityDeleteModal(false);
      setConfirmOpportunityDelete(false);
      navigate("/opportunities");
    } catch (error) {
      console.error("Delete opportunity error:", error);
      setMessage(error?.message || "Failed to delete opportunity");
      setShowSuccessModal(true);
    } finally {
      setLoading(false);
    }
  };
  // const handleDeleteContact = async () => {
  //   try {
  //     setLoading(true);

  //     const response = await deleteAccountContact(id, editingContact.id);
  //     if (response?.message) {
  //       setMessage("Deleted Successfully");
  //       setShowDeleteModal(false);
  //       setShowSuccessModal(true);
  //       await loadContactsdata();
  //     } else {
  //       setMessage("Deleted unSuccessfully");
  //     }
  //   } catch (error) {
  //     console.error("Delete error:", error);

  //     setPopup({
  //       open: true,
  //       message: "Something went wrong while deleting",
  //       success: false,
  //     });
  //   } finally {
  //     setLoading(false);
  //   }
  // };
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };
  const stages = [
    { key: "qualify", label: "Qualify" },
    { key: "develop", label: "Develop" },
    { key: "propose", label: "Propose" },
    { key: "execute", label: "Execute" },
    { key: "closed", label: "Closed" },
  ];

  const currentStage = opportunityDetails?.stage?.raw?.toLowerCase() || "";
  const currentIndex = stages.findIndex((s) => s.key === currentStage);
  const handleCloseEdit = () => {
    setShowSuccessModal(false);
    setEditingContact(null);
  };
  const handleSelectContact = (contact) => {
    setContacts((prev) => {
      const exists = prev.some((c) => c.id === contact.id);
      if (exists) return prev; // prevent duplicates
      return [...prev, contact];
    });

    setShowContactPicker(false);
  };
  const toggleTempContact = (contact) => {
    setSelectedTempContacts((prev) => {
      const exists = prev.some((c) => c.id === contact.id);

      if (exists) {
        return prev.filter((c) => c.id !== contact.id);
      }

      return [...prev, contact];
    });
  };
  const confirmAddContacts = () => {
    setContacts((prev) => {
      const map = new Map(prev.map((c) => [c.id, c]));

      selectedTempContacts.forEach((c) => {
        map.set(c.id, c);
      });

      return Array.from(map.values());
    });

    setShowContactPicker(false);
  };
  function getBadgeClass(type) {
    switch (type) {
      case "email":
        return "dv-badge-success";
      case "meeting":
        return "dv-badge-info";
      case "crm":
        return "dv-badge-warning";
      default:
        return "dv-badge-info";
    }
  }

  function getTypeLabel(type) {
    switch (type) {
      case "email":
        return "Email";
      case "meeting":
        return "Meeting";
      case "crm":
        return "CRM Update";
      case "call":
        return "Call";
      default:
        return type;
    }
  }

  function getBadge(type) {
    switch (type) {
      case "email":
        return "Expansion signal";
      case "meeting":
        return "High-confidence close";
      case "crm":
        return "Record Updated";
      default:
        return "Activity";
    }
  }

  function formatDate(date) {
    return new Date(date).toLocaleString();
  }

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .dv-page {
          --sd-action-bg: #1a73e8;
          --sd-action-hover: #155fc4;
          --sd-action-active: #114fa6;
          font-family: 'Segoe UI', sans-serif;
          font-size: 13px;
          background: #f5f6f8;
          color: #1a1a2e;
          height: 100vh;
          overflow: hidden;
        }

         body {
         padding: 0px;
        }

        .main {
          height: 100%;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .quarter-bar {
          background: #f5f6f8;
          border-bottom: 1px solid #e8eaed;
          padding: 4px 24px;
          font-size: 12px;
          color: #555;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .closure-badge { color: #e2001a; font-weight: 600; }

        .page-header {
          background: #f5f6f8;
          border-bottom: 1px solid #e8eaed;
          padding: 10px 0px 0;
          margin-top: 38px;
        }
        .page-header-row { display: flex; align-items: center; gap: 14px; margin-bottom: 10px;padding: 0px 24px; }
        .back-btn { font-size: 18px; cursor: pointer; color: #444; }
        .topbar-title {     font-size: 28px;
    font-weight: 700; color: #111; }
        .topbar-account { font-size: 12px; text-decoration: underline; cursor: pointer;padding: 4px 10px;
    background: #ECFDF5;
    border-radius: 4px; text-decoration: none; }
        .topbar-actions { margin-left: auto; display: flex; align-items: center; gap: 10px; }
        .btn-edit {
          padding: 6px 18px; border: 1px solid #ccc; border-radius: 6px; background: #fff;
          cursor: pointer; font-size: 13px; font-weight: 500;
        }
        .btn-delete {
          width: 32px;
          height: 32px;
          border-radius: 50px;
          border: none;
          background: #e2001a;
          color: #fff;
          cursor: pointer;
          font-size: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .tabs-outer {
          height: 64px;
          background: #fff;
          display: flex;
          align-items: center;
          border-top: 1px solid #e8eaed;
        }

        .tabs {
          height: 40px;
          display: flex;
          gap: 0;
          background: #fff;
          justify-content: center;
          width: 100%;
          border-bottom: none;
        }

        .tab {
          padding: 8px 16px;
          font-size: 14px;
          cursor: pointer;
          border-bottom: 2px solid transparent;
          white-space: nowrap;
          background: #fff;
          border-top: 0;
          border-left: 0;
          border-right: 0;
          height: 100%;
          display: flex;
          align-items: center;
        }
        .tab.active {
          color: #1a73e8;
          border-bottom-color: #1a73e8;
          font-weight: 500;
          background: #EFF6FF;
        
        }

        .content { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; gap: 20px; }
        .col-main { flex: 1; display: flex; flex-direction: column; gap: 16px;    }
        .col-side { width: 280px; min-width: 280px; display: flex; flex-direction: column; gap: 16px; }

        .side-tabs {
          display: flex;
          background: #fff;
          border: 1px solid #e8eaed;
          border-radius: 8px;
          padding: 4px;
          gap: 4px;
        }
        .side-tab {
          flex: 1;
          background: transparent;
          border-radius: 6px;
          padding: 8px 6px;
          font-size: 11px;
          color: #555;
          cursor: pointer;
          font-weight: 600;
          text-decoration: none;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .side-tab.active {
          background: #e8f0fe;
          color: #1a73e8;
        }

        .dv-side-list {
          display: grid;
          gap: 8px;
          font-size: 12px;
          color: #444;
          line-height: 1.5;
        }
        .dv-side-list b { color: #111; }

        .pipeline {
          background: #fff; border: 1px solid #e8eaed; border-radius: 8px; padding: 20px 24px;
          display: flex; align-items: flex-start; justify-content: space-between; position: relative;
          overflow: hidden;
        }
        .stage { display: flex; flex-direction: column; align-items: center; gap: 6px; position: relative; z-index: 1; flex: 1; }
        .stage::after {
          content: '';
          position: absolute;
          top: 14px;
          left: calc(50% + 18px);
          width: calc(100% - 36px);
          height: 0;
          border-top: 2px dashed #ccc;
          z-index: 0;
        }
        .stage:first-child::before { display: none; }
        .stage:last-child::after { display: none; }
        .stage-dot {
          width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
          font-size: 14px; font-weight: bold; border: 2px solid #ddd; background: #fff;
        }
        .stage-dot.done { background: #34a853; border-color: #34a853; color: #fff; }
        .stage-dot.active { border-color: #1a73e8; border-width: 3px; }
        .stage-dot.inactive { border-color: #ccc; }
        .stage-label { font-size: 11px; color: #555; white-space: nowrap; }

        .card { background: #fff; border: 1px solid #e8eaed; border-radius: 8px; padding: 20px; }
        .card-title { font-size: 18px; font-weight: 700; margin-bottom: 14px; color: #111; }

        .metrics { display: flex; gap: 12px; margin-bottom: 16px; }
        .metric { flex: 1; border: 1px solid #e8eaed; border-radius: 6px; padding: 10px 14px; }
        .metric-label { font-size: 11px; color: #777; margin-bottom: 4px; }
        .metric-value { font-size: 17px; font-weight: 700; }
        .metric-sub { font-size: 11px; color: #555; margin-left: 6px; }
 .deal-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.summary-image {
  width: 40px;
  height: 40px;
  object-fit: contain;
  flex-shrink: 0;
}
  .deal-content {
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  gap: 12px;
}

.deal-image {
  width: 40px;
  height: 40px;
  flex-shrink: 0;
}
      .deal-inner {
  position: relative;
  padding: 16px;
  border-radius: 8px;
  background: #fff;
}

.deal-inner::before {
  content: "";
  position: absolute;
  inset: 0;
  padding: 2px; /* border thickness */
  border-radius: 8px;
  background: linear-gradient(to bottom, #AD1A98, #3768C7);

  -webkit-mask:
    linear-gradient(#fff 0 0) content-box,
    linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;

  pointer-events: none;
}
        .deal-inner-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px; }
        .deal-inner-title { font-size: 14px; font-weight: 700; }
        .deal-close { font-size: 12px; color: #555; }
        .deal-id { font-size: 11px; color: #888; margin-bottom: 10px; }
        .deal-desc { font-size: 12px; color: #333; line-height: 1.6; margin-bottom: 12px; }
         .deal-meta { display: flex; justify-content: center; align-items-center; gap: 20px; font-size: 11px; color: #555; border-top: 1px solid #eee; margin-bottom:0;  }
        .deal-ai { font-size: 11px; color: #777; text-align: center; }
        .deal-ai a { color: #1a73e8; }

        .actions-header { display: flex; gap:20px; align-items: center; margin-bottom: 14px; }
        .actions-count { font-size: 12px; color: #555; }
        .due-today { color: #e2001a; font-weight: 600; }
        .action-item { border: 1px solid #e8eaed; border-radius: 8px; padding: 14px; }
        .action-item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .action-item-title { font-size: 13px; font-weight: 600; }
        .due-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 3px 10px;
          border-radius: 999px;
          background: #fef2f2;
          border: 1px solid #fecaca;
          color: #b91c1c;
          font-size: 11px;
          font-weight: 600;
          line-height: 1;
        }
        .due-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #dc2626;
          display: inline-block;
        }
        .action-item-desc { font-size: 12px; color: #555; margin-bottom: 12px; }
        .btn-action,
        .sd-action-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: #1D4ED8;
          color: #fff;
          font-size: 13px;
          font-weight: 600;
          font-family: 'Inter', sans-serif;
          padding: 8px 22px;
          border-radius: 50px !important;
          border: none;
          cursor: pointer;
          width: fit-content;
          margin-top: auto;
          align-self: flex-start;
          transition: background .15s;
        }
        .btn-action:hover,
        .sd-action-btn:hover { background: #1F3A8A; }

        .account-header { display: flex; align-items: center; margin-bottom: 14px; }
        .account-logo {
          width: 40px; height: 40px; background: #2c3e50; border-radius: 6px; display: flex;
          align-items: center; justify-content: center; color: #fff; font-size: 18px; margin-right: 12px;
        }
        .account-meta { font-size: 12px; color: #666; }
        .account-meta-line { font-size: 11px; color: #888; margin-top: 2px; }
        .account-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .account-field { background: #f9f9fb; border-radius: 6px; padding: 10px 12px; }
        .field-label { font-size: 10px; color: #888; margin-bottom: 3px; }
        .field-value { font-size: 13px; font-weight: 600; }
        .account-field.full { grid-column: 1 / -1; }

        .health-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .health-badge { font-size: 12px; color: #34a853; font-weight: 600; }
        .health-date { font-size: 11px; color: #888; margin-bottom: 14px; }
        .health-main {
          display: flex; justify-content: space-between; align-items: center;
          padding: 10px 0; border-bottom: 1px solid #eee; margin-bottom: 10px;
        }
        .health-main-label { font-size: 13px; font-weight: 600; }
        .health-main-val { font-size: 14px; font-weight: 700; color: #34a853; }
        .health-row {
          display: flex; justify-content: space-between; font-size: 12px; color: #555;
          padding: 5px 0; border-bottom: 1px solid #f0f0f0;
        }
        .health-row:last-child { border-bottom: none; }
        .health-pct { font-weight: 600; color: #333; }

        /* Activity tab layout (from provided design) */
        .dv-activity-container {
          padding: 0;
          background: #fafafa;
          border-radius: 12px;
        }
        .dv-activity-timeline {
          background: white;
          border: 1px solid #cbd5e1;
          border-radius: 12px;
          padding: 24px;
        }
        .dv-timeline {
          display: flex;
          flex-direction: column;
          gap: 0;
        }
        .dv-convo-row {
          display: grid;
          grid-template-columns: 1fr 48px 1fr;
          align-items: stretch;
          gap: 8px;
         
        }

        /* Ensure card margin does not skew row center alignment */
        .dv-convo-row .dv-activity-card {
          margin-bottom: 0;
        }

        .dv-activity-card {
          background: white;
          border: 1px solid #cbd5e1;
          border-radius: 12px;
          margin-bottom: 24px; /* default for non-conversation contexts */
          overflow: hidden;
        }

        .dv-convo-track {
          position: relative;
          width: 48px;
          min-height: 100%;
        }
        .dv-convo-line {
          width: 2px;
          height: 100%;
          background: #cbd5e1;
          position: absolute;
          left: 50%;
          top: 0;
          transform: translateX(-50%);
        }
        .dv-convo-dot {
          width: 24px;
          height: 24px;
          background: #1d4ed8;
          border: 2px solid #e2e8f0;
          border-radius: 12px;
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%);
          z-index: 1;
        }

        .dv-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 12px 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
          line-height: 1.2;
          white-space: nowrap;
          flex-shrink: 0;
        }

        /* restore badge backgrounds */
        .dv-badge-success { background: #ecfdf5; color: #0f172a; }
        .dv-badge-info { background: #eff6ff; color: #0f172a; }
        .dv-badge-warning { background: #fff7ed; color: #0f172a; }

        .dv-activity-body { padding: 16px; }
        .dv-activity-timestamp { font-size: 10px; color: #6a7282; margin-bottom: 8px; }
        .dv-activity-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
        .dv-activity-description { font-size: 14px; line-height: 22px; color: #334155; }

        .dv-timeline-track {
          width: 48px;
          position: relative;
          flex-shrink: 0;
        }
        .dv-timeline-line {
          width: 2px;
          height: 100%;
          background: #cbd5e1;
          position: absolute;
          top: 0;
          left: 50%;
          transform: translateX(-50%);
        }
        .dv-timeline-dot {
          width: 24px;
          height: 24px;
          background: #1d4ed8;
          border: 2px solid #e2e8f0;
          border-radius: 12px;
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%);
          z-index: 1;
        }

        .dv-sub-tabs {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 100%;
          margin-bottom: 16px;
          background: transparent;
        }
        .dv-sub-title {
          padding: 6px 12px;
          font-size: 16px;
          color: #334155;
          font-weight: 600;
          line-height: 1;
          white-space: nowrap;
          display: inline-flex;
          align-items: center;
        }
        .dv-sub-icon {
          width: 18px;
          height: 18px;
          object-fit: contain;
          vertical-align: middle;
        }

        .dv-activity-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 14px 16px;
          border-bottom: 1px solid #cbd5e1;
          min-height: 56px;
          flex-wrap: nowrap;
        }
        .dv-activity-type {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 16px;
          flex: 1;
          min-width: 0;
          white-space: nowrap;
        }

        /* Documents tab */
        .dv-doc-wrap {
          background: #fff;
          border: 1px solid #CBD5E1;
          border-radius: 12px;
          padding: 24px;
        }
        .dv-doc-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }

        .dv-doc-header .card-title {
          font-size: 18px;
        }

        .dv-doc-upload {
          border: none;
          cursor: pointer;
        }

        /* keep layout while using shared sd-action-btn visuals */
        .sd-action-btn.dv-doc-upload {
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }

        .dv-doc-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }
        @media (max-width: 900px) {
          .dv-doc-grid {
            grid-template-columns: 1fr;
          }
        }
        .dv-doc-card {
          background: #fff;
          border: 1px solid #CBD5E1;
          border-radius: 12px;
          padding: 16px;
        }
        .dv-doc-card:hover { box-shadow: 0 4px 6px rgba(0,0,0,.08); }
        .dv-doc-row { display: flex; align-items: flex-start; gap: 12px; }
        .dv-doc-icon {
          width: 48px;
          height: 48px;
          background: #f3f4f6;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          color: #6b7280;
        }
        .dv-doc-info { flex: 1; min-width: 0; }
        .dv-doc-name {
          font-size: 16px;
          font-weight: 500;
          color: #111827;
          margin-bottom: 4px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .dv-doc-meta { font-size: 14px; color: #6b7280; }
        .dv-doc-actions { display: flex; gap: 6px; flex-shrink: 0; }
        .dv-doc-btn {
          width: 36px;
          height: 36px;
          border: none;
          background: transparent;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: #6b7280;
        }
        .dv-doc-btn:hover { background: #f3f4f6; }
        .dv-doc-btn.dv-doc-delete { color: #ef4444; }
        .dv-doc-btn.dv-doc-delete:hover { background: #fef2f2; }

        /* Notes tab */
        .dv-notes-wrap { background: #fff; border: 1px solid #CBD5E1; border-radius: 12px; padding: 24px; }
        .dv-notes-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
        .dv-notes-header .card-title { font-size: 18px; margin-bottom: 0; }

        .dv-note-editor {
          background: #fff; border: 1px solid #CBD5E1; border-radius: 12px;
          padding: 24px; margin-bottom: 16px;
        }
        .dv-note-textarea {
          position: relative;
          width: 100%;
          min-height: 200px;
          resize: vertical;
          border: 2px solid transparent;
          border-radius: 8px;
          padding: 12px;
          font: inherit;
          background: linear-gradient(#F8FAFC, #F8FAFC) padding-box, linear-gradient(to bottom, #AD1A98, #3768C7) border-box;
          color: #111827;
          margin-bottom: 16px;
        }
        .dv-notes-upload {
          border: none;
          cursor: pointer;
        }
        .dv-note-clear {
          padding: 10px 16px; border: none; border-radius: 8px;
          background: #CBD5E1; color: #374151; font-weight: 500; cursor: pointer;
        }
        .dv-note-clear:hover { background: #d1d5db; }

        .dv-notes-list { display: flex; flex-direction: column; gap: 12px; }
        .dv-note-card {
          background: #fff; border: 1px solid #CBD5E1; border-radius: 12px;
          padding: 16px; display: flex; align-items: center; gap: 12px;
        }
        .dv-note-card:hover { box-shadow: 0 4px 6px rgba(0,0,0,.08); }

        .dv-note-icon {
          width: 48px; height: 48px; border-radius: 8px; background: #f3f4f6;
          display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #6b7280;
        }
        .dv-note-info { flex: 1; min-width: 0; }
        .dv-note-title {
          font-size: 16px; font-weight: 500; color: #111827; margin-bottom: 4px;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .dv-note-meta { font-size: 14px; color: #6b7280; }

        .dv-note-card-actions { display: flex; gap: 6px; flex-shrink: 0; }
        .dv-note-card-btn {
          width: 36px; height: 36px; border: none; background: transparent;
          border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; color: #6b7280;
        }
        .dv-note-card-btn:hover { background: #f3f4f6; }
        .dv-note-card-btn.dv-note-delete { color: #ef4444; }
        .dv-note-card-btn.dv-note-delete:hover { background: #fef2f2; }

        /* Execute tab */
        .dv-exec-wrap { background: #ffffffff; border-radius: 12px;  padding:20px}
        .dv-exec-section { margin-bottom: 20px; }
        .dv-exec-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        /* place title + meta on same line */
        .dv-exec-header > div {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .dv-exec-title {
          font-size: 18px;
          font-weight: 700;
          color: #111827;
          margin: 0;
        }
        .dv-exec-meta {
          display: flex;
          gap: 12px;
          font-size: 14px;
          color: #6b7280;
          margin-top: 0;
        }
        .dv-exec-view { color: #2563eb; text-decoration: none; font-size: 14px; font-weight: 500; }
        .dv-exec-view:hover { text-decoration: underline; }

        .dv-exec-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 12px;
        }
        .dv-exec-card {
          background: #fff;
          border: 1px solid #CBD5E1;
          border-radius: 12px;
          padding: 18px;
        }
        .dv-exec-card-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .dv-exec-card-title {
          font-size: 16px;
          font-weight: 700; /* bold title */
          color: #111827;
          margin-bottom: 0;
        }

        .dv-exec-tag {
          display: inline-flex; align-items: center; gap: 4px;
          padding: 4px 10px; border-radius: 999px;
          background: #fef2f2; color: #dc2626; font-size: 12px; font-weight: 500;
        }
        .dv-exec-details {
          display: flex; flex-wrap: wrap; gap: 8px;
          font-size: 13px; color: #6b7280; margin-bottom: 12px;
        }
        .dv-exec-dot { color: #9ca3af; }

        .sd-priority-meta {
          font-size: 12px;
          font-weight: 400;
          color: #6b7280;
        }

        .sd-priority-due-today {
          color: #E1251B;
          font-weight: 600;
        }

        /* Complete Information two-column layout */
        .dv-ci-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
        .dv-ci-col { display: flex; flex-direction: column; gap: 16px; }

        /* Form field helpers */
.fields {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

/* default 4-column layout */
.field {
  display: flex;
  flex-direction: column;
  gap: 4px;

  flex: 1 1 calc(25% - 16px);
  min-width: 0;
}

/* FULL WIDTH OVERRIDE */
.field.full {
  flex: 0 0 100% !important;
  max-width: 100%;
}
        .field label { font-weight: 600; font-size: 14px; line-height: 20px; color: #0f172a; }
        .field input, .field select {
          background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 4px;
          height: 40px; padding: 0 12px; font-size: 14px; color: #475569;
          width: 100%; outline: none; transition: border-color .15s;
        }
        .field input:focus, .field select:focus { border-color: #94a3b8; }
        .field input::placeholder { color: #94a3b8; }

        .select-wrap { position: relative; }
        .select-wrap select { padding-right: 30px; appearance: none; cursor: pointer; }
        .select-wrap::after {
          content: ''; position: absolute; right: 11px; top: 50%; transform: translateY(-50%);
          border-left: 5px solid transparent; border-right: 5px solid transparent;
          border-top: 5px solid #0f172a; pointer-events: none;
        }

        .search-wrap { position: relative; }
        .search-wrap input { padding-left: 34px; }
        .search-wrap svg { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; }

        .toggle-row { height: 40px; display: flex; align-items: center; }
        .toggle { position: relative; width: 40px; height: 24px; display: inline-block; }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; inset: 0; background: #334155; border-radius: 100px; cursor: pointer; transition: background .2s; }
        .slider::before {
          content: ''; position: absolute; width: 16px; height: 16px;
          background: #f8fafc; border-radius: 50%; top: 4px; left: 4px; transition: transform .2s;
        }
        .toggle input:checked + .slider { background: #3b82f6; }
        .toggle input:checked + .slider::before { transform: translateX(16px); }

        /* Contacts tab */
        .dv-ct-wrap { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; }
        .ct-card.ct-card-primary { background: #EFF6FF; }
        .dv-ct-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
        .dv-ct-header h2 { font-size: 20px; font-weight: 700; color: #0f172a; }
        .dv-ct-btn {
          display: flex; align-items: center; gap: 8px;
          background: #1e3a8a; color: #fff; border: none; border-radius: 8px;
          padding: 10px 20px; font-size: 14px; font-weight: 600; cursor: pointer;
        }
        .dv-ct-btn:hover { background: #1e40af; }

        .dv-ct-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

        .dv-ct-card { border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
        .dv-ct-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
        .dv-ct-name-block { display: flex; flex-direction: column; gap: 2px; }
        .dv-ct-name-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .dv-ct-name { font-size: 16px; font-weight: 700; color: #0f172a; }
        .dv-ct-div { width: 1px; height: 16px; background: #cbd5e1; flex-shrink: 0; }
        .dv-ct-role { font-size: 13px; color: #475569; }
        .dv-ct-company { font-size: 13px; color: #475569; }
        .dv-ct-badge {
          flex-shrink: 0; background: #f1f5f9; border: 1px solid #e2e8f0;
          border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 500;
          color: #334155; white-space: nowrap;
        }
        .dv-ct-info-row { display: flex; flex-direction: column; gap: 4px; }
        .dv-ct-info-line { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #475569; }
        .dv-ct-info-sep { width: 1px; height: 13px; background: #cbd5e1; flex-shrink: 0; }
        .dv-ct-actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
        .dv-ct-btn-action {
          display: flex; align-items: center; gap: 6px; border-radius: 8px;
          padding: 8px 16px; font-size: 13px; font-weight: 600; cursor: pointer;
        }
        .dv-ct-btn-email {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: #E2E8F0;
          color: #374151;
          border: none;
          font-size: 12.5px;
          font-weight: 600;
          font-family: 'Inter', sans-serif;
          padding: 7px 16px;
          border-radius: 50px;
          cursor: pointer;
          transition: background .15s, border-color .15s;
        }
        .dv-ct-btn-email:hover { background: #CBD5E1; }
        .dv-ct-btn-call {
          background: #1D4ED8 !important;
          border-radius: 50px !important;
          border: none !important;
          color: #fff !important;
          transition: background .15s;
        }
        .dv-ct-btn-call:hover { background: #1F3A8A !important; }

        /* Contact Picker Modal */
        .ct-modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.45);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .ct-modal {
          background: #fff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18);
          width: 480px;
          max-width: 95vw;
          max-height: 80vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        .ct-modal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 20px;
          border-bottom: 1px solid #e2e8f0;
        }
        .ct-modal-header h3 {
          font-size: 17px;
          font-weight: 700;
          color: #0f172a;
          margin: 0;
        }
        .ct-modal-header button {
          background: transparent;
          border: none;
          font-size: 18px;
          color: #64748b;
          cursor: pointer;
          line-height: 1;
          padding: 2px 6px;
          border-radius: 4px;
        }
        .ct-modal-header button:hover { background: #f1f5f9; color: #0f172a; }
        .ct-modal-list {
          flex: 1;
          overflow-y: auto;
          padding: 12px 16px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .ct-modal-item {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          padding: 12px 14px;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          cursor: pointer;
          transition: border-color 0.15s, background 0.15s;
          background: #fff;
        }
        .ct-modal-item:hover { background: #f8fafc; border-color: #94a3b8; }
        .ct-modal-item.selected {
          background: #eff6ff;
          border-color: #3b82f6;
        }
        .ct-modal-item input[type="checkbox"] {
          margin-top: 2px;
          width: 16px;
          height: 16px;
          flex-shrink: 0;
          accent-color: #1d4ed8;
          cursor: pointer;
        }
        .ct-modal-item-info { display: flex; flex-direction: column; gap: 2px; }
        .ct-modal-item-name { font-size: 14px; font-weight: 600; color: #0f172a; }
        .ct-modal-item-role { font-size: 12px; color: #475569; }
        .ct-modal-item-email { font-size: 12px; color: #94a3b8; }
        .ct-modal-footer {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
          padding: 14px 20px;
          border-top: 1px solid #e2e8f0;
          background: #f8fafc;
        }
        .ct-modal-footer button {
          padding: 8px 20px;
          border-radius: 8px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          border: none;
        }
        .ct-modal-footer button:first-child {
          background: #e2e8f0;
          color: #374151;
        }
        .ct-modal-footer button:first-child:hover { background: #cbd5e1; }
        .ct-modal-footer button:last-child {
          background: #1d4ed8;
          color: #fff;
        }
        .ct-modal-footer button:last-child:hover { background: #1e3a8a; }

        /* Custom checkbox for contact picker */
        .ct-picker-checkbox:checked::after {
          content: '✓';
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          color: #fff;
          font-size: 11px;
          font-weight: 700;
          line-height: 1;
        }

      `}</style>

      <div className="dv-page">
        <div className="main">
          {/* Quarter bar */}
          <div className="quarter-bar">
            Q2 FY2024 · Week 10 of 12 ·{" "}
            <span className="closure-badge">Closure Phase</span>
          </div>

          {/* Page header */}
          <div className="page-header">
            <div className="page-header-row">
              <button
                className="back-btn"
                onClick={() => navigate("/opportunities")}
              >
                <i
                  className="bi bi-chevron-left"
                  style={{
                    fontSize: 16,
                    WebkitTextStroke: "1px",
                    color: "#0F172A",
                  }}
                ></i>
              </button>
              <span className="topbar-title">
                {opportunityDetails?.id
                  ? `LVO-${opportunityDetails.id.slice(-3)}`
                  : ""}{" "}
                - {opportunityDetails?.name}
              </span>
              {/* <div className="topbar-actions">
                <button className="sd-btn-prep">Edit</button>
                <button className="btn-delete" aria-label="Delete">
                  <i className="bi bi-trash"></i>
                </button>
              </div> */}
            </div>
            <div className="tabs-outer">
              <div className="tabs">
                <button
                  className={`tab ${activeTopTab === "overview" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("overview")}
                >
                  Overview
                </button>
                <button
                  className={`tab ${activeTopTab === "complete" ? "active" : ""}`}
                  onClick={() => setActiveTopTab("complete")}
                >
                  Complete Information
                </button>
                <button
                  className={`tab ${activeTopTab === "activity" ? "active" : ""}`}
                  onClick={() => {
                    setActiveTopTab("activity");
                    loadActivities();
                  }}
                >
                  Activity
                </button>
                <button
                  className={`tab ${activeTopTab === "contacts" ? "active" : ""}`}
                  onClick={() => {
                    setActiveTopTab("contacts");
                    loadContactsData();
                  }}
                >
                  Contacts
                </button>
                <button
                  className={`tab ${activeTopTab === "notes" ? "active" : ""}`}
                  onClick={async () => {
                    setActiveTopTab("notes");
                    try {
                      const res = await fetchOpportunityNotes(id);
                      if (res && res.length > 0) {
                        setNoteText(res[0].notes || "");
                      }
                    } catch (err) {
                      console.error("Failed to fetch notes:", err);
                    }
                  }}
                >
                  Files & Notes
                </button>
              </div>
            </div>
          </div>

          {/* Content */}
          {activeTopTab === "overview" && (
            <div
              className="pipeline"
              style={{
                margin: "20px 24px 0",
                border: "none",
                background: "transparent",
                padding: "10px 0",
              }}
            >
              {stages.map((stage, i) => {
                let dotClass = "stage-dot";
                if (i <= currentIndex) dotClass += " done";
                else dotClass += " inactive";

                return (
                  <div className="stage" key={stage.key}>
                    <div className={dotClass}>
                      {i <= currentIndex ? "✓" : ""}
                    </div>
                    <div className="stage-label">
                      {i === currentIndex ? (
                        <strong>
                          {stage.label}
                          {opportunityDetails?.daysInStage
                            ? ` (${opportunityDetails.daysInStage}D)`
                            : ""}
                        </strong>
                      ) : (
                        stage.label
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div className="content">
            <div className="col-main">
              {activeTopTab === "activity" ? (
                <div className="dv-activity-container">
                  <div className="dv-activity-timeline">
                    <div className="dv-sub-tabs">
                      <span className="dv-sub-title">
                        <img
                          src={leftArrowIcon}
                          alt="left"
                          className="dv-sub-icon"
                          style={{ marginRight: 5 }}
                        />
                        Inbound
                      </span>
                      <span className="dv-sub-title">
                        Outbound
                        <img
                          src={rightArrowIcon}
                          alt="right"
                          className="dv-sub-icon"
                          style={{ marginLeft: 5 }}
                        />
                      </span>
                    </div>

                    <div className="dv-timeline">
                      {activities.items?.map((act, i) => (
                        <div className="dv-convo-row" key={act.id}>
                          {/* LEFT SIDE (Inbound) */}
                          <div className="dv-convo-side">
                            {act.direction === "inbound" && (
                              <div className="dv-activity-card">
                                <div className="dv-activity-header">
                                  <div className="dv-activity-type">
                                    <span>{getTypeLabel(act.type)}</span>
                                  </div>

                                  <span
                                    className={`dv-badge ${getBadgeClass(act.type)}`}
                                  >
                                    {getBadge(act.type)}
                                  </span>
                                </div>

                                <div className="dv-activity-body">
                                  <div className="dv-activity-timestamp">
                                    {formatDate(act.eventDate)}
                                  </div>
                                  <div className="dv-activity-title">
                                    {act.subject}
                                  </div>
                                  <div className="dv-activity-description">
                                    {act.body}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>

                          {/* CENTER LINE */}
                          <div className="dv-convo-track">
                            <div className="dv-convo-line"></div>
                            <div className="dv-convo-dot"></div>
                          </div>

                          {/* RIGHT SIDE (Outbound) */}
                          <div className="dv-convo-side">
                            {act.direction === "outbound" && (
                              <div className="dv-activity-card">
                                <div className="dv-activity-header">
                                  <div className="dv-activity-type">
                                    <span>{getTypeLabel(act.type)}</span>
                                  </div>
                                  <span
                                    className={`dv-badge ${getBadgeClass(act.type)}`}
                                  >
                                    {getBadge(act.type)}
                                  </span>
                                </div>
                                <div className="dv-activity-body">
                                  <div className="dv-activity-timestamp">
                                    {formatDate(act.eventDate)}
                                  </div>
                                  <div className="dv-activity-title">
                                    {act.subject}
                                  </div>
                                  <div className="dv-activity-description">
                                    {act.body}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : activeTopTab === "complete" ? (
                <>
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">Deal Summary</div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field full">
                        <label>Summary</label>
                        <input
                          name="summary"
                          value={formData.summary || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Account Name</label>
                        <input
                          name="accountName"
                          value={formData.accountName || ""}
                          onChange={handleChange}
                          placeholder="Input"
                        />
                      </div>

                      <div className="ct-field">
                        <label>Owner</label>
                        <input
                          name="ownerName"
                          value={formData.ownerName || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Region</label>
                        <select
                          name="region"
                          value={formData.region}
                          onChange={handleChange}
                        >
                          <option value="APAC">APAC</option>
                          <option value="EMEA">EMEA</option>
                          <option value="NA">NA</option>
                          <option value="LATAM">LATAM</option>
                        </select>
                      </div>

                      <div className="ct-field">
                        <label>Lead Origin</label>
                        <select
                          name="leadOrigin"
                          value={formData.leadOrigin}
                          onChange={handleChange}
                        >
                          <option value="Partner">Partner</option>
                          <option value="Direct">Direct</option>
                          <option value="Marketing">Marketing</option>
                          <option value="Inbound">Inbound</option>
                        </select>
                      </div>
                      <div className="ct-field">
                        <label>Priority</label>
                        <select
                          name="priority"
                          value={formData.priority}
                          onChange={handleChange}
                        >
                          <option value="">Select</option>
                          <option value="High">High</option>
                          <option value="Medium">Medium</option>
                          <option value="Low">Low</option>
                        </select>
                      </div>
                      <div className="ct-field">
                        <label>Partner Involved</label>
                        <div className="ct-toggle-row">
                          <label className="ct-toggle">
                            <input
                              type="checkbox"
                              checked={isDecisionMaker}
                              onChange={(e) =>
                                setIsDecisionMaker(e.target.checked)
                              }
                            />
                            <span className="ct-toggle-slider"></span>
                          </label>
                        </div>
                      </div>
                      {/* <div className="ct-field">
                      <label>Partner Involved</label>
                      <input
                        type="checkbox"
                        name="partnerInvolved"
                        checked={formData.partnerInvolved || ""}
                        onChange={handleChange}
                      />
                    </div> */}
                      <div className="ct-field">
                        <label>Parent Opportunity</label>
                        <input
                          name="parentOpportunityName"
                          value={formData.parentOpportunityName || ""}
                          onChange={handleChange}
                          placeholder="Select"
                        />
                      </div>
                      <div className="ct-field">
                        <label>Child Opportunity</label>
                        <input
                          name="childOpportunities"
                          value={formData.childOpportunities || ""}
                          onChange={handleChange}
                          placeholder="Select"
                        />
                      </div>
                      <div className="ct-field">
                        <label>Total Account Value</label>
                        <input
                          name="totalAccountValue"
                          value={formData.totalAccountValue || ""}
                          onChange={handleChange}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Open Deals Count</label>
                        <input
                          name="openDealsCount"
                          value={formData.openDealsCount || ""}
                          onChange={handleChange}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Account Segment</label>
                        <input
                          name="accountSegment"
                          value={formData.accountSegment || ""}
                          onChange={handleChange}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Account Industry</label>
                        <input
                          name="accountIndustry"
                          value={formData.accountIndustry || ""}
                          onChange={handleChange}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Account Employee Count</label>
                        <input
                          name="accountEmployeeCount"
                          value={formData.accountEmployeeCount || ""}
                          onChange={handleChange}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Days in Stage</label>
                        <input
                          name="daysInStage"
                          value={formData.daysInStage || ""}
                          onChange={handleChange}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Stage Entry Date</label>
                        <input
                          type="date"
                          name="stageEntryDate"
                          value={formData?.stageEntryDate?.slice(0, 10) || ""}
                          onChange={handleChange}
                          onClick={(e) => e.target.showPicker?.()}
                        />
                      </div>

                      <div className="ct-field">
                        <label>Probability</label>
                        <input
                          name="probability"
                          value={formData.closeProbability}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Created Date</label>
                        <input
                          type="date"
                          name="createdDate"
                          value={formData.createdAt?.slice(0, 10) || ""}
                          readOnly
                          disabled
                        />
                      </div>

                      <div className="ct-field">
                        <label>Created By</label>
                        <input
                          name="createdBy"
                          value={formData.createdBy || ""}
                          readOnly
                          disabled
                        />
                      </div>

                      <div className="ct-field">
                        <label>Modified Date</label>
                        <input
                          type="date"
                          name="modifiedDate"
                          value={formData.modifiedAt?.slice(0, 10) || ""}
                          readOnly
                          disabled
                        />
                      </div>
                      <div className="ct-field">
                        <label>Modified By</label>
                        <input
                          name="modifiedBy"
                          value={formData.modifiedBy || ""}
                          readOnly
                          disabled
                        />
                      </div>
                    </div>
                    <div className="ct-bottom-bar">
                      <button
                        className="ct-btn-cancel"
                        // onClick={handleCloseEdit}
                      >
                        Cancel
                      </button>
                      <button
                        className="ct-btn-delete"
                        onClick={() => {
                          setConfirmOpportunityDelete(false);
                          setShowOpportunityDeleteModal(true);
                        }}
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

                  {/* Next Steps & Actions + Verbal Commit */}
                  <div
                    style={{
                      display: "flex",
                      gap: 16,
                      alignItems: "flex-start",
                    }}
                  >
                    {/* Next Steps & Actions card */}
                    <div className="ct-new-form" style={{ flex: 1 }}>
                      <div className="ct-form-header">
                        <div className="ct-form-title">
                          Next Steps & Actions
                        </div>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 16,
                        }}
                      >
                        <div style={{ flex: 1 }}>
                          {nextActionsList.map((item, idx) => (
                            <div
                              className="ct-fields"
                              key={idx}
                              style={{ marginBottom: 12, position: "relative" }}
                            >
                              {idx === 0 ? (
                                <>
                                  <div className="ct-field">
                                    <label>Next Action</label>
                                    <input
                                      value={item.nextAction}
                                      onChange={(e) => {
                                        const updated = [...nextActionsList];
                                        updated[idx].nextAction =
                                          e.target.value;
                                        setNextActionsList(updated);
                                      }}
                                    />
                                  </div>
                                  <div className="ct-field">
                                    <label>Next Action Due Date</label>
                                    <input
                                      type="date"
                                      value={
                                        item.nextActionDueDate?.slice(0, 10) ||
                                        ""
                                      }
                                      onChange={(e) => {
                                        const updated = [...nextActionsList];
                                        updated[idx].nextActionDueDate =
                                          e.target.value;
                                        setNextActionsList(updated);
                                      }}
                                      onClick={(e) => e.target.showPicker?.()}
                                    />
                                  </div>
                                </>
                              ) : (
                                <>
                                  <div className="ct-field">
                                    <input
                                      placeholder="Next Action"
                                      value={item.nextAction}
                                      onChange={(e) => {
                                        const updated = [...nextActionsList];
                                        updated[idx].nextAction =
                                          e.target.value;
                                        setNextActionsList(updated);
                                      }}
                                    />
                                  </div>
                                  <div
                                    className="ct-field"
                                    style={{
                                      display: "flex",
                                      flexDirection: "row",
                                      alignItems: "center",
                                      gap: 8,
                                    }}
                                  >
                                    <input
                                      type="date"
                                      placeholder="Next Action Due Date"
                                      value={
                                        item.nextActionDueDate?.slice(0, 10) ||
                                        ""
                                      }
                                      style={{ flex: 1 }}
                                      onChange={(e) => {
                                        const updated = [...nextActionsList];
                                        updated[idx].nextActionDueDate =
                                          e.target.value;
                                        setNextActionsList(updated);
                                      }}
                                      onClick={(e) => e.target.showPicker?.()}
                                    />
                                    <button
                                      onClick={() => {
                                        const updated = nextActionsList.filter(
                                          (_, i) => i !== idx,
                                        );
                                        setNextActionsList(updated);
                                      }}
                                      style={{
                                        background: "transparent",
                                        border: "none",
                                        cursor: "pointer",
                                        color: "#ef4444",
                                        fontSize: 18,
                                        padding: 4,
                                        display: "flex",
                                        alignItems: "center",
                                        flexShrink: 0,
                                      }}
                                      title="Remove"
                                    >
                                      <svg
                                        width="20"
                                        height="20"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      >
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                      </svg>
                                    </button>
                                  </div>
                                </>
                              )}
                            </div>
                          ))}
                        </div>
                        <button
                          className="sd-action-btn"
                          style={{ flexShrink: 0 }}
                          onClick={() =>
                            setNextActionsList([
                              ...nextActionsList,
                              { nextAction: "", nextActionDueDate: "" },
                            ])
                          }
                        >
                          Add New
                        </button>
                      </div>
                    </div>

                    {/* Verbal Commit card */}
                    <div className="ct-new-form" style={{ flex: 1 }}>
                      <div className="ct-form-header">
                        <div className="ct-form-title">Verbal Commit</div>
                      </div>
                      <div className="ct-fields">
                        <div className="ct-field">
                          <label>Verbal Commit Date</label>
                          <input
                            type="date"
                            name="verbalCommitDate"
                            value={
                              formData.verbalCommitDate?.slice(0, 10) || ""
                            }
                            onChange={handleChange}
                            onClick={(e) => e.target.showPicker?.()}
                          />
                        </div>
                        <div className="">
                          <label>Verbal / Written Acceptance</label>
                          <div className="ct-toggle-row">
                            <label className="ct-toggle">
                              <input
                                type="checkbox"
                                name="verbalWrittenAcceptance"
                                checked={
                                  formData.verbalWrittenAcceptance || false
                                }
                                onChange={handleChange}
                              />
                              <span className="ct-toggle-slider"></span>
                            </label>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Competitors */}
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">Competitors</div>
                    </div>
                    <div
                      style={{ display: "flex", alignItems: "center", gap: 16 }}
                    >
                      <div style={{ flex: 1 }}>
                        {competitorsList.map((item, idx) => (
                          <div
                            className="ct-fields"
                            key={idx}
                            style={{ marginBottom: 12, position: "relative" }}
                          >
                            {idx === 0 ? (
                              <>
                                <div className="ct-field">
                                  <label>Competitors</label>
                                  <input
                                    value={item.competitor}
                                    onChange={(e) => {
                                      const updated = [...competitorsList];
                                      updated[idx].competitor = e.target.value;
                                      setCompetitorsList(updated);
                                    }}
                                  />
                                </div>
                                <div className="ct-field">
                                  <label>Competitor Type</label>
                                  <select
                                    value={item.competitorType}
                                    onChange={(e) => {
                                      const updated = [...competitorsList];
                                      updated[idx].competitorType =
                                        e.target.value;
                                      setCompetitorsList(updated);
                                    }}
                                  >
                                    <option value="">Select</option>
                                    <option value="Incumbent">Incumbent</option>
                                    <option value="Secondary">Secondary</option>
                                  </select>
                                </div>
                                <div className="ct-field">
                                  <label>Reselling Partner</label>
                                  <input
                                    value={item.resellingPartner}
                                    onChange={(e) => {
                                      const updated = [...competitorsList];
                                      updated[idx].resellingPartner =
                                        e.target.value;
                                      setCompetitorsList(updated);
                                    }}
                                  />
                                </div>
                                <div
                                  className="ct-field"
                                  style={{
                                    flex: "0 0 auto",
                                    alignSelf: "flex-end",
                                  }}
                                >
                                  <button
                                    onClick={async () => {
                                      try {
                                        const { updateCompetitor } =
                                          await import("../api/client");
                                        await updateCompetitor(id, {
                                          competitorName: item.competitor,
                                          competitorType: item.competitorType,
                                          resellingPartner:
                                            item.resellingPartner,
                                        });
                                        setMessage(
                                          "Competitor saved successfully",
                                        );
                                        setShowSuccessModal(true);
                                      } catch (err) {
                                        console.error(err);
                                        setMessage("Failed to save competitor");
                                        setShowSuccessModal(true);
                                      }
                                    }}
                                    style={{
                                      background: "transparent",
                                      border: "none",
                                      cursor: "pointer",
                                      color: "#1D4ED8",
                                      padding: 4,
                                      display: "flex",
                                      alignItems: "center",
                                    }}
                                    title="Save Competitor"
                                  >
                                    <svg
                                      width="18"
                                      height="18"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      stroke="currentColor"
                                      strokeWidth="2"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                    >
                                      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                    </svg>
                                  </button>
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="ct-field">
                                  <input
                                    placeholder="Competitors"
                                    value={item.competitor}
                                    onChange={(e) => {
                                      const updated = [...competitorsList];
                                      updated[idx].competitor = e.target.value;
                                      setCompetitorsList(updated);
                                    }}
                                  />
                                </div>
                                <div className="ct-field">
                                  <select
                                    value={item.competitorType}
                                    onChange={(e) => {
                                      const updated = [...competitorsList];
                                      updated[idx].competitorType =
                                        e.target.value;
                                      setCompetitorsList(updated);
                                    }}
                                  >
                                    <option value="">Select</option>
                                    <option value="Incumbent">Incumbent</option>
                                    <option value="Secondary">Secondary</option>
                                  </select>
                                </div>
                                <div
                                  className="ct-field"
                                  style={{
                                    display: "flex",
                                    flexDirection: "row",
                                    alignItems: "center",
                                    gap: 8,
                                  }}
                                >
                                  <input
                                    placeholder="Reselling Partner"
                                    value={item.resellingPartner}
                                    style={{ flex: 1 }}
                                    onChange={(e) => {
                                      const updated = [...competitorsList];
                                      updated[idx].resellingPartner =
                                        e.target.value;
                                      setCompetitorsList(updated);
                                    }}
                                  />
                                  <button
                                    onClick={async () => {
                                      try {
                                        const { updateCompetitor } =
                                          await import("../api/client");
                                        await updateCompetitor(id, {
                                          competitorName: item.competitor,
                                          competitorType: item.competitorType,
                                          resellingPartner:
                                            item.resellingPartner,
                                        });
                                        setMessage(
                                          "Competitor saved successfully",
                                        );
                                        setShowSuccessModal(true);
                                      } catch (err) {
                                        console.error(err);
                                        setMessage("Failed to save competitor");
                                        setShowSuccessModal(true);
                                      }
                                    }}
                                    style={{
                                      background: "transparent",
                                      border: "none",
                                      cursor: "pointer",
                                      color: "#1D4ED8",
                                      padding: 4,
                                      display: "flex",
                                      alignItems: "center",
                                      flexShrink: 0,
                                    }}
                                    title="Save Competitor"
                                  >
                                    <svg
                                      width="18"
                                      height="18"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      stroke="currentColor"
                                      strokeWidth="2"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                    >
                                      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                    </svg>
                                  </button>
                                  <button
                                    onClick={() => {
                                      const updated = competitorsList.filter(
                                        (_, i) => i !== idx,
                                      );
                                      setCompetitorsList(updated);
                                    }}
                                    style={{
                                      background: "transparent",
                                      border: "none",
                                      cursor: "pointer",
                                      color: "#ef4444",
                                      fontSize: 18,
                                      padding: 4,
                                      display: "flex",
                                      alignItems: "center",
                                      flexShrink: 0,
                                    }}
                                    title="Remove"
                                  >
                                    <svg
                                      width="20"
                                      height="20"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      stroke="currentColor"
                                      strokeWidth="2"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                    >
                                      <line x1="18" y1="6" x2="6" y2="18" />
                                      <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                  </button>
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                      <button
                        className="sd-action-btn"
                        style={{ flexShrink: 0 }}
                        onClick={() =>
                          setCompetitorsList([
                            ...competitorsList,
                            {
                              competitor: "",
                              competitorType: "",
                              resellingPartner: "",
                            },
                          ])
                        }
                      >
                        Add New
                      </button>
                    </div>
                  </div>

                  {/* Order and Fulfilment */}
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">Order and Fulfilment</div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field">
                        <label>Sell Through Week (auto)</label>
                        <input
                          name="sellThroughWeek"
                          value={formData.sellThroughWeek || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Competitor Type</label>
                        <input
                          name="orderCompetitorType"
                          value={formData.orderCompetitorType || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Order Date</label>
                        <input
                          type="date"
                          name="orderDate"
                          value={formData.orderDate?.slice(0, 10) || ""}
                          onChange={handleChange}
                          onClick={(e) => e.target.showPicker?.()}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Shipping Date</label>
                        <input
                          type="date"
                          name="shippingDate"
                          value={formData.shippingDate?.slice(0, 10) || ""}
                          onChange={handleChange}
                          onClick={(e) => e.target.showPicker?.()}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Sales Order Reference / PO</label>
                        <input
                          name="salesOrderRef"
                          value={formData.salesOrderRef || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Order Number</label>
                        <input
                          name="orderNumber"
                          value={formData.orderNumber || ""}
                          onChange={handleChange}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Internal Stakeholder */}
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">Internal Stakeholder</div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field">
                        <label>Solution & Service Executive</label>
                        <input
                          name="solutionServiceExecutive"
                          value={formData.solutionServiceExecutive || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Solution & Service Domain Specialist</label>
                        <input
                          name="solutionServiceDomainSpecialist"
                          value={formData.solutionServiceDomainSpecialist || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>LGFS Sales Representatives</label>
                        <input
                          name="lgfsSalesReps"
                          value={formData.lgfsSalesReps || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>LGFS Sales Support</label>
                        <input
                          name="lgfsSalesSupport"
                          value={formData.lgfsSalesSupport || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Deal Desk Analyst</label>
                        <input
                          name="dealDeskAnalyst"
                          value={formData.dealDeskAnalyst || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Deal Engagement Manager</label>
                        <input
                          name="dealEngagementManager"
                          value={formData.dealEngagementManager || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>SSDS Channel</label>
                        <input
                          name="ssdsChannel"
                          value={formData.ssdsChannel || ""}
                          onChange={handleChange}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Deal Qualification & Governance */}
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">
                        Deal Qualification & Governance
                      </div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field">
                        <label>Forecast Category</label>
                        <input
                          name="forecastCategory"
                          value={formData.forecastCategory || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Sales Motion</label>
                        <input
                          name="saleMotion"
                          value={
                            formData.saleMotion?.raw ||
                            formData.saleMotion ||
                            ""
                          }
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Deal Tempo Class</label>
                        <input
                          name="tempoClass"
                          value={formData.tempoClass || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Deal Protection Status</label>
                        <input
                          name="dealProtectionStatus"
                          value={formData.dealProtectionStatus || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Deal Registration Ref</label>
                        <input
                          name="dealRegistrationRef"
                          value={formData.dealRegistrationRef || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Number of Countries</label>
                        <input
                          name="numberOfCountries"
                          value={formData.numberOfCountries || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>SOW Required</label>
                        <div className="ct-toggle-row">
                          <label className="ct-toggle">
                            <input
                              type="checkbox"
                              name="sowRequired"
                              checked={formData.sowRequired || false}
                              onChange={handleChange}
                            />
                            <span className="ct-toggle-slider"></span>
                          </label>
                        </div>
                      </div>
                      <div className="ct-field">
                        <label>Multi-Country Solution Required</label>
                        <div className="ct-toggle-row">
                          <label className="ct-toggle">
                            <input
                              type="checkbox"
                              name="multiCountrySolution"
                              checked={formData.multiCountrySolution || false}
                              onChange={handleChange}
                            />
                            <span className="ct-toggle-slider"></span>
                          </label>
                        </div>
                      </div>
                      <div className="ct-field">
                        <label>Deal Qualification Review</label>
                        <div className="ct-toggle-row">
                          <label className="ct-toggle">
                            <input
                              type="checkbox"
                              name="dealQualificationReview"
                              checked={
                                formData.dealQualificationReview || false
                              }
                              onChange={handleChange}
                            />
                            <span className="ct-toggle-slider"></span>
                          </label>
                        </div>
                      </div>
                      <div className="ct-field">
                        <label>Solution Handover Artefacts</label>
                        <input
                          name="solutionHandoverArtefacts"
                          value={formData.solutionHandoverArtefacts || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>RFP Reference</label>
                        <input
                          name="rfpReference"
                          value={formData.rfpReference || ""}
                          onChange={handleChange}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Commercial */}
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">Commercial</div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field">
                        <label>Estimated Deal Value</label>
                        <input
                          name="estimatedDealValue"
                          value={formData.estimatedDealValue || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Estimated Close Date</label>
                        <input
                          type="date"
                          name="estimatedCloseDate"
                          value={
                            formData.estimatedCloseDate?.slice(0, 10) || ""
                          }
                          onChange={handleChange}
                          onClick={(e) => e.target.showPicker?.()}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Budget Confirmed</label>
                        <div className="ct-toggle-row">
                          <label className="ct-toggle">
                            <input
                              type="checkbox"
                              name="budgetConfirmed"
                              checked={formData.budgetConfirmed || false}
                              onChange={handleChange}
                            />
                            <span className="ct-toggle-slider"></span>
                          </label>
                        </div>
                      </div>
                      <div className="ct-field">
                        <label>Quote Reference</label>
                        <input
                          name="quoteReference"
                          value={formData.quoteReference || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Partner Commercial Model</label>
                        <input
                          name="partnerCommercialModel"
                          value={formData.partnerCommercialModel || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Actual / Confirmed Revenue</label>
                        <input
                          name="actualConfirmedRevenue"
                          value={formData.actualConfirmedRevenue || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Reseller / Channel Account</label>
                        <input
                          name="resellerChannelAccount"
                          value={formData.resellerChannelAccount || ""}
                          onChange={handleChange}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Solution */}
                  <div className="ct-new-form">
                    <div className="ct-form-header">
                      <div className="ct-form-title">Solution</div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field">
                        <label>Solution Area</label>
                        <input
                          name="solutionArea"
                          value={formData.solutionArea || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Sub-Solution Area</label>
                        <input
                          name="subSolutionArea"
                          value={formData.subSolutionArea || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Solution Certifications</label>
                        <input
                          name="solutionCertifications"
                          value={formData.solutionCertifications || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Solution Offerings</label>
                        <input
                          name="solutionOfferings"
                          value={formData.solutionOfferings || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Leasing Vendor</label>
                        <input
                          name="leasingVendor"
                          value={formData.leasingVendor || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Sales Model</label>
                        <input
                          name="salesModel"
                          value={formData.salesModel || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Service Model</label>
                        <input
                          name="serviceModel"
                          value={formData.serviceModel || ""}
                          onChange={handleChange}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Close & Outcome */}
                  <div className="ct-new-form" style={{ marginBottom: 24 }}>
                    <div className="ct-form-header">
                      <div className="ct-form-title">Close & Outcome</div>
                    </div>
                    <div className="ct-fields">
                      <div className="ct-field">
                        <label>Actual Revenue</label>
                        <input
                          name="actualRevenue"
                          value={formData.actualRevenue || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Actual Close Date</label>
                        <input
                          type="date"
                          name="actualCloseDate"
                          value={formData.actualCloseDate?.slice(0, 10) || ""}
                          onChange={handleChange}
                          onClick={(e) => e.target.showPicker?.()}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Close Reason</label>
                        <input
                          name="closeReason"
                          value={formData.closeReason || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Sales Order Reference</label>
                        <input
                          name="salesOrderReference"
                          value={formData.salesOrderReference || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Won Solution Category</label>
                        <input
                          name="wonSolutionCategory"
                          value={formData.wonSolutionCategory || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Win Notes / Commentary</label>
                        <input
                          name="winNotesCommentary"
                          value={formData.winNotesCommentary || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Invoice Number</label>
                        <input
                          name="invoiceNumber"
                          value={formData.invoiceNumber || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Loss Reason</label>
                        <input
                          name="lossReason"
                          value={formData.lossReason || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Competitor Won</label>
                        <input
                          name="competitorWon"
                          value={formData.competitorWon || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Lost Solution Category</label>
                        <input
                          name="lostSolutionCategory"
                          value={formData.lostSolutionCategory || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Lost Revenue Value</label>
                        <input
                          name="lostRevenueValue"
                          value={formData.lostRevenueValue || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Loss Notes / Commentary</label>
                        <input
                          name="lossNotesCommentary"
                          value={formData.lossNotesCommentary || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Deal Appeal</label>
                        <input
                          name="dealAppeal"
                          value={formData.dealAppeal || ""}
                          onChange={handleChange}
                        />
                      </div>
                      <div className="ct-field">
                        <label>Re-engagement Date</label>
                        <input
                          type="date"
                          name="reEngagementDate"
                          value={formData.reEngagementDate?.slice(0, 10) || ""}
                          onChange={handleChange}
                          onClick={(e) => e.target.showPicker?.()}
                        />
                      </div>
                    </div>
                  </div>
                </>
              ) : activeTopTab === "contacts" ? (
                <div className="dv-ct-wrap">
                  {!editingContact && (
                    <div className="dv-ct-header">
                      <h2>Contact List</h2>
                      <div style={{ position: "relative" }} ref={pickerRef}>
                        <button
                          className="dv-ct-btn sd-action-btn"
                          onClick={loadAvailableContacts}
                        >
                          <svg
                            width="18"
                            height="18"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                            <circle cx="9" cy="7" r="4" />
                            <line x1="19" y1="8" x2="19" y2="14" />
                            <line x1="22" y1="11" x2="16" y2="11" />
                          </svg>
                          New Contact
                        </button>
                        {showContactPicker && (
                          <div
                            style={{
                              position: "absolute",
                              top: "calc(100% + 8px)",
                              right: 0,
                              width: 256,
                              background: "#fff",
                              border: "1px solid #e2e8f0",
                              borderRadius: 10,
                              boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                              zIndex: 100,
                              display: "flex",
                              flexDirection: "column",
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                padding: "10px 12px 8px",
                                borderBottom: "1px solid #e2e8f0",
                              }}
                            >
                              <i
                                className="ri-search-line"
                                style={{
                                  fontSize: 16,
                                  color: "#94a3b8",
                                  flexShrink: 0,
                                }}
                              ></i>
                              <input
                                type="text"
                                placeholder="Search contact"
                                value={contactSearch}
                                onChange={(e) =>
                                  setContactSearch(e.target.value)
                                }
                                autoFocus
                                style={{
                                  width: "100%",
                                  padding: 0,
                                  fontSize: 13,
                                  border: "none",
                                  outline: "none",
                                  background: "transparent",
                                  color: "#0f172a",
                                }}
                              />
                            </div>
                            <div
                              style={{
                                overflowY: "auto",
                                maxHeight: 208,
                                padding: "4px 10px 6px",
                                display: "flex",
                                flexDirection: "column",
                                gap: 0,
                              }}
                            >
                              {availableContacts
                                .filter((c) =>
                                  c.name
                                    ?.toLowerCase()
                                    .includes(contactSearch.toLowerCase()),
                                )
                                .map((c) => {
                                  const isAdded = contacts.some(
                                    (x) => x.id === c.id,
                                  );
                                  return (
                                    <label
                                      key={c.id}
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                        padding: "8px 8px",
                                        cursor: "pointer",
                                        borderBottom: "1px solid #e2e8f0",
                                      }}
                                    >
                                      <input
                                        type="checkbox"
                                        className="ct-picker-checkbox"
                                        checked={isAdded}
                                        onChange={() => {
                                          if (isAdded) {
                                            setContacts((prev) =>
                                              prev.filter((x) => x.id !== c.id),
                                            );
                                          } else {
                                            setContacts((prev) => {
                                              const map = new Map(
                                                prev.map((x) => [x.id, x]),
                                              );
                                              map.set(c.id, c);
                                              return Array.from(map.values());
                                            });
                                          }
                                        }}
                                        style={{
                                          width: 15,
                                          height: 15,
                                          accentColor: "#1d4ed8",
                                          appearance: "none",
                                          WebkitAppearance: "none",
                                          background: isAdded
                                            ? "#1d4ed8"
                                            : "#CBD5E1",
                                          borderRadius: 3,
                                          border: "none",
                                          cursor: "pointer",
                                          position: "relative",
                                          flexShrink: 0,
                                        }}
                                      />
                                      <span
                                        style={{
                                          fontSize: 13,
                                          color: "#0f172a",
                                          whiteSpace: "nowrap",
                                        }}
                                      >
                                        {c.name}
                                      </span>
                                    </label>
                                  );
                                })}
                              <p style={{ textAlign: "center", margin: "6px" }}>
                                Show all contacts
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {editingContact && (
                    <div className="ct-new-form">
                      <div className="ct-form-header">
                        <div className="ct-form-title">Edit Contact</div>
                        <button
                          className="ct-form-close"
                          onClick={handleCloseEdit}
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
                        </div>
                        <div className="ct-field">
                          <label>Email</label>
                          <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                          />
                        </div>

                        <div className="ct-field">
                          <label>Decision Maker</label>
                          <div className="ct-toggle-row">
                            <label className="ct-toggle">
                              <input
                                type="checkbox"
                                checked={isDecisionMaker}
                                onChange={(e) =>
                                  setIsDecisionMaker(e.target.checked)
                                }
                              />
                              <span className="ct-toggle-slider"></span>
                            </label>
                          </div>
                        </div>
                      </div>
                      <div className="ct-bottom-bar">
                        <button
                          className="ct-btn-cancel"
                          onClick={handleCloseEdit}
                        >
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
                          onClick={handleEditContact}
                          // disabled={isUpdateDisabled}
                        >
                          {loading ? "Updating..." : "Update"}
                        </button>
                      </div>
                    </div>
                  )}
                  {!editingContact && (
                    <div className="ct-grid">
                      {contacts.map((c) => (
                        <div
                          className={`ct-card${
                            toBool(
                              c?.isDecisionMaker ??
                                c?.isdecisionmaker ??
                                c?.is_primary,
                            )
                              ? " ct-card-primary"
                              : ""
                          }${
                            editingContact?.id === c.id ? " ct-card-active" : ""
                          }`}
                          key={c.id}
                          style={{ gap: 0 }}
                        >
                          {/* Header: avatar circle + name + edit */}
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 10,
                              padding: "0 0 10px",
                              borderBottom: "1px solid #e2e8f0",
                            }}
                          >
                            <div
                              style={{
                                width: 36,
                                height: 36,
                                borderRadius: "50%",
                                background: "#E2E8F0",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontSize: 15,
                                fontWeight: 700,
                                flexShrink: 0,
                              }}
                            >
                              {c.name?.charAt(0)?.toUpperCase() || "?"}
                            </div>
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div style={{ fontSize: 14, color: "#0f172a" }}>
                                <span style={{ fontWeight: 700 }}>
                                  {c.name}
                                </span>
                                {c.role ? ` | ${c.role}` : ""}
                              </div>
                            </div>
                            {toBool(
                              c?.isDecisionMaker ??
                                c?.isdecisionmaker ??
                                c?.is_primary,
                            ) && (
                              <span
                                className="ct-badge"
                                style={{ marginRight: 4 }}
                              >
                                <i className="ri-star-fill"></i>
                                Decision Maker
                              </span>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleCardClick(c);
                              }}
                              style={{
                                border: "none",
                                background: "transparent",
                                cursor: "pointer",
                                display: "inline-flex",
                                alignItems: "center",
                                justifyContent: "center",
                                color: "#475569",
                                padding: 0,
                                width: "auto",
                                height: "auto",
                                flexShrink: 0,
                              }}
                              title="Edit"
                            >
                              <i
                                className="ri-pencil-line"
                                style={{ fontSize: 14 }}
                              ></i>
                            </button>
                          </div>

                          {/* Email row */}
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 10,
                              padding: "10px 0",
                            }}
                          >
                            <div
                              style={{
                                width: 30,
                                height: 30,
                                borderRadius: "50%",
                                background: "#F1F5F9",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                              }}
                            >
                              <i
                                className="ri-mail-line"
                                style={{ fontSize: 14, color: "#475569" }}
                              ></i>
                            </div>
                            <div
                              style={{
                                flex: 1,
                                display: "flex",
                                flexDirection: "column",
                                minWidth: 0,
                              }}
                            >
                              <span
                                style={{
                                  fontSize: 11,
                                  color: "#64748b",
                                  fontWeight: 500,
                                }}
                              >
                                Email
                              </span>
                              <span
                                style={{
                                  fontSize: 13,
                                  color: "#0f172a",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {c.email || "-"}
                              </span>
                            </div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                if (c.email)
                                  window.location.href = `mailto:${c.email}`;
                              }}
                              style={{
                                background: "transparent",
                                border: "none",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: 4,
                                fontSize: 13,
                                fontWeight: 600,
                                color: "#0f172a",
                                padding: 0,
                              }}
                            >
                              Compose{" "}
                              <i
                                className="ri-arrow-right-s-line"
                                style={{ fontSize: 14 }}
                              ></i>
                            </button>
                          </div>

                          {/* Phone row */}
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 10,
                              padding: "10px 0",
                            }}
                          >
                            <div
                              style={{
                                width: 30,
                                height: 30,
                                borderRadius: "50%",
                                background: "#F1F5F9",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                              }}
                            >
                              <i
                                className="ri-phone-line"
                                style={{ fontSize: 14, color: "#475569" }}
                              ></i>
                            </div>
                            <div
                              style={{
                                flex: 1,
                                display: "flex",
                                flexDirection: "column",
                                minWidth: 0,
                              }}
                            >
                              <span
                                style={{
                                  fontSize: 11,
                                  color: "#64748b",
                                  fontWeight: 500,
                                }}
                              >
                                Phone
                              </span>
                              <span style={{ fontSize: 13, color: "#0f172a" }}>
                                {c.phone || "-"}
                              </span>
                            </div>
                            <button
                              // onClick={(e) => {
                              //   e.stopPropagation();
                              //   if (c.phone) window.location.href = `tel:${c.phone}`;
                              // }}
                              style={{
                                background: "transparent",
                                border: "none",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: 4,
                                fontSize: 13,
                                fontWeight: 600,
                                color: "#0f172a",
                                padding: 0,
                              }}
                            >
                              Call{" "}
                              <i
                                className="ri-arrow-right-s-line"
                                style={{ fontSize: 14 }}
                              ></i>
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : activeTopTab === "notes" ? (
                <div className="dv-notes-wrap">
                  <div className="dv-notes-header">
                    <h2 className="card-title">File & Notes</h2>
                  </div>

                  <div className="dv-note-editor">
                    <textarea
                      className="dv-note-textarea"
                      placeholder="Save a note"
                      maxLength={500}
                      value={noteText}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val.length > 500) {
                          setNoteError("Maximum 500 characters allowed");
                        } else {
                          setNoteError("");
                        }
                        setNoteText(val.slice(0, 500));
                      }}
                    ></textarea>
                    {noteError && (
                      <div
                        style={{
                          color: "#B91C1C",
                          fontSize: 12,
                          marginBottom: 8,
                        }}
                      >
                        {noteError}
                      </div>
                    )}
                    <div
                      className="dv-note-actions"
                      style={{ display: "flex", gap: 12, alignItems: "center" }}
                    >
                      <button
                        className="ct-btn-cancel"
                        onClick={() => {
                          setNoteText("");
                          setNoteError("");
                        }}
                      >
                        Clear
                      </button>
                      <button
                        className="sd-action-btn dv-note-draft"
                        disabled={loading || !noteText.trim() || noteError}
                        onClick={async () => {
                          if (noteText.trim().length > 500) {
                            setNoteError("Maximum 500 characters allowed");
                            return;
                          }
                          try {
                            setLoading(true);
                            const response = await createOpportunityNote(
                              id,
                              noteText.trim(),
                            );
                            if (response) {
                              setMessage("Note saved successfully");
                              setShowSuccessModal(true);
                              setNoteText("");
                              setNoteError("");
                            } else {
                              setMessage("Failed to save note");
                              setShowSuccessModal(true);
                            }
                          } catch (error) {
                            console.error("Note save error:", error);
                            setMessage("Failed to save note");
                            setShowSuccessModal(true);
                          } finally {
                            setLoading(false);
                          }
                        }}
                      >
                        {loading ? "Saving..." : "Draft"}
                      </button>
                      <span
                        style={{
                          fontSize: 11,
                          color: "#6b7280",
                          marginLeft: "auto",
                        }}
                      >
                        {noteText.length}/500
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  {/* Deal Summary */}
                  <div className="card">
                    <div className="card-title">Deal Summary</div>
                    <div className="deal-inner">
                      <div className="deal-content">
                        <img
                          src={accounticon}
                          alt="Account"
                          className="deal-image"
                        />

                        <div className="deal-desc">
                          Ford is Lenovo's largest automotive enterprise account
                          in North America with 177,000+ employees. Key decision
                          maker Brian Novak (VP IT Infrastructure) has completed
                          three prior Lenovo purchases totaling $12M. The
                          account has 3 open opportunities and an annual revenue
                          potential of $25M. Strong executive relationship —
                          Brian met our CTO at Lenovo Accelerate conference last
                          quarter. Manufacturing division urgently needs data
                          center expansion for new production line launching
                          August 1.
                        </div>
                      </div>
                    </div>
                    <div className="deal-meta text-center">
                      <p>
                        AI disclaimer information goes here with links and
                        resources available.
                      </p>
                    </div>
                  </div>

                  {/* Recommended Actions */}
                  <div className="card">
                    <div className="actions-header">
                      <img
                        src={accounticon}
                        alt="Account"
                        className="deal-image"
                      />
                      <div className="card-title" style={{ marginBottom: 0 }}>
                        Recommended Actions
                      </div>
                      <div className="sd-priority-meta">
                        4 total actions ·{" "}
                        <span className="sd-priority-due-today">
                          2 due today
                        </span>
                      </div>
                    </div>

                    <div className="deal-inner mb-4">
                      <div className="actions-headers">
                        <div className="card-title" style={{ marginBottom: 0 }}>
                          Send Reminder
                        </div>
                        <div className="sd-priority-meta">
                          <span className="sd-priority-due-today">
                            Due today
                          </span>
                        </div>
                      </div>
                      <div className="ad-action-info">
                        <div className="ad-action-title">
                         {opportunityDetails?.name}{""}
                          <span className="ad-action-amount"> {formatCurrencyShort(opportunityDetails?.value || 0)}</span>
                        </div>
                      </div>
                      <div className="deal-content">
                        <div className="deal-desc">
                          The pricing proposal sent to procurement on Jun 18 has
                          not received any response. No email opens detected and
                          the budget approval deadline is approaching in 5 days.
                          A follow-up call is recommended to confirm receipt and
                          address any concerns before the window closes.
                        </div>
                      </div>
                      <div className="ad-action-buttons">
                        <button className="ad-btn-secondary">
                          Complete Task
                        </button>
                        <button className="ad-btn-gradient">
                          <img
                            src={accounticon}
                            alt="Account"
                            className="deal-image-button"
                          />
                          Draft Email
                        </button>
                      </div>
                    </div>
                    <div className="deal-inner">
                      <div className="actions-headers">
                        <div className="card-title" style={{ marginBottom: 0 }}>
                          Confirmation Needed
                        </div>
                        <div className="sd-priority-meta">
                          <span className="sd-priority-due-today">
                            Due today
                          </span>
                        </div>
                      </div>
                      <div className="ad-action-info">
                        <div className="ad-action-title">
                          {opportunityDetails?.name}{" "}
                          <span className="ad-action-amount"> {formatCurrencyShort(opportunityDetails?.value || 0)}</span>
                        </div>
                      </div>
                      <div className="deal-content">
                        <div className="deal-desc">
                          The technical evaluation was completed last week but
                          no formal sign-off has been received from the IT
                          director. Competitor Dell has a demo scheduled for next
                          Tuesday. Securing written confirmation of preferred
                          vendor status before then is critical to maintaining
                          deal momentum.
                        </div>
                      </div>
                      <div className="ad-action-buttons">
                        <button className="ad-btn-secondary">
                          Complete Task
                        </button>
                        <button className="ad-btn-gradient">
                          <img
                            src={accounticon}
                            alt="Account"
                            className="deal-image-button"
                          />
                          Draft Email
                        </button>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>

            {activeTopTab !== "activity" &&
              activeTopTab !== "notes" &&
              activeTopTab !== "complete" &&
              activeTopTab !== "contacts" && (
                <div className="col-side">
                  <div className="card account-card">
                    <div className="account-header">
                      <img
                        src={buildingicon}
                        alt="Account"
                        className="account-image"
                      />
                      <div>
                        <div className="account-meta">
                          {" "}
                          {opportunityDetails?.account?.name || "-"}
                        </div>
                        <div className="account-meta-line">
                          {[
                            opportunityDetails?.account?.industry,
                            opportunityDetails?.account?.territory,
                            opportunityDetails?.account?.employeeCount
                              ? `${opportunityDetails.account.employeeCount >= 1000 ? `${Math.round(opportunityDetails.account.employeeCount / 1000)}K` : opportunityDetails.account.employeeCount}`
                              : null,
                          ]
                            .filter(Boolean)
                            .join("   ") || "  "}
                        </div>
                      </div>
                    </div>
                    <div className="account-grid">
                      <div className="">
                        <div className="field-label">Revenue Potential</div>
                        <div className="field-value">
                          {formatCurrencyShort(
                            opportunityDetails?.value || "-",
                          )}{" "}
                          year
                        </div>
                      </div>
                      <div className="">
                        <div className="field-label">Open Opps</div>
                        <div className="field-value">
                          {opportunityDetails?.account?.openDealsCount}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="card">
                    <div
                      className="account-header"
                      style={{ fontSize: 15, fontWeight: 700 }}
                    >
                      Risks
                    </div>
                    <div className="health-body">
                      {opportunityDetails?.risks?.length > 0 ? (
                        opportunityDetails.risks.map((risk, index) => (
                          <div key={index} className="health-row">
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                              }}
                            >
                              <span style={{ fontWeight: 700 }}>
                                {risk.name || "-"}
                              </span>
                              <span style={{ fontSize: 11, color: "#888" }}>
                                {risk.message || "-"}
                              </span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="health-row">
                          <span>No risks found</span>
                          <span className="health-pct">-</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="card account-card">
                    <div className="account-header">
                      <div>
                        <div style={{ fontSize: 15, fontWeight: 700 }}>
                          Deal Health Analysis
                        </div>
                        <div
                          style={{ fontSize: 11, color: "#888", marginTop: 4 }}
                        >
                          Last Activity:{" "}
                          {opportunityDetails?.health?.updatedAt
                            ? formatDateMMDDYY(
                                opportunityDetails.health.updatedAt,
                              )
                            : "-"}
                        </div>
                      </div>
                    </div>

                    {/* Deal Health Inner Card */}
                    <div
                      style={{
                        marginTop: 16,
                        border: "1px solid #e8eaed",
                        borderRadius: 8,
                        padding: 16,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: 12,
                        }}
                      >
                        <span style={{ fontSize: 14, fontWeight: 700 }}>
                          Deal Health
                        </span>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                          }}
                        >
                          <span
                            style={{
                              padding: "2px 10px",
                              borderRadius: 999,
                              fontSize: 11,
                              fontWeight: 600,
                              background:
                                opportunityDetails?.health?.band === "RED"
                                  ? "#FEE2E2"
                                  : opportunityDetails?.health?.band === "GREEN"
                                    ? "#DCFCE7"
                                    : opportunityDetails?.health?.band ===
                                        "YELLOW"
                                      ? "#FEF9C3"
                                      : "#F3F4F6",
                              color:
                                opportunityDetails?.health?.band === "RED"
                                  ? "#B91C1C"
                                  : opportunityDetails?.health?.band === "GREEN"
                                    ? "#15803D"
                                    : opportunityDetails?.health?.band ===
                                        "YELLOW"
                                      ? "#A16207"
                                      : "#555",
                            }}
                          >
                            {opportunityDetails?.health?.band === "RED"
                              ? "Critical"
                              : opportunityDetails?.health?.band === "YELLOW"
                                ? "At Risk"
                                : opportunityDetails?.health?.band === "GREEN"
                                  ? "Healthy"
                                  : "-"}
                          </span>
                          <span
                            style={{
                              fontSize: 14,
                              fontWeight: 700,
                              color:
                                opportunityDetails?.health?.band === "RED"
                                  ? "#B91C1C"
                                  : opportunityDetails?.health?.band === "GREEN"
                                    ? "#15803D"
                                    : opportunityDetails?.health?.band ===
                                        "YELLOW"
                                      ? "#A16207"
                                      : "#555",
                            }}
                          >
                            {opportunityDetails?.health?.score || 0}%
                          </span>
                        </div>
                      </div>

                      {opportunityDetails?.health?.components &&
                        (() => {
                          const components =
                            opportunityDetails.health.components;
                          return Object.entries(components).map(
                            ([key, comp]) => {
                              const weightedValue =
                                ((comp.weight || 0) / 100) * (comp.score || 0);
                              const progressPercent =
                                comp.weight > 0
                                  ? (weightedValue / comp.weight) * 100
                                  : 0;
                              const clampedPercent = Math.min(
                                Math.max(progressPercent, 0),
                                100,
                              );
                              const barColor =
                                clampedPercent >= 75
                                  ? "#34a853"
                                  : clampedPercent >= 50
                                    ? "#F59E0B"
                                    : "#EF4444";
                              return (
                                <div key={key} style={{ marginBottom: 10 }}>
                                  <div
                                    style={{
                                      display: "flex",
                                      justifyContent: "space-between",
                                      alignItems: "center",
                                      marginBottom: 4,
                                    }}
                                  >
                                    <span
                                      style={{
                                        fontSize: 12,
                                        color: "#555",
                                        textTransform: "capitalize",
                                      }}
                                    >
                                      {key.replace(/_/g, " ")}
                                    </span>
                                    <span
                                      style={{ fontSize: 12, fontWeight: 600 }}
                                    >
                                      {clampedPercent.toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="sd-progress-track">
                                    <div
                                      style={{
                                        width: `${clampedPercent}%`,
                                        height: "100%",
                                        background: barColor,
                                        borderRadius: 99,
                                      }}
                                    ></div>
                                  </div>
                                </div>
                              );
                            },
                          );
                        })()}
                    </div>
                  </div>
                  <div className="card">
                    <div
                      className="account-header"
                      style={{ fontSize: 15, fontWeight: 700 }}
                    >
                      Competitorsdata
                    </div>
                    <div className="health-body">
                      {opportunityDetails?.competitors?.length > 0 ? (
                        opportunityDetails.competitors.map((comp, index) => (
                          <div key={index} className="health-row">
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                              }}
                            >
                              <span style={{ fontWeight: 600 }}>
                                {comp.name || "-"}
                              </span>
                              <span style={{ fontSize: 11, color: "#888" }}>
                                {comp.competitorName || "-"}
                              </span>
                            </div>
                            <span
                              className="dv-badge dv-badge-warning"
                              style={{
                                borderRadius: "999px",
                                background: "#FFF7ED",
                                color: "#C2410C",
                              }}
                            >
                              {comp.competitorType || "-"}
                            </span>
                          </div>
                        ))
                      ) : (
                        <div className="health-row">
                          <span>No competitors found</span>
                          <span className="health-pct">-</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="card">
                    <div className="health-header">
                      <div className="card-title" style={{ marginBottom: 0 }}>
                        Deal Summary
                      </div>
                      {opportunityDetails?.priority && (
                        <span className="dv-badge dv-badge-warning">
                          {opportunityDetails.priority}
                        </span>
                      )}
                    </div>

                    <div className="health-row">
                      <span>Deal Value</span>
                      <span className="health-pct">
                        {formatCurrencyShort(opportunityDetails?.value || 0)}
                      </span>
                    </div>

                    <div className="health-row">
                      <span>Close Date</span>
                      <span className="health-pct">
                        {opportunityDetails?.closeDate
                          ? formatDateMMDDYY(opportunityDetails.closeDate)
                          : "-"}
                      </span>
                    </div>

                    <div className="health-row">
                      <span>Forecast</span>
                      <span className="health-pct">
                        {opportunityDetails?.forecastCategory || "-"}
                      </span>
                    </div>

                    <div className="health-row">
                      <span>Owner</span>
                      <span className="health-pct">
                        {opportunityDetails?.ownerName || "-"}
                      </span>
                    </div>

                    <div className="health-row">
                      <span>Region</span>
                      <span className="health-pct">
                        {opportunityDetails?.account?.territory || "-"}
                      </span>
                    </div>

                    <div className="health-row">
                      <span>Lead Origin</span>
                      <span className="health-pct">
                        {opportunityDetails?.leadOrigin || "-"}
                      </span>
                    </div>

                    <div className="health-row">
                      <span>Partner</span>
                      <span className="health-pct">
                        {opportunityDetails?.partnerInvolved ? "Yes" : "No"}
                      </span>
                    </div>
                  </div>
                </div>
              )}
          </div>
        </div>
        <Modal
          show={showOpportunityDeleteModal}
          onHide={() => {
            setShowOpportunityDeleteModal(false);
            setConfirmOpportunityDelete(false);
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
              Delete Opportunity
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <p className="delete-text">
              Are you sure you want to delete this opportunity?
            </p>

            <div className="form-check mt-3">
              <input
                className="form-check-input"
                type="checkbox"
                id="confirmOpportunityDelete"
                checked={confirmOpportunityDelete}
                onChange={(e) => setConfirmOpportunityDelete(e.target.checked)}
              />

              <label
                className="form-check-label bold-text"
                htmlFor="confirmOpportunityDelete"
              >
                I want to delete the opportunity.
              </label>
            </div>
          </Modal.Body>

          <Modal.Footer style={{ border: "none" }}>
            <button
              className="ct-btn-save"
              onClick={() => {
                setShowOpportunityDeleteModal(false);
                setConfirmOpportunityDelete(false);
              }}
            >
              Cancel
            </button>

            <button
              className="ct-btn-save"
              onClick={handleDeleteOpportunity}
              disabled={!confirmOpportunityDelete || loading}
              style={{
                opacity: !confirmOpportunityDelete || loading ? 0.5 : 1,
                cursor:
                  !confirmOpportunityDelete || loading
                    ? "not-allowed"
                    : "pointer",
              }}
            >
              {loading ? "Deleting..." : "Delete"}
            </button>
          </Modal.Footer>
        </Modal>
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
