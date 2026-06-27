import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { fetchCustomerDetails } from "../../../api/client";
export default function CustomerInformation() {
  const { id } = useParams();
  const [accountDetails, setAccountDetails] = useState({});
  useEffect(() => {
    fetchCustomerDetails(id).then((response) => {
      setAccountDetails(response);
    });
  }, [id]);
  //   const basic = accountDetails?.basicInformation || {};
  // const billing = accountDetails?.billingAddress || {};
  // const shipping = accountDetails?.shippingAddress || {};
  // const legal = accountDetails?.identityAndLegal || {};
  // const commercial = accountDetails?.commercialTerms || {};
  // const territory = accountDetails?.territoryAndOwnership || {};
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
        .ci-content { display: flex; flex-direction: column; gap: 16px; }
        .ci-card { background: #fff; border: 1px solid var(--opd-border); border-radius: 10px; padding: 20px 24px; display: flex; flex-direction: column; gap: 20px; }
        .ci-card-title { font-size: 16px; font-weight: 700; color: #0f172a; line-height: 1.4; }
        .ci-fields { display: flex; flex-wrap: wrap; gap: 16px; }
        .ci-field { display: flex; flex-direction: column; gap: 4px; flex: 0 0 calc(25% - 12px); min-width: 160px; }
        .ci-field.half { flex: 0 0 calc(50% - 8px); }
        .ci-field.full { flex: 0 0 100%; }
        .ci-field label { font-weight: 400; font-size: 13px; color: #0f172a; line-height: 20px; }
        .ci-field input, .ci-field select {
          background: transparent; border: none; border-radius: 4px;
          height: 38px; padding: 0 0px; font-size: 16px; font-weight: 700; color: #334155;
          width: 100%; outline: none;
        }
        .ci-field input:focus, .ci-field select:focus { border: none; }
        .ci-field input::placeholder { color: #94a3b8; }
        .ci-card-territory .ci-field input,
        .ci-card-territory .ci-field select {
          background: #f8fafc;
          border: 2px solid #cbd5e1;
          border-radius: 4px;
          padding:0px 10px;
        }
        .ci-card-territory .ci-field input:focus,
        .ci-card-territory .ci-field select:focus {
          border: 2px solid #94a3b8;
        }
        .select-wrap { position: relative; }
        .select-wrap select { padding-right: 28px; appearance: none; cursor: pointer; }
        .select-wrap::after { content: ''; position: absolute; right: 10px; top: 50%; transform: translateY(-50%); border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #64748b; pointer-events: none; }
      `}</style>

      <div className="ci-content mb-4">
        {/* Aliases for clean mapping */}
        {(() => {
          const basic = accountDetails?.basicInformation ?? {};
          const billing = accountDetails?.billingAddress ?? {};
          const shipping = accountDetails?.shippingAddress ?? {};
          const legal = accountDetails?.identityAndLegal ?? {};
          const commercial = accountDetails?.commercialTerms ?? {};
          const territory = accountDetails?.territoryAndOwnership ?? {};

          return (
            <>
              {/* Basic Information */}
              <div className="ci-card">
                <div className="ci-card-title">Basic Information</div>

                <div className="ci-fields">
                  <div className="ci-field">
                    <label>Account ID</label>
                    <input type="text" value={basic.accountId || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>Account Name</label>
                    <input
                      type="text"
                      value={basic.accountName || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label> Type</label>
                    <input
                      type="text"
                      value={basic.accountType || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Segment</label>
                    <input type="text" value={basic.segment || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>Sub-Segment</label>
                    <input
                      type="text"
                      value={basic.subSegment || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Industry Segment</label>
                    <input
                      type="text"
                      value={basic.industrySegment || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>GTM Segment</label>
                    <input
                      type="text"
                      value={basic.gtmSegment || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Annual Revenue</label>
                    <input
                      type="text"
                      value={basic.annualRevenue || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Employee Count</label>
                    <input
                      type="text"
                      value={basic.employeeCount || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Sales Known As</label>
                    <input
                      type="text"
                      value={basic.sellerKnownAs || ""}
                      readOnly
                    />
                  </div>
                </div>
              </div>

              {/* Billing Address */}
              <div className="ci-card">
                <div className="ci-card-title">Billing Address</div>

                <div className="ci-fields">
                  <div className="ci-field half">
                    <label>Address Line 1</label>
                    <input type="text" value={billing.line1 || ""} readOnly />
                  </div>

                  <div className="ci-field half">
                    <label>Address Line 2</label>
                    <input type="text" value={billing.line2 || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>City</label>
                    <input type="text" value={billing.city || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>State/Province</label>
                    <input
                      type="text"
                      value={billing.stateProvince || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Postal Code</label>
                    <input
                      type="text"
                      value={billing.postalCode || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Country</label>
                    <input type="text" value={billing.country || ""} readOnly />
                  </div>
                </div>
              </div>

              {/* Shipping Address */}
              <div className="ci-card">
                <div className="ci-card-title">Shipping Address</div>

                <div className="ci-fields">
                  <div className="ci-field half">
                    <label>Address Line 1</label>
                    <input type="text" value={shipping.line1 || ""} readOnly />
                  </div>

                  <div className="ci-field half">
                    <label>Address Line 2</label>
                    <input type="text" value={shipping.line2 || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>City</label>
                    <input type="text" value={shipping.city || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>State/Province</label>
                    <input
                      type="text"
                      value={shipping.stateProvince || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Postal Code</label>
                    <input
                      type="text"
                      value={shipping.postalCode || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Country</label>
                    <input
                      type="text"
                      value={shipping.country || ""}
                      readOnly
                    />
                  </div>
                </div>
              </div>

              <div className="ci-card">
                <div className="ci-card-title">Hierarchy & Relationships</div>
                <div className="ci-fields">
                  <div className="ci-field">
                    <label>Business Group</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>Role</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>Organization Group</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>Parent Account</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>Billing Account</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>Child Account</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>Account Hierarchy Level</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                  <div className="ci-field">
                    <label>MBO</label>
                    <input type="text" placeholder="-" readOnly />
                  </div>
                </div>
              </div>

              {/* Identity & Legal */}
              <div className="ci-card">
                <div className="ci-card-title">Identity & Legal</div>

                <div className="ci-fields">
                  <div className="ci-field">
                    <label>Legal Name in Local Language</label>
                    <input
                      type="text"
                      value={legal.legalNameLocal || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Local Language</label>
                    <input
                      type="text"
                      value={legal.localLanguage || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Alias</label>
                    <input type="text" value={legal.alias || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>Tax ID / VAT Number</label>
                    <input
                      type="text"
                      value={legal.taxVatNumber || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Legal Entity</label>
                    <input
                      type="text"
                      value={legal.legalEntity || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Main Phone</label>
                    <input type="text" value={legal.mainPhone || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>Website</label>
                    <input type="text" value={legal.website || ""} readOnly />
                  </div>

                  <div className="ci-field">
                    <label>LinkedIn URL</label>
                    <input
                      type="text"
                      value={legal.linkedinUrl || ""}
                      readOnly
                    />
                  </div>
                </div>
              </div>

              {/* Account Classification */}
              <div className="ci-card">
          <div className="ci-card-title">Account Classification</div>
          <div className="ci-fields">
            <div className="ci-field">
              <label>Relationship Strength</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Account Notes / Descriptions</label>
              <input type="text" placeholder="-" readOnly/>
            </div>
            <div className="ci-field">
              <label>Tags / Labels</label>
              <input type="text" placeholder="-" readOnly/>
            </div>
            <div className="ci-field">
              <label>Named Account Flag</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Strategic Account Flag</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>
        </div>


              {/* Commercial Terms */}
              <div className="ci-card">
                <div className="ci-card-title">Commercial Terms</div>

                <div className="ci-fields">
                  <div className="ci-field">
                    <label>Default Currency</label>
                    <input
                      type="text"
                      value={commercial.defaultCurrency || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Payment Terms</label>
                    <input
                      type="text"
                      value={commercial.paymentTerms || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>Price List</label>
                    <input
                      type="text"
                      value={commercial.priceList || ""}
                      readOnly
                    />
                  </div>

                  <div className="ci-field">
                    <label>DocuSign Configuration</label>
                    <input
                      type="text"
                      value={commercial.dealSignConfig || ""}
                      readOnly
                    />
                  </div>
                </div>
              </div>

              {/* Insights & KPIs */}
            <div className="ci-card">
          <div className="ci-card-title">Insights & KPIs</div>
          <div className="ci-fields">
            <div className="ci-field">
              <label>Total Account Value</label>
               <input
                type="text"
                value={accountDetails?.totalAccountValue || ""}
                readOnly
              />
            </div>
            <div className="ci-field">
              <label>Last Interaction Date</label>
            <input
                type="text"
                value={accountDetails?.lastInteraction || ""}
                readOnly
              />
            </div>
            <div className="ci-field">
              <label>Account KPI Snapshot</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Daily KPI Trend</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>
        </div>

              {/* Partners & Channel */}
               <div className="ci-card">
          <div className="ci-card-title">Partners & Channel</div>
          <div className="ci-fields">
            <div className="ci-field">
              <label>Preferred Channel Partner</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Distributor Account</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Alliance Partner</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Channel Categorization</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>
        </div>

              {/* Planning & Growth */}
            <div className="ci-card">
          <div className="ci-card-title">Planning & Growth</div>
          <div className="ci-fields">
            <div className="ci-field">
              <label>Account Plan</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Account Growth Plan</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Strategic Objectives</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Key Growth Actions</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field half">
              <label>Resource Plan</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>
        </div>


              {/* Systems & Integration */}
              <div className="ci-card">
                <div className="ci-card-title">Systems & Integration</div>

                <div className="ci-fields">
                  <div className="ci-field">
                    <label>MDM ID</label>
                    <input type="text" value="" readOnly />
                  </div>

                  <div className="ci-field">
                    <label>Source ID</label>
                    <input type="text" value="" readOnly />
                  </div>

                  <div className="ci-field">
                    <label>DNB ID</label>
                    <input type="text" value="" readOnly />
                  </div>
                </div>
              </div>

              {/* Opportunities & Orders */}
             <div className="ci-card">
          <div className="ci-card-title">Opportunities & Orders</div>
          <div className="ci-fields">
            <div className="ci-field">
              <label>Open Opportunities</label>
            <input
                type="text"
                value={accountDetails?.activeOpportunitiesCount || ""}
                readOnly
              />
            </div>
            <div className="ci-field">
              <label>Closed Opportunities</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Quotes</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Orders</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Deal Registrations</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Deal Protections</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Billing Milestones</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Funding Requests</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Concession Requests</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>
        </div>

              {/* Service & Support */}
           <div className="ci-card">
          <div className="ci-card-title">Service & Support</div>
          <div className="ci-fields">
            <div className="ci-field">
              <label>Trip Reports</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Account Services</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Return Requests</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Helpdesk Requests</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>
        </div>


              {/* Territory & Ownership */}
               <div className="ci-card ">

                <div className="ci-card-title">Territory & Ownership</div>
          <div className="ci-fields ">
            <div className="ci-field">
              <label>Territory / Region</label>
              <input
                type="text"
                value={territory?.region || ""}
                readOnly
              />
            </div>
            
            <div className="ci-field">
              <label>Assigned Owner</label>
             <input
                type="text"
                value={territory?.assignedOwnerName || ""}
                readOnly
              />
            </div>
            <div className="ci-field">
              <label>Record Owner</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Territory</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Future Territory</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Org</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Territory Move Reason</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Geographic Unit</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Office</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>


          <div className="ci-card-title mt-4">Input</div>
          <div className="ci-fields ci-card-territory">
            <div className="ci-field">
              <label>Territory / Region</label>
              <input
                type="text"
                value={territory?.region || ""}
                readOnly
              />
            </div>
            <div className="ci-field">
              <label>Assigned Owner</label>
             <input
                type="text"
                value={territory?.assignedOwnerName || ""}
                readOnly
              />
            </div>
            <div className="ci-field">
              <label>Record Owner</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Territory</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Future Territory</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Org</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Territory Move Reason</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Geographic Unit</label>
              <input type="text" placeholder="-" readOnly />
            </div>
            <div className="ci-field">
              <label>Sales Office</label>
              <input type="text" placeholder="-" readOnly />
            </div>
          </div>


        </div>

            </>
          );
        })()}
      </div>
    </>
  );
}
