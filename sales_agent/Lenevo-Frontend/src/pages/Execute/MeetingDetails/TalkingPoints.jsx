function TalkingPoints() {
  const points = [
    {
      id: 1,
      description:
        "Open by acknowledging the budget expansion — 'We're excited to support Infosys's growing fleet needs and happy to right-size the proposal to 1,400 units.",
    },
    {
      id: 2,
      description:
        "Lead DaaS with the CFO angle: zero upfront CAPEX, predictable monthly opex, and built-in refresh at 36 months. Relevant since Finance opened the deck.",
    },
    {
      id: 3,
      description:
        "Differentiate on ProSupport+: 1-business-day onsite response, accidental damage protection, 99.9% uptime SLA — Dell doesn't match this at the same price tier.",
    },
    {
      id: 4,
      description:
        "Address the staggered delivery ask directly — propose phased delivery: 500 units in Jul, 500 in Aug, 400 in Sep, aligned to Infosys's campus rollout schedule.",
    },
    {
      id: 5,
      description:
        "Close with a clear timeline: revised proposal by June 17th ? legal review by June 19th ? approval meeting June 20th ? PO by June 22nd.",
    },
  ];

  return (
    <div className="tp-container">
      {points.map((point) => (
        <div className="tp-card" key={point.id}>
          <div className="tp-number">{point.id}</div>
          <div className="tp-desc">{point.description}</div>
        </div>
      ))}
    </div>
  );
}

export default TalkingPoints;
