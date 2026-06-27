export default function SharedLibrary() {
  const documents = [
    {
      id: 1,
      name: "Team Playbook - Enterprise",
      source: "SharePoint",
      updatedOn: "2026-06-20",
    },
    {
      id: 2,
      name: "Competitive Battle Card",
      source: "Google Drive",
      updatedOn: "2026-06-17",
    },
    {
      id: 3,
      name: "ROI Calculator Template",
      source: "OneDrive",
      updatedOn: "2026-06-14",
    },
    {
      id: 4,
      name: "Customer Success Stories",
      source: "SharePoint",
      updatedOn: "2026-06-11",
    },
  ];

  return (
    <div className="library-grid">
      {documents.map((doc) => (
        <div className="library-card" key={doc.id}>
          <div className="library-card-left">
            <div className="library-card-icon">
              <i className="ri-file-text-line"></i>
            </div>
            <div className="library-card-info">
              <div className="library-card-name">{doc.name}</div>
              <div className="library-card-meta">
                <span>Source: {doc.source}</span>
                <span>Updated on: {doc.updatedOn}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
