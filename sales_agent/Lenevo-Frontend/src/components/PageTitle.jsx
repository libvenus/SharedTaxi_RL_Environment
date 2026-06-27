function PageTitle({ title, subtitle }) {
  return (
    <div className="page-title">
      <h2 className="title-text">{title}</h2>
      {subtitle && <p className="subtitle-text text-muted">{subtitle}</p>}
    </div>
  );
}

export default PageTitle;