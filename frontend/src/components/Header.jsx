import { useNavigate } from "react-router-dom";

export default function Header({ title, language, onAction, actionIcon, showBack = true }) {
  const navigate = useNavigate();

  return (
    <div className="header">
      {showBack ? (
        <button className="header-back" onClick={() => navigate(-1)}>
          ←
        </button>
      ) : (
        <div style={{ width: 36 }} />
      )}

      <span className="header-title">{title}</span>

      {onAction ? (
        <button className="header-action" onClick={onAction}>
          {actionIcon || "⋯"}
        </button>
      ) : (
        <div style={{ width: 36 }} />
      )}
    </div>
  );
}