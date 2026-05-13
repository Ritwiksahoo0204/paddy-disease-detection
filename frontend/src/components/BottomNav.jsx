import { useNavigate, useLocation } from "react-router-dom";

const NAV_ITEMS = {
  en: [
    { icon: "🏠", label: "Home",      path: "/" },
    { icon: "🕐", label: "History",   path: "/history" },
    { icon: "📊", label: "Dashboard", path: "/dashboard" },
  ],
  hi: [
    { icon: "🏠", label: "होम",       path: "/" },
    { icon: "🕐", label: "इतिहास",    path: "/history" },
    { icon: "📊", label: "डैशबोर्ड",  path: "/dashboard" },
  ],
  bn: [
    { icon: "🏠", label: "হোম",       path: "/" },
    { icon: "🕐", label: "ইতিহাস",    path: "/history" },
    { icon: "📊", label: "ড্যাশবোর্ড", path: "/dashboard" },
  ],
};

export default function BottomNav({ language = "en" }) {
  const navigate = useNavigate();
  const location = useLocation();
  const items    = NAV_ITEMS[language] || NAV_ITEMS.en;

  const hiddenPaths = ["/camera", "/analyzing"];
  if (hiddenPaths.includes(location.pathname)) return null;

  return (
    <nav className="bottom-nav">
      {items.map((item) => (
        <button
          key={item.path}
          className={`nav-item ${location.pathname === item.path ? "active" : ""}`}
          onClick={() => navigate(item.path)}
        >
          <span className="nav-icon">{item.icon}</span>
          <span className="nav-label">{item.label}</span>
        </button>
      ))}
    </nav>
  );
}