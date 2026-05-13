import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import BottomNav from "../components/BottomNav";

const LABELS = {
  en: {
    title:      "Scan History",
    empty:      "No scans yet",
    emptyDesc:  "Your previous scans will appear here",
    confidence: "Confidence",
    loading:    "Loading history...",
    error:      "Could not load history",
  },
  hi: {
    title:      "स्कैन इतिहास",
    empty:      "अभी तक कोई स्कैन नहीं",
    emptyDesc:  "आपके पिछले स्कैन यहाँ दिखेंगे",
    confidence: "विश्वास",
    loading:    "इतिहास लोड हो रहा है...",
    error:      "इतिहास लोड नहीं हो सका",
  },
  bn: {
    title:      "স্ক্যান ইতিহাস",
    empty:      "এখনো কোনো স্ক্যান নেই",
    emptyDesc:  "আপনার আগের স্ক্যানগুলো এখানে দেখাবে",
    confidence: "আস্থা",
    loading:    "ইতিহাস লোড হচ্ছে...",
    error:      "ইতিহাস লোড করা যায়নি",
  },
};

const DISEASE_ICONS = {
  Bacterialblight: "🦠",
  Blast:           "💥",
  Brownspot:       "🟤",
  Healthy:         "✅",
  Tungro:          "🐛",
};

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("en-IN", {
    day: "numeric", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit"
  });
}

function getSeverityClass(severity) {
  if (!severity) return "None";
  const s = severity.toLowerCase();
  if (s === "severe") return "Severe";
  if (s === "moderate") return "Moderate";
  return "Mild";
}

export default function History({ language }) {
  const navigate         = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(false);
  const L = LABELS[language] || LABELS.en;

  useEffect(() => {
    const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
    fetch(`${API}/history`)
      .then((r) => r.json())
      .then((data) => { setHistory(data.history || []); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, []);

  return (
    <div className="history-page animate-in">
      <Header title={L.title} language={language} showBack={false} />

      {loading && (
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <div className="loading-spinner" />
          <p style={{ color: "var(--gray-500)", fontSize: "0.85rem" }}>{L.loading}</p>
        </div>
      )}

      {error && (
        <div className="empty-history">
          <div className="empty-icon">⚠️</div>
          <div className="empty-text">{L.error}</div>
        </div>
      )}

      {!loading && !error && history.length === 0 && (
        <div className="empty-history">
          <div className="empty-icon">🌿</div>
          <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.5rem" }}>{L.empty}</div>
          <div className="empty-text">{L.emptyDesc}</div>
          <button
            className="btn-primary"
            style={{ marginTop: "1.5rem", width: "auto", padding: "0.75rem 2rem" }}
            onClick={() => navigate("/")}
          >
            Start Scanning
          </button>
        </div>
      )}

      {!loading && !error && history.length > 0 && (
        <div className="history-list">
          {history.map((item) => (
            <div className="history-card" key={item.id}>
              {/* Image */}
              <div className="history-img">
                {item.image_base64 ? (
                  <img
                    src={`data:image/jpeg;base64,${item.image_base64}`}
                    alt={item.disease}
                    style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: "var(--radius-xs)" }}
                  />
                ) : (
                  <span>{DISEASE_ICONS[item.disease] || "🌿"}</span>
                )}
              </div>

              {/* Info */}
              <div className="history-info">
                <div className="history-disease">{item.disease}</div>
                <div className="history-confidence">
                  {L.confidence}: {item.confidence?.toFixed(1)}%
                </div>
                <div className="history-date">{formatDate(item.timestamp)}</div>
              </div>

              {/* Badges */}
              <div className="history-badges">
                <span className={`severity-badge ${getSeverityClass(item.severity)}`}>
                  {getSeverityClass(item.severity)}
                </span>
                {item.was_correct && (
                  <span style={{
                    fontSize: "0.7rem",
                    color: item.was_correct === "yes" ? "var(--green-700)" : "var(--red)",
                    fontWeight: 600
                  }}>
                    {item.was_correct === "yes" ? "✅ Correct" : "❌ Wrong"}
                  </span>
                )}
                <span className="chevron">›</span>
              </div>
            </div>
          ))}
        </div>
      )}

      <BottomNav language={language} />
    </div>
  );
}