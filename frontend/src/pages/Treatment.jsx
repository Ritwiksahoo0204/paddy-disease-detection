import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import BottomNav from "../components/BottomNav";

const LABELS = {
  en: {
    title: "Treatment & Prevention",
    treatment: "Recommended Treatment",
    prevention: "Prevention Tips",
    expert: "When to Consult an Expert",
    expertText:
      "If the disease severity increases or spreads rapidly, consult your agriculture expert.",
    helpline: "📞 Kisan Call Centre: 1800-180-1551 (Toll Free, 24/7)",
    back: "Back to Result",
  },
  hi: {
    title: "उपचार और बचाव",
    treatment: "अनुशंसित उपचार",
    prevention: "बचाव के सुझाव",
    expert: "विशेषज्ञ से कब मिलें",
    expertText:
      "यदि रोग की गंभीरता बढ़े या तेजी से फैले, तो अपने कृषि विशेषज्ञ से संपर्क करें।",
    helpline: "📞 किसान कॉल सेंटर: 1800-180-1551 (निःशुल्क, 24/7)",
    back: "परिणाम पर वापस जाएं",
  },
  bn: {
    title: "চিকিৎসা ও প্রতিরোধ",
    treatment: "প্রস্তাবিত চিকিৎসা",
    prevention: "প্রতিরোধ টিপস",
    expert: "কখন বিশেষজ্ঞের সাথে পরামর্শ করবেন",
    expertText:
      "রোগের তীব্রতা বাড়লে বা দ্রুত ছড়িয়ে পড়লে আপনার কৃষি বিশেষজ্ঞের সাথে যোগাযোগ করুন।",
    helpline: "📞 কিসান কল সেন্টার: 1800-180-1551 (বিনামূল্যে, 24/7)",
    back: "ফলাফলে ফিরে যান",
  },
};

const DISEASE_ICONS = {
  Bacterialblight: "🦠",
  Blast: "💥",
  Brownspot: "🟤",
  Healthy: "✅",
  Tungro: "🐛",
};

/* ── Pesticide Bottle SVG (matches reference) ── */
function BottleIllustration() {
  return (
    <svg width="72" height="90" viewBox="0 0 72 90" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Cap */}
      <rect x="26" y="2" width="20" height="10" rx="3" fill="#1a472a"/>
      {/* Neck */}
      <rect x="29" y="12" width="14" height="8" rx="2" fill="#2d6a4f"/>
      {/* Body */}
      <rect x="14" y="20" width="44" height="62" rx="8" fill="#2d6a4f"/>
      {/* Label background */}
      <rect x="18" y="28" width="36" height="42" rx="5" fill="white" opacity="0.9"/>
      {/* Label leaf icon */}
      <text x="36" y="46" textAnchor="middle" fontSize="14">🌿</text>
      {/* Label lines */}
      <rect x="22" y="50" width="28" height="3" rx="1.5" fill="#2d6a4f" opacity="0.4"/>
      <rect x="24" y="56" width="24" height="2.5" rx="1.25" fill="#2d6a4f" opacity="0.3"/>
      <rect x="26" y="61" width="20" height="2.5" rx="1.25" fill="#2d6a4f" opacity="0.3"/>
      {/* Shine */}
      <rect x="54" y="24" width="3" height="20" rx="1.5" fill="white" opacity="0.2"/>
    </svg>
  );
}

/* ── Farmer Illustration SVG (matches reference) ── */
function FarmerIllustration() {
  return (
    <svg width="72" height="80" viewBox="0 0 72 80" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Hat brim */}
      <ellipse cx="36" cy="18" rx="22" ry="5" fill="#d97706"/>
      {/* Hat top */}
      <rect x="24" y="6" width="24" height="14" rx="4" fill="#d97706"/>
      {/* Hat band */}
      <rect x="24" y="16" width="24" height="3" fill="#92400e"/>
      {/* Head */}
      <circle cx="36" cy="28" r="10" fill="#fed7aa"/>
      {/* Eyes */}
      <circle cx="32" cy="27" r="1.5" fill="#374151"/>
      <circle cx="40" cy="27" r="1.5" fill="#374151"/>
      {/* Smile */}
      <path d="M32 32 Q36 35 40 32" stroke="#374151" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
      {/* Body */}
      <rect x="24" y="38" width="24" height="24" rx="6" fill="#2d6a4f"/>
      {/* Shirt collar */}
      <path d="M32 38 L36 44 L40 38" fill="white" opacity="0.6"/>
      {/* Left arm */}
      <rect x="12" y="40" width="12" height="7" rx="3.5" fill="#2d6a4f" transform="rotate(-20 12 40)"/>
      {/* Right arm — holding tool */}
      <rect x="48" y="40" width="12" height="7" rx="3.5" fill="#2d6a4f" transform="rotate(20 48 40)"/>
      {/* Tool / stick */}
      <rect x="57" y="36" width="3" height="28" rx="1.5" fill="#92400e"/>
      <ellipse cx="58.5" cy="35" rx="5" ry="3" fill="#52b788"/>
      {/* Legs */}
      <rect x="26" y="60" width="9" height="18" rx="4" fill="#1e3a5f"/>
      <rect x="37" y="60" width="9" height="18" rx="4" fill="#1e3a5f"/>
      {/* Boots */}
      <rect x="24" y="74" width="13" height="6" rx="3" fill="#111827"/>
      <rect x="35" y="74" width="13" height="6" rx="3" fill="#111827"/>
    </svg>
  );
}

/* ── Shield SVG for prevention ── */
function ShieldIllustration() {
  return (
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M28 4 L48 12 L48 30 C48 42 28 52 28 52 C28 52 8 42 8 30 L8 12 Z" fill="#d8f3dc"/>
      <path d="M28 8 L44 15 L44 30 C44 40 28 48 28 48 C28 48 12 40 12 30 L12 15 Z" fill="#52b788" opacity="0.5"/>
      <text x="28" y="33" textAnchor="middle" fontSize="18">🌿</text>
    </svg>
  );
}

export default function Treatment({ language, result }) {
  const navigate = useNavigate();
  const L = LABELS[language] || LABELS.en;

  if (!result) return null;

  return (
    <div className="treatment-page animate-in" style={{ paddingBottom: "90px" }}>
      <Header title={L.title} language={language} />

      {/* ── Disease Tag ── */}
      <div style={{ padding: "1rem 1.25rem 0" }}>
        <div style={{
          display: "flex", alignItems: "center", gap: "0.6rem",
          background: "var(--white)", borderRadius: "var(--radius-sm)",
          padding: "0.875rem 1rem", boxShadow: "var(--shadow-sm)"
        }}>
          <span style={{ fontSize: "1.5rem" }}>{DISEASE_ICONS[result.disease] || "🌿"}</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: "0.95rem" }}>{result.disease}</div>
            <div style={{ fontSize: "0.75rem", color: "var(--gray-500)" }}>
              {result.severity} severity — {result.risk_level} risk
            </div>
          </div>
          <div style={{
            marginLeft: "auto", padding: "0.3rem 0.75rem", borderRadius: "20px",
            fontSize: "0.75rem", fontWeight: 700,
            background: result.risk_level === "High" ? "#fef2f2"
                      : result.risk_level === "Moderate" ? "var(--orange-light)"
                      : "var(--green-100)",
            color: result.risk_level === "High" ? "var(--red)"
                 : result.risk_level === "Moderate" ? "var(--orange)"
                 : "var(--green-700)"
          }}>
            {result.risk_level} Risk
          </div>
        </div>
      </div>

      {/* ── Severity Note ── */}
      {result.severity_note && (
        <div style={{
          margin: "0.75rem 1.25rem 0", padding: "0.75rem 1rem",
          background: "var(--green-50)", borderLeft: "4px solid var(--green-700)",
          borderRadius: "0 var(--radius-xs) var(--radius-xs) 0",
          fontSize: "0.82rem", color: "var(--green-900)", fontWeight: 600
        }}>
          📊 {result.severity_note}
        </div>
      )}

      {/* ── Recommended Treatment — with bottle illustration ── */}
      <div className="treatment-section" style={{ position: "relative", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
          <div style={{ flex: 1, paddingRight: "0.5rem" }}>
            <div className="treatment-section-title" style={{ color: "var(--green-700)" }}>
              💊 {L.treatment}
            </div>
            <ul className="treatment-list">
              {result.treatment?.map((item, i) => (
                <li className="treatment-item" key={i}>
                  <div className="treatment-bullet">{i + 1}</div>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
          {/* Bottle illustration */}
          <div style={{ flexShrink: 0, opacity: 0.9, marginTop: "0.5rem" }}>
            <BottleIllustration />
          </div>
        </div>
      </div>

      {/* ── Prevention Tips — with shield illustration ── */}
      <div className="treatment-section" style={{ background: "#f0fdf4", position: "relative", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
          <div style={{ flex: 1, paddingRight: "0.5rem" }}>
            <div className="treatment-section-title" style={{ color: "var(--green-800)" }}>
              🛡️ {L.prevention}
            </div>
            <ul className="treatment-list">
              {result.prevention?.map((item, i) => (
                <li className="treatment-item" key={i}>
                  <div className="treatment-bullet" style={{ background: "var(--green-100)", color: "var(--green-700)" }}>✓</div>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
          {/* Shield illustration */}
          <div style={{ flexShrink: 0, marginTop: "0.5rem" }}>
            <ShieldIllustration />
          </div>
        </div>
      </div>

      {/* ── When to Consult — with farmer illustration ── */}
      <div style={{
        margin: "0 1.25rem 0.75rem",
        background: "var(--green-50)",
        border: "1px solid var(--green-100)",
        borderRadius: "var(--radius-sm)",
        padding: "1.25rem",
        display: "flex",
        alignItems: "center",
        gap: "1rem",
        overflow: "hidden"
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--green-900)", marginBottom: "0.5rem" }}>
            {L.expert}
          </div>
          <p style={{ fontSize: "0.82rem", color: "var(--green-700)", lineHeight: 1.6 }}>
            {L.expertText}
          </p>
        </div>
        {/* Farmer illustration */}
        <div style={{ flexShrink: 0 }}>
          <FarmerIllustration />
        </div>
      </div>

      {/* ── Helpline ── */}
      <div className="helpline-box" style={{ margin: "0 1.25rem 0.75rem" }}>
        {L.helpline}
      </div>

      {/* ── Disclaimer ── */}
      <div className="disclaimer-box" style={{ margin: "0 1.25rem 1rem" }}>
        <div className="disclaimer-title">⚠️ Important Disclaimer</div>
        <p className="disclaimer-text">
          {result.disclaimer?.[language] || result.disclaimer?.en ||
            "Always verify treatment with a certified agricultural officer before applying."}
        </p>
      </div>

      {/* ── Back Button ── */}
      <div style={{ padding: "0 1.25rem 1rem" }}>
        <button className="btn-secondary" style={{ width: "100%" }} onClick={() => navigate("/result")}>
          ← {L.back}
        </button>
      </div>

      <BottomNav language={language} />
    </div>
  );
}