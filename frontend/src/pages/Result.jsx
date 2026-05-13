import { useNavigate } from "react-router-dom";
import { useState } from "react";
import Header from "../components/Header";
import BottomNav from "../components/BottomNav";

const LABELS = {
  en: {
    confidence: "Confidence",
    severity: "Severity",
    original: "Original Image",
    heatmap: "AI Detection (Heatmap)",
    about: "About this Disease",
    viewMore: "View More ↓",
    top3: "Top 3 Predictions",
    warning: "⚠️ Warning",
    treatment: "View Treatment →",
    scanAgain: "Scan Another",
    feedbackQ: "Was this prediction correct?",
    yes: "✅ Yes, Correct",
    no: "❌ No, Wrong",
    thanks: "✅ Thanks for your feedback!",
    disclaimer: "⚠️ Disclaimer",
    helpline: "📞 Kisan Call Centre: 1800-180-1551",
    result: "Result",
    notPaddy: "Not a Paddy Leaf",
    notPaddyMsg: "Please upload a clear photo of a rice/paddy leaf.",
    uncertain: "Low Confidence",
    qualityErr: "Image Quality Issue",
    tryAgain: "Try Again",
    error: "Connection Error",
    errorMsg: "Make sure the backend server is running on port 8000.",
    severe: "Severe",
    moderate: "Moderate",
    mild: "Mild",
    none: "None",
  },
  hi: {
    confidence: "विश्वास",
    severity: "गंभीरता",
    original: "मूल छवि",
    heatmap: "AI जांच (हीटमैप)",
    about: "इस रोग के बारे में",
    viewMore: "और देखें ↓",
    top3: "शीर्ष 3 भविष्यवाणियाँ",
    warning: "⚠️ चेतावनी",
    treatment: "उपचार देखें →",
    scanAgain: "दूसरी स्कैन करें",
    feedbackQ: "क्या यह भविष्यवाणी सही थी?",
    yes: "✅ हाँ, सही",
    no: "❌ नहीं, गलत",
    thanks: "✅ आपकी प्रतिक्रिया के लिए धन्यवाद!",
    disclaimer: "⚠️ अस्वीकरण",
    helpline: "📞 किसान कॉल सेंटर: 1800-180-1551",
    result: "परिणाम",
    notPaddy: "धान की पत्ती नहीं",
    notPaddyMsg: "कृपया धान की पत्ती की स्पष्ट फोटो अपलोड करें।",
    uncertain: "कम विश्वास",
    qualityErr: "छवि गुणवत्ता समस्या",
    tryAgain: "पुनः प्रयास करें",
    error: "कनेक्शन त्रुटि",
    errorMsg: "सुनिश्चित करें कि बैकएंड सर्वर चल रहा है।",
    severe: "गंभीर",
    moderate: "मध्यम",
    mild: "हल्का",
    none: "कोई नहीं",
  },
  bn: {
    confidence: "আস্থা",
    severity: "তীব্রতা",
    original: "মূল ছবি",
    heatmap: "AI শনাক্তকরণ (হিটম্যাপ)",
    about: "এই রোগ সম্পর্কে",
    viewMore: "আরো দেখুন ↓",
    top3: "শীর্ষ ৩ পূর্বাভাস",
    warning: "⚠️ সতর্কতা",
    treatment: "চিকিৎসা দেখুন →",
    scanAgain: "আরেকটি স্ক্যান",
    feedbackQ: "এই পূর্বাভাস কি সঠিক ছিল?",
    yes: "✅ হ্যাঁ, সঠিক",
    no: "❌ না, ভুল",
    thanks: "✅ আপনার মতামতের জন্য ধন্যবাদ!",
    disclaimer: "⚠️ দাবিত্যাগ",
    helpline: "📞 কিসান কল সেন্টার: 1800-180-1551",
    result: "ফলাফল",
    notPaddy: "ধানের পাতা নয়",
    notPaddyMsg: "অনুগ্রহ করে ধানের পাতার স্পষ্ট ছবি আপলোড করুন।",
    uncertain: "কম আস্থা",
    qualityErr: "ছবির মান সমস্যা",
    tryAgain: "আবার চেষ্টা করুন",
    error: "সংযোগ ত্রুটি",
    errorMsg: "নিশ্চিত করুন ব্যাকএন্ড সার্ভার চলছে।",
    severe: "গুরুতর",
    moderate: "মাঝারি",
    mild: "হালকা",
    none: "কোনোটি নয়",
  },
};

// ── FIX 2: SVG colored dots instead of emoji (cross-platform safe) ──
const DISEASE_ICONS = {
  Bacterialblight: (
    <svg width="32" height="32" viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="14" fill="#fef2f2" stroke="#dc2626" strokeWidth="2"/>
      <text x="16" y="21" textAnchor="middle" fontSize="16">🦠</text>
    </svg>
  ),
  Blast: (
    <svg width="32" height="32" viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="14" fill="#fff7ed" stroke="#ea580c" strokeWidth="2"/>
      <text x="16" y="21" textAnchor="middle" fontSize="16">💥</text>
    </svg>
  ),
  Brownspot: (
    // FIX: solid brown circle instead of 🟤 which renders badly
    <svg width="32" height="32" viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="14" fill="#92400e"/>
      <circle cx="16" cy="16" r="8" fill="#b45309"/>
      <circle cx="16" cy="16" r="4" fill="#d97706"/>
    </svg>
  ),
  Healthy: (
    <svg width="32" height="32" viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="14" fill="#d8f3dc" stroke="#2d6a4f" strokeWidth="2"/>
      <text x="16" y="21" textAnchor="middle" fontSize="16">✅</text>
    </svg>
  ),
  Tungro: (
    <svg width="32" height="32" viewBox="0 0 32 32">
      <circle cx="16" cy="16" r="14" fill="#fef9c3" stroke="#d97706" strokeWidth="2"/>
      <text x="16" y="21" textAnchor="middle" fontSize="16">🐛</text>
    </svg>
  ),
};

const SCIENTIFIC = {
  Bacterialblight: "Xanthomonas oryzae",
  Blast: "Magnaporthe oryzae",
  Brownspot: "Cochliobolus miyabeanus",
  Healthy: "No disease detected",
  Tungro: "Rice Tungro Virus",
};

function getRiskClass(risk) {
  if (!risk) return "low";
  const r = risk.toLowerCase();
  if (r === "high") return "high";
  if (r === "moderate") return "moderate";
  return "low";
}

// ── Edge case screens ──
function EdgeScreen({ icon, title, msg, tip, btnLabel, onBtn, language }) {
  return (
    <div className="page" style={{
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "2rem", textAlign: "center", minHeight: "100vh"
    }}>
      <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>{icon}</div>
      <div style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "0.5rem" }}>{title}</div>
      <div style={{ fontSize: "0.85rem", color: "var(--gray-500)", marginBottom: "1rem" }}>{msg}</div>
      {tip && <div style={{ fontSize: "0.82rem", color: "var(--green-700)", marginBottom: "1.5rem", fontWeight: 600 }}>💡 {tip}</div>}
      <button className="btn-primary" onClick={onBtn}>{btnLabel}</button>
      <BottomNav language={language} />
    </div>
  );
}

export default function Result({ language, result, setResult }) {
  const navigate = useNavigate();
  const [feedback, setFeedback] = useState(null);
  const [showFull, setShowFull] = useState(false);
  const L = LABELS[language] || LABELS.en;

  if (!result) return null;

  const goHome = () => { setResult(null); navigate("/"); };

  if (result.status === "error")
    return <EdgeScreen icon="❌" title={L.error} msg={L.errorMsg} btnLabel={L.tryAgain} onBtn={goHome} language={language} />;

  if (result.status === "not_paddy")
    return <EdgeScreen icon="🚫" title={L.notPaddy} msg={result.message || L.notPaddyMsg} tip={result.tip} btnLabel={L.tryAgain} onBtn={goHome} language={language} />;

  if (result.status === "quality_error")
    return <EdgeScreen icon="📷" title={L.qualityErr} msg={result.message} tip={result.tip} btnLabel={L.tryAgain} onBtn={goHome} language={language} />;

  if (result.status === "uncertain")
    return <EdgeScreen icon="🤔" title={L.uncertain} msg={result.message} tip={result.tip} btnLabel={L.tryAgain} onBtn={goHome} language={language} />;

  // ── Success ──
  const riskClass = getRiskClass(result.risk_level);
  const conf      = result.confidence || 0;
  const severity  = result.severity || "None";

  async function sendFeedback(correct) {
    setFeedback(correct);
    if (result.prediction_id) {
      try {
        await fetch(`/feedback/${result.prediction_id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ was_correct: correct ? "yes" : "no" }),
        });
      } catch { /* silent */ }
    }
  }

  return (
    <div className="result-page animate-in">
      <Header title={L.result} language={language} />

      {/* ── Disease Banner ── */}
      <div className={`result-disease-header ${riskClass}`}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>

          {/* FIX 2: use SVG icon instead of raw emoji */}
          <span style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
            {DISEASE_ICONS[result.disease] || (
              <svg width="32" height="32" viewBox="0 0 32 32">
                <circle cx="16" cy="16" r="14" fill="rgba(255,255,255,0.2)"/>
                <text x="16" y="21" textAnchor="middle" fontSize="16">🌿</text>
              </svg>
            )}
          </span>

          <div>
            <div className="disease-name">{result.disease}</div>
            <div className="disease-scientific">{SCIENTIFIC[result.disease] || ""}</div>
          </div>
          <div className="risk-badge" style={{ marginLeft: "auto" }}>
            {result.risk_level || "Low"} Risk
          </div>
        </div>
      </div>

      {/* ── Confidence ── */}
      <div className="result-section">
        <div className="section-title">{L.confidence}</div>
        <div className="confidence-row">
          <span>{L.confidence}</span>
          <span style={{ color: "var(--green-700)", fontWeight: 800 }}>{conf.toFixed(1)}%</span>
        </div>
        <div className="conf-bar">
          <div className="conf-fill" style={{ width: `${conf}%` }} />
        </div>
        <div className="section-title" style={{ marginTop: "0.75rem" }}>{L.severity}</div>
        <div className="severity-row">
          {["Severe", "Moderate", "Mild", "None"].map((s) => (
            <div key={s} className={`severity-pill ${severity === s ? `active ${s.toLowerCase()}` : ""}`}>
              {L[s.toLowerCase()] || s}
            </div>
          ))}
        </div>
      </div>

      {/* ── FIX 1: Heatmap — show Original + Heatmap side by side, no duplicate ── */}
      {(result.image || result.heatmap) && (
        <div className="result-section">
          <div className="section-title">{L.original} / {L.heatmap}</div>
          <div className="heatmap-container">
            {/* Original image */}
            {result.image && (
              <div>
                <img
                  src={`data:image/jpeg;base64,${result.image}`}
                  alt="original leaf"
                  className="heatmap-img"
                />
                <div className="heatmap-label">{L.original}</div>
              </div>
            )}
            {/* Heatmap only — NOT a composite, just the pure heatmap overlay */}
            {result.heatmap && (
              <div>
                <img
                  src={`data:image/png;base64,${result.heatmap}`}
                  alt="AI heatmap"
                  className="heatmap-img"
                />
                <div className="heatmap-label">{L.heatmap}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── About Disease ── */}
      {result.description && (
        <div className="result-section">
          <div className="section-title">{L.about}</div>
          <p style={{ fontSize: "0.85rem", color: "var(--gray-500)", lineHeight: 1.6 }}>
            {showFull ? result.description : result.description.slice(0, 120) + "..."}
          </p>
          <button
            onClick={() => setShowFull(!showFull)}
            style={{ background: "none", border: "none", color: "var(--green-700)", fontWeight: 700,
              fontSize: "0.82rem", cursor: "pointer", marginTop: "0.5rem", padding: 0, fontFamily: "Poppins, sans-serif" }}
          >
            {showFull ? "Show less ↑" : L.viewMore}
          </button>
        </div>
      )}

      {/* ── Top 3 Predictions ── */}
      {result.top3 && result.top3.length > 0 && (
        <div className="result-section">
          <div className="section-title">{L.top3}</div>
          {result.top3.map((item, i) => (
            <div className="top3-item" key={i}>
              <div className={`top3-rank ${i === 0 ? "first" : ""}`}>{i + 1}</div>
              <span className="top3-name">{item.disease}</span>
              <div className="top3-bar-wrap">
                <div className="top3-bar" style={{ width: `${item.probability}%` }} />
              </div>
              <span className="top3-prob">{item.probability.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Warning ── */}
      {result.warning && (
        <div className="warning-box">
          <span>⚠️</span>
          <span>{result.warning}</span>
        </div>
      )}

      {/* ── Actions ── */}
      <div className="result-actions">
        <button className="btn-result-action btn-treatment" onClick={() => navigate("/treatment")}>
          {L.treatment}
        </button>
        <button className="btn-result-action btn-scan-again" onClick={goHome}>
          {L.scanAgain}
        </button>
      </div>

      {/* ── Feedback ── */}
      <div className="feedback-section">
        <div className="feedback-title">{L.feedbackQ}</div>
        {feedback === null ? (
          <div className="feedback-buttons">
            <button className="btn-feedback correct" onClick={() => sendFeedback(true)}>{L.yes}</button>
            <button className="btn-feedback wrong"   onClick={() => sendFeedback(false)}>{L.no}</button>
          </div>
        ) : (
          <div style={{ textAlign: "center", color: "var(--green-700)", fontWeight: 600, fontSize: "0.85rem" }}>
            {L.thanks}
          </div>
        )}
      </div>

      {/* ── Disclaimer ── */}
      <div className="disclaimer-box">
        <div className="disclaimer-title">{L.disclaimer}</div>
        <p className="disclaimer-text">
          {result.disclaimer?.[language] || result.disclaimer?.en ||
            "This AI tool provides preliminary disease screening ONLY. Always verify with a certified agricultural officer."}
        </p>
      </div>
      <div className="helpline-box">{L.helpline}</div>

      <BottomNav language={language} />
    </div>
  );
}