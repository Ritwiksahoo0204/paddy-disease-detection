import { useNavigate } from "react-router-dom";
import { useRef } from "react";
import BottomNav from "../components/BottomNav";

const CONTENT = {
  en: {
    title: "Paddy Disease\nDetection AI",
    subtitle: "Smart AI for Healthy Paddy",
    takePhoto: "📷  Take Photo",
    uploadGallery: "🖼️  Upload from Gallery",
    tipsTitle: "Tips for Better Detection",
    tips: [
      { icon: "☀️", text: "Use clear and bright lighting" },
      { icon: "🎯", text: "Focus on a single leaf" },
      { icon: "📵", text: "Avoid blurry images" },
      { icon: "🌿", text: "Keep the leaf centered" },
    ],
    statsLabels: ["21,974", "5", "94.26%"],
    statsNames: ["Images", "Diseases", "Accuracy"],
  },
  hi: {
    title: "धान रोग\nपहचान AI",
    subtitle: "स्वस्थ धान के लिए स्मार्ट AI",
    takePhoto: "📷  फोटो लें",
    uploadGallery: "🖼️  गैलरी से अपलोड करें",
    tipsTitle: "बेहतर जांच के लिए सुझाव",
    tips: [
      { icon: "☀️", text: "साफ और उज्ज्वल रोशनी का उपयोग करें" },
      { icon: "🎯", text: "एक पत्ती पर ध्यान केंद्रित करें" },
      { icon: "📵", text: "धुंधली छवियों से बचें" },
      { icon: "🌿", text: "पत्ती को केंद्र में रखें" },
    ],
    statsLabels: ["21,974", "5", "94.26%"],
    statsNames: ["छवियां", "रोग", "सटीकता"],
  },
  bn: {
    title: "ধান রোগ\nশনাক্তকরণ AI",
    subtitle: "সুস্থ ধানের জন্য স্মার্ট AI",
    takePhoto: "📷  ছবি তুলুন",
    uploadGallery: "🖼️  গ্যালারি থেকে আপলোড করুন",
    tipsTitle: "ভালো শনাক্তকরণের জন্য টিপস",
    tips: [
      { icon: "☀️", text: "পরিষ্কার ও উজ্জ্বল আলো ব্যবহার করুন" },
      { icon: "🎯", text: "একটি পাতায় ফোকাস করুন" },
      { icon: "📵", text: "ঝাপসা ছবি এড়িয়ে চলুন" },
      { icon: "🌿", text: "পাতাটি কেন্দ্রে রাখুন" },
    ],
    statsLabels: ["21,974", "5", "94.26%"],
    statsNames: ["ছবি", "রোগ", "নির্ভুলতা"],
  },
};

const LANGS = ["en", "hi", "bn"];
const LANG_LABELS = { en: "EN", hi: "हि", bn: "বাং" };

export default function Home({ language, setLanguage, setResult }) {
  const navigate  = useNavigate();
  const fileInput = useRef(null);
  const c         = CONTENT[language] || CONTENT.en;

  async function handleFile(file) {
    if (!file) return;
    navigate("/analyzing");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res  = await fetch(`${API}/predict`, { method: "POST", body: formData });
      const data = await res.json();
      setResult(data);
      navigate("/result");
    } catch {
      setResult({ status: "error", message: "Could not connect to server. Make sure the backend is running." });
      navigate("/result");
    }
  }

  return (
    <div className="page">
      {/* ── Hero ── */}
      <div className="hero">
        <div className="hero-brand">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flex: 1 }}>
            <span className="hero-logo">🌾</span>
            <div>
              <div className="hero-name">Paddy Doctor</div>
              <div className="hero-tagline">AI Powered</div>
            </div>
          </div>
          {/* Language selector */}
          <div className="lang-selector-hero">
            {LANGS.map((l) => (
              <button
                key={l}
                className={`lang-btn-hero ${language === l ? "active" : ""}`}
                onClick={() => setLanguage(l)}
              >
                {LANG_LABELS[l]}
              </button>
            ))}
          </div>
        </div>

        <div className="hero-title">{c.title}</div>
        <div className="hero-subtitle">{c.subtitle}</div>
        <div className="hero-image">🌾</div>
      </div>

      {/* ── Stats Row ── */}
      <div className="stats-row">
        {c.statsLabels.map((val, i) => (
          <div className="stat-card" key={i}>
            <div className="stat-value">{val}</div>
            <div className="stat-label">{c.statsNames[i]}</div>
          </div>
        ))}
      </div>

      {/* ── Action Buttons ── */}
      <div className="action-buttons">
        <button className="btn-primary" onClick={() => navigate("/camera")}>
          <span className="btn-icon">📷</span>
          {c.takePhoto.replace("📷  ", "")}
        </button>

        <button
          className="btn-secondary"
          onClick={() => fileInput.current?.click()}
        >
          <span className="btn-icon">🖼️</span>
          {c.uploadGallery.replace("🖼️  ", "")}
        </button>

        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {/* ── Tips ── */}
      <div className="tips-card">
        <div className="tips-title">{c.tipsTitle}</div>
        <div className="tips-grid">
          {c.tips.map((tip, i) => (
            <div className="tip-item" key={i}>
              <span>{tip.icon}</span>
              <span>{tip.text}</span>
            </div>
          ))}
        </div>
      </div>

      <BottomNav language={language} />
    </div>
  );
}