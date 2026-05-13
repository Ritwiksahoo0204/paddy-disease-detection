const CONTENT = {
  en: {
    title: "Paddy Doctor",
    subtitle: "Before You Continue",
    warning: "⚠️ Important Disclaimer",
    warningText: "This AI tool provides preliminary disease screening ONLY. It is NOT a substitute for professional agricultural advice. Wrong treatment can seriously damage your crop and cause financial loss.",
    points: [
      "✅ Use for initial screening only",
      "❌ Never apply chemicals based solely on AI result",
      "👨‍🌾 Always verify with a certified agricultural officer",
      "📸 Upload clear, well-lit photos for best accuracy",
      "🔄 Results may vary based on image quality",
    ],
    helpline: "📞 Kisan Call Centre: 1800-180-1551 (Toll Free, 24/7)",
    accept: "I Understand — Continue",
  },
  hi: {
    title: "पैडी डॉक्टर",
    subtitle: "जारी रखने से पहले",
    warning: "⚠️ महत्वपूर्ण चेतावनी",
    warningText: "यह AI उपकरण केवल प्रारंभिक रोग जांच के लिए है। यह पेशेवर कृषि सलाह का विकल्प नहीं है। गलत उपचार से फसल को नुकसान हो सकता है।",
    points: [
      "✅ केवल प्रारंभिक जांच के लिए उपयोग करें",
      "❌ AI परिणाम के आधार पर कभी रसायन न लगाएं",
      "👨‍🌾 हमेशा प्रमाणित कृषि अधिकारी से सत्यापित करें",
      "📸 सटीक परिणाम के लिए स्पष्ट फोटो अपलोड करें",
      "🔄 छवि गुणवत्ता के आधार पर परिणाम भिन्न हो सकते हैं",
    ],
    helpline: "📞 किसान कॉल सेंटर: 1800-180-1551 (निःशुल्क, 24/7)",
    accept: "मैं समझता हूं — जारी रखें",
  },
  bn: {
    title: "পেডি ডক্টর",
    subtitle: "চালিয়ে যাওয়ার আগে",
    warning: "⚠️ গুরুত্বপূর্ণ সতর্কতা",
    warningText: "এই AI টুল শুধুমাত্র প্রাথমিক রোগ নির্ণয়ের জন্য। এটি পেশাদার কৃষি পরামর্শের বিকল্প নয়। ভুল চিকিৎসা ফসলের মারাত্মক ক্ষতি করতে পারে।",
    points: [
      "✅ শুধুমাত্র প্রাথমিক স্ক্রিনিংয়ের জন্য ব্যবহার করুন",
      "❌ AI ফলাফলের ভিত্তিতে রাসায়নিক প্রয়োগ করবেন না",
      "👨‍🌾 সর্বদা একজন কৃষি কর্মকর্তার সাথে যাচাই করুন",
      "📸 সঠিক ফলাফলের জন্য স্পষ্ট ছবি আপলোড করুন",
      "🔄 ছবির মানের উপর ভিত্তি করে ফলাফল পরিবর্তিত হতে পারে",
    ],
    helpline: "📞 কিসান কল সেন্টার: 1800-180-1551 (বিনামূল্যে, 24/7)",
    accept: "আমি বুঝলাম — চালিয়ে যান",
  },
};

export default function DisclaimerModal({ language, onAccept }) {
  const c = CONTENT[language] || CONTENT.en;
  return (
    <div className="modal-overlay">
      <div className="modal-card">
        <div className="modal-header">
          <div className="modal-icon">🌾</div>
          <div className="modal-title">{c.title}</div>
        </div>
        <div className="modal-body">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem", color: "#1a472a" }}>
            {c.subtitle}
          </h3>
          <div className="modal-warning">
            <div className="modal-warning-title">{c.warning}</div>
            <p className="modal-warning-text">{c.warningText}</p>
          </div>
          <ul className="modal-points">
            {c.points.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
          <div className="modal-helpline">{c.helpline}</div>
          <button className="btn-accept" onClick={onAccept}>{c.accept}</button>
        </div>
      </div>
    </div>
  );
}