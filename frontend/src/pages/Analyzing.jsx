import { useEffect, useState } from "react";

const CONTENT = {
  en: {
    title: "AI is analyzing the leaf",
    subtitle: "Please wait a few seconds",
    checking: "What we check?",
    checks: [
      "Leaf pattern and texture",
      "Color and spot analysis",
      "Disease type identification",
      "Severity estimation",
    ],
  },
  hi: {
    title: "AI पत्ती का विश्लेषण कर रहा है",
    subtitle: "कृपया कुछ सेकंड प्रतीक्षा करें",
    checking: "हम क्या जांचते हैं?",
    checks: [
      "पत्ती का पैटर्न और बनावट",
      "रंग और धब्बे का विश्लेषण",
      "रोग प्रकार की पहचान",
      "गंभीरता का अनुमान",
    ],
  },
  bn: {
    title: "AI পাতাটি বিশ্লেষণ করছে",
    subtitle: "অনুগ্রহ করে কয়েক সেকেন্ড অপেক্ষা করুন",
    checking: "আমরা কী পরীক্ষা করি?",
    checks: [
      "পাতার প্যাটার্ন এবং গঠন",
      "রঙ ও দাগ বিশ্লেষণ",
      "রোগের ধরন শনাক্তকরণ",
      "তীব্রতার অনুমান",
    ],
  },
};

const RADIUS      = 80;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function Analyzing({ language }) {
  const [progress, setProgress] = useState(0);
  const [doneCount, setDoneCount] = useState(0);
  const c = CONTENT[language] || CONTENT.en;

  useEffect(() => {
    // Animate progress from 0 → 95 over ~3 seconds
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 95) { clearInterval(interval); return 95; }
        return p + 1;
      });
    }, 35);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Mark checklist items as done progressively
    const thresholds = [20, 45, 70, 90];
    const done = thresholds.filter((t) => progress >= t).length;
    setDoneCount(done);
  }, [progress]);

  const offset = CIRCUMFERENCE - (progress / 100) * CIRCUMFERENCE;

  return (
    <div className="analyzing-page animate-in">
      {/* ── Circular Progress ── */}
      <div className="circular-progress">
        <svg width="180" height="180" viewBox="0 0 180 180">
          <circle
            className="progress-bg"
            cx="90" cy="90" r={RADIUS}
          />
          <circle
            className="progress-fill"
            cx="90" cy="90" r={RADIUS}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="progress-text">
          <span className="progress-percent">{progress}%</span>
          <span className="progress-icon">🌿</span>
        </div>
      </div>

      {/* ── Text ── */}
      <div>
        <div className="analyzing-title">{c.title}</div>
        <div className="analyzing-subtitle">{c.subtitle}</div>
      </div>

      {/* ── Checklist ── */}
      <div className="check-list">
        <div className="check-title">{c.checking}</div>
        {c.checks.map((item, i) => (
          <div
            key={i}
            className={`check-item ${i < doneCount ? "done" : ""}`}
          >
            <span>{i < doneCount ? "✅" : "⏳"}</span>
            <span>{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
}