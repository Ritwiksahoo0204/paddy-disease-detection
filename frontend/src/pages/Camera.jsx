import { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const TIPS = {
  en: "Make sure the leaf is clear and visible",
  hi: "सुनिश्चित करें कि पत्ती साफ और दिखाई दे रही है",
  bn: "নিশ্চিত করুন পাতাটি পরিষ্কার এবং দৃশ্যমান",
};

const TITLES = {
  en: "Capture Leaf Image",
  hi: "पत्ती की छवि कैप्चर करें",
  bn: "পাতার ছবি তুলুন",
};

export default function Camera({ language, setResult }) {
  const navigate    = useNavigate();
  const videoRef    = useRef(null);
  const fileRef     = useRef(null);
  const streamRef   = useRef(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [ready, setReady]           = useState(false);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [facingMode]);

  async function startCamera() {
    stopCamera();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setReady(true);
      }
    } catch {
      // Camera not available — fall back to file upload
      setReady(false);
      fileRef.current?.click();
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setReady(false);
  }

  async function capture() {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => sendImage(blob), "image/jpeg", 0.92);
  }

  async function sendImage(blob) {
    stopCamera();
    navigate("/analyzing");
    const formData = new FormData();
    formData.append("file", blob, "capture.jpg");
    try {
      const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res  = await fetch(`${API}/predict`, { method: "POST", body: formData });
      const data = await res.json();
      setResult(data);
      navigate("/result");
    } catch {
      setResult({ status: "error", message: "Could not connect to server." });
      navigate("/result");
    }
  }

  async function handleGallery(e) {
    const file = e.target.files[0];
    if (!file) return;
    await sendImage(file);
  }

  return (
    <div className="camera-page">
      {/* ── Top bar ── */}
      <div className="camera-header">
        <button
          className="camera-header-btn"
          onClick={() => { stopCamera(); navigate(-1); }}
        >
          ←
        </button>
        <span className="camera-title">{TITLES[language] || TITLES.en}</span>
        <div style={{ width: 40 }} />
      </div>

      {/* ── Video feed ── */}
      <div className="camera-view">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-video"
        />

        {/* Leaf frame guide */}
        <div className="camera-frame" />

        {/* Tip */}
        <div className="camera-tip">
          ☀️ {TIPS[language] || TIPS.en}
        </div>

        {/* Controls */}
        <div className="camera-controls">
          {/* Gallery */}
          <button
            className="camera-gallery-btn"
            onClick={() => fileRef.current?.click()}
          >
            🖼️
          </button>

          {/* Capture */}
          <button className="camera-capture-btn" onClick={capture} />

          {/* Flip */}
          <button
            className="camera-flip-btn"
            onClick={() =>
              setFacingMode((f) => (f === "environment" ? "user" : "environment"))
            }
          >
            🔄
          </button>
        </div>
      </div>

      {/* Hidden file input for gallery */}
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleGallery}
      />
    </div>
  );
}