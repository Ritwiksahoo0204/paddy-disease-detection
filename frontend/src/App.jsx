import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import Camera from "./pages/Camera";
import Analyzing from "./pages/Analyzing";
import Result from "./pages/Result";
import Treatment from "./pages/Treatment";
import History from "./pages/History";
import Dashboard from "./pages/Dashboard";
import DisclaimerModal from "./components/DisclaimerModal";
import "./App.css";

const DISCLAIMER_KEY = "paddy_doctor_disclaimer_accepted";

export default function App() {
  const [showDisclaimer, setShowDisclaimer] = useState(
    () => localStorage.getItem(DISCLAIMER_KEY) !== "true"
  );
  const [language, setLanguage]             = useState("en");
  const [result, setResult]                 = useState(null);

  function handleAcceptDisclaimer() {
    localStorage.setItem(DISCLAIMER_KEY, "true");
    setShowDisclaimer(false);
  }

  return (
    <BrowserRouter>
      {showDisclaimer && (
        <DisclaimerModal
          language={language}
          onAccept={handleAcceptDisclaimer}
        />
      )}
      <Routes>
        <Route path="/"           element={<Home language={language} setLanguage={setLanguage} setResult={setResult} />} />
        <Route path="/camera"     element={<Camera language={language} setResult={setResult} />} />
        <Route path="/analyzing"  element={<Analyzing language={language} />} />
        <Route path="/result"     element={result ? <Result language={language} result={result} setResult={setResult} /> : <Navigate to="/" />} />
        <Route path="/result/:id" element={<Result language={language} result={result} setResult={setResult} />} />
        <Route path="/treatment"  element={result ? <Treatment language={language} result={result} /> : <Navigate to="/" />} />
        <Route path="/history"    element={<History language={language} />} />
        <Route path="/dashboard"  element={<Dashboard />} />
        <Route path="*"           element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
}