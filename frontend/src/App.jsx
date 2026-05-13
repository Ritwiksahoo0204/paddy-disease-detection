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

export default function App() {
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const [language, setLanguage]             = useState("en");
  const [result, setResult]                 = useState(null);

  return (
    <BrowserRouter>
      {showDisclaimer && (
        <DisclaimerModal
          language={language}
          onAccept={() => setShowDisclaimer(false)}
        />
      )}
      <Routes>
        <Route path="/"           element={<Home language={language} setLanguage={setLanguage} setResult={setResult} />} />
        <Route path="/camera"     element={<Camera language={language} setResult={setResult} />} />
        <Route path="/analyzing"  element={<Analyzing language={language} />} />
        <Route path="/result"     element={result ? <Result language={language} result={result} setResult={setResult} /> : <Navigate to="/" />} />
        <Route path="/treatment"  element={result ? <Treatment language={language} result={result} /> : <Navigate to="/" />} />
        <Route path="/history"    element={<History language={language} />} />
        <Route path="/dashboard"  element={<Dashboard />} />
        <Route path="*"           element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
}