import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Legend
} from "recharts";

const COLORS = ["#2d6a4f", "#52b788", "#ea580c", "#dc2626", "#d97706"];

const ADMIN_USER = import.meta.env.VITE_ADMIN_USER || "admin";
const ADMIN_PASS = import.meta.env.VITE_ADMIN_PASS || "paddydoctor2024";

export default function Dashboard() {
  const navigate            = useNavigate();
  const [authed, setAuthed] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]   = useState("");
  const [stats, setStats]   = useState(null);
  const [loading, setLoading] = useState(false);

  function handleLogin(e) {
    e.preventDefault();
    if (username === ADMIN_USER && password === ADMIN_PASS) {
      setAuthed(true);
      setError("");
      loadStats();
    } else {
      setError("Invalid username or password");
    }
  }

  async function loadStats() {
    setLoading(true);
    try {
      const creds = btoa(`${ADMIN_USER}:${ADMIN_PASS}`);
      const res   = await fetch("http://localhost:8000/dashboard/stats", {
        headers: { Authorization: `Basic ${creds}` },
      });
      const data  = await res.json();
      setStats(data);
    } catch {
      setStats(null);
    } finally {
      setLoading(false);
    }
  }

  // ── Login Screen ──
  if (!authed) {
    return (
      <div className="admin-login">
        <div className="login-card">
          <div className="login-logo">🌾</div>
          <div className="login-title">Admin Dashboard</div>
          <div className="login-sub">Paddy Doctor — Analytics Panel</div>

          {error && <div className="login-error">{error}</div>}

          <form onSubmit={handleLogin}>
            <input
              className="login-input"
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <input
              className="login-input"
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button className="btn-login" type="submit">
              Login →
            </button>
          </form>

          <button
            onClick={() => navigate("/")}
            style={{
              marginTop: "1rem", background: "none", border: "none",
              color: "var(--gray-500)", fontSize: "0.82rem", cursor: "pointer",
              fontFamily: "Poppins, sans-serif"
            }}
          >
            ← Back to App
          </button>
        </div>
      </div>
    );
  }

  // ── Dashboard Screen ──
  return (
    <div className="dashboard-page">
      {/* ── Header ── */}
      <div className="dashboard-header">
        <div>
          <div className="dashboard-title">📊 Dashboard</div>
          <div className="dashboard-sub">Paddy Doctor Analytics</div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <span className="admin-badge">👤 Admin</span>
          <button
            onClick={() => { setAuthed(false); setStats(null); }}
            style={{
              background: "rgba(255,255,255,0.15)", border: "none",
              color: "white", padding: "0.3rem 0.75rem", borderRadius: "20px",
              fontSize: "0.75rem", cursor: "pointer", fontFamily: "Poppins, sans-serif"
            }}
          >
            Logout
          </button>
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <div className="loading-spinner" />
        </div>
      )}

      {!loading && stats && (
        <>
          {/* ── Stats Grid ── */}
          <div className="stats-grid">
            <div className="stat-card-dash">
              <div className="stat-icon">📸</div>
              <div className="stat-number">{stats.total_scans ?? 0}</div>
              <div className="stat-name">Total Scans</div>
              <div className="stat-change">↑ All time</div>
            </div>
            <div className="stat-card-dash">
              <div className="stat-icon">🦠</div>
              <div className="stat-number">{stats.diseased_leaves ?? 0}</div>
              <div className="stat-name">Diseased Leaves</div>
              <div className="stat-change" style={{ color: "var(--red)" }}>Needs attention</div>
            </div>
            <div className="stat-card-dash">
              <div className="stat-icon">✅</div>
              <div className="stat-number">{stats.healthy_leaves ?? 0}</div>
              <div className="stat-name">Healthy Leaves</div>
              <div className="stat-change">↑ Good news</div>
            </div>
            <div className="stat-card-dash">
              <div className="stat-icon">🎯</div>
              <div className="stat-number">{stats.model_accuracy}%</div>
              <div className="stat-name">Model Accuracy</div>
              <div className="stat-change">MobileNetV2</div>
            </div>
          </div>

          {/* ── Disease Distribution Pie ── */}
          {stats.disease_distribution?.length > 0 && (
            <div className="chart-card">
              <div className="chart-title">🥧 Disease Distribution</div>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={stats.disease_distribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                    labelLine={false}
                  >
                    {stats.disease_distribution.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(val) => [`${val} scans`, ""]} />
                </PieChart>
              </ResponsiveContainer>

              {/* Legend */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "0.5rem" }}>
                {stats.disease_distribution.map((item, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.75rem" }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS[i % COLORS.length] }} />
                    <span>{item.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Daily Scans Line Chart ── */}
          {stats.daily_scans?.length > 0 && (
            <div className="chart-card">
              <div className="chart-title">📈 Scans Over Time</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={[...stats.daily_scans].reverse()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10 }}
                    tickFormatter={(d) => d?.slice(5)}
                  />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="var(--green-700)"
                    strokeWidth={2}
                    dot={{ fill: "var(--green-700)", r: 4 }}
                    name="Scans"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ── Feedback Stats ── */}
          {stats.accuracy_feedback && Object.keys(stats.accuracy_feedback).length > 0 && (
            <div className="chart-card">
              <div className="chart-title">💬 User Feedback</div>
              <div style={{ display: "flex", gap: "1rem" }}>
                <div style={{
                  flex: 1, background: "var(--green-50)", borderRadius: "var(--radius-xs)",
                  padding: "1rem", textAlign: "center"
                }}>
                  <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--green-700)" }}>
                    {stats.accuracy_feedback.yes || 0}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--gray-500)" }}>✅ Correct</div>
                </div>
                <div style={{
                  flex: 1, background: "#fef2f2", borderRadius: "var(--radius-xs)",
                  padding: "1rem", textAlign: "center"
                }}>
                  <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--red)" }}>
                    {stats.accuracy_feedback.no || 0}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--gray-500)" }}>❌ Wrong</div>
                </div>
              </div>
            </div>
          )}

          {/* ── No Data ── */}
          {stats.total_scans === 0 && (
            <div style={{ textAlign: "center", padding: "2rem", color: "var(--gray-500)" }}>
              <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📊</div>
              <div style={{ fontSize: "0.9rem" }}>No scan data yet. Start using the app!</div>
            </div>
          )}
        </>
      )}

      {!loading && !stats && (
        <div style={{ textAlign: "center", padding: "2rem", color: "var(--gray-500)" }}>
          <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>⚠️</div>
          <div style={{ fontSize: "0.9rem" }}>Could not load stats. Make sure backend is running.</div>
          <button
            className="btn-primary"
            style={{ marginTop: "1rem", width: "auto", padding: "0.75rem 2rem" }}
            onClick={loadStats}
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
}