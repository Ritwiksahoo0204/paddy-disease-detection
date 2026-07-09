// Generates (or reuses) a random, anonymous per-browser ID so each device's
// scan history stays private without requiring a login system.
// Stored in localStorage so it persists across visits on the same browser.

const SESSION_KEY = "paddy_doctor_session_id";

export function getSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}