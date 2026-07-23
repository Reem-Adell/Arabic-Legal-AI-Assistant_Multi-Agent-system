// Same-origin API base: FastAPI serves this frontend, so relative paths work
// both locally and behind the ngrok tunnel.
const API_BASE = "";

const SESSION_ID = (() => {
  let id = sessionStorage.getItem("legal_ai_session_id");
  if (!id) {
    id = "sess_" + Math.random().toString(36).slice(2) + Date.now();
    sessionStorage.setItem("legal_ai_session_id", id);
  }
  return id;
})();

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const appointmentBox = document.getElementById("appointment-box");
const lawyerList = document.getElementById("lawyer-list");

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

function renderAppointment(appt) {
  if (!appt) return;
  appointmentBox.classList.remove("empty");
  appointmentBox.innerHTML = `
    <div class="row"><span class="label">المحامي:</span><span>${appt.lawyer_name}</span></div>
    <div class="row"><span class="label">التخصص:</span><span>${appt.lawyer_specialization}</span></div>
    <div class="row"><span class="label">التاريخ:</span><span>${appt.appointment_date}</span></div>
    <div class="row"><span class="label">الوقت:</span><span>${appt.appointment_time}</span></div>
    <div class="row"><span class="label">الحالة:</span><span>${appt.status}</span></div>
  `;
}

async function loadLawyers() {
  try {
    const res = await fetch(`${API_BASE}/api/lawyers`);
    const lawyers = await res.json();
    lawyerList.innerHTML = lawyers
      .map(l => `<li><span>${l.name}</span><span class="spec">${l.specialization}</span></li>`)
      .join("");
  } catch (e) {
    console.error("failed to load lawyers", e);
  }
}

async function sendMessage(message) {
  addMessage(message, "user");
  const typingEl = addMessage("...يكتب المساعد", "bot typing");

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID, message }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    typingEl.remove();
    addMessage(data.reply, "bot");
    if (data.appointment) {
      renderAppointment(data.appointment);
    }
  } catch (e) {
    typingEl.remove();
    addMessage("حدث خطأ أثناء الاتصال بالخادم. حاول مرة أخرى.", "bot");
    console.error(e);
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const value = chatInput.value.trim();
  if (!value) return;
  chatInput.value = "";
  sendMessage(value);
});

loadLawyers();
