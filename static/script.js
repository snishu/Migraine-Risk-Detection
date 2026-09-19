// Live-update label next to each range slider
const sliders = [
  "age", "sleep_hours", "stress_level", "noise_sensitivity", "light_sensitivity",
  "screen_time", "water_intake", "caffeine_cups", "skipped_meals", "physical_activity"
];

sliders.forEach((id) => {
  const el = document.getElementById(id);
  const label = document.getElementById(id + "_val");
  if (el && label) {
    el.addEventListener("input", () => (label.textContent = el.value));
  }
});

const form = document.getElementById("riskForm");
const submitBtn = document.getElementById("submitBtn");
const emptyState = document.getElementById("emptyState");
const resultBody = document.getElementById("resultBody");
const riskBadge = document.getElementById("riskBadge");
const riskText = document.getElementById("riskText");
const probList = document.getElementById("probList");
const tipsList = document.getElementById("tipsList");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  const payload = {
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    sleep_hours: document.getElementById("sleep_hours").value,
    stress_level: document.getElementById("stress_level").value,
    screen_time: document.getElementById("screen_time").value,
    water_intake: document.getElementById("water_intake").value,
    skipped_meals: document.getElementById("skipped_meals").value,
    caffeine_cups: document.getElementById("caffeine_cups").value,
    physical_activity: document.getElementById("physical_activity").value,
    weather_sensitive: document.getElementById("weather_sensitive").checked ? 1 : 0,
    family_history: document.getElementById("family_history").checked ? 1 : 0,
    hormonal_changes: document.getElementById("hormonal_changes").checked ? 1 : 0,
    alcohol: document.getElementById("alcohol").checked ? 1 : 0,
    noise_sensitivity: document.getElementById("noise_sensitivity").value,
    light_sensitivity: document.getElementById("light_sensitivity").value,
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.success) {
      alert("Something went wrong: " + data.error);
      return;
    }

    renderResult(data);
  } catch (err) {
    alert("Could not reach the server. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check my migraine risk";
  }
});

function renderResult(data) {
  emptyState.style.display = "none";
  resultBody.classList.add("show");

  riskBadge.className = "risk-badge " + data.prediction;
  riskText.textContent = data.prediction + " risk";

  probList.innerHTML = "";
  if (data.probabilities) {
    // Sort High, Medium, Low for consistent display
    const order = ["High", "Medium", "Low"];
    order.forEach((cls) => {
      if (!(cls in data.probabilities)) return;
      const pct = data.probabilities[cls];
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-label"><span>${cls}</span><span>${pct}%</span></div>
        <div class="prob-track"><div class="prob-fill ${cls}" style="width:${pct}%"></div></div>
      `;
      probList.appendChild(row);
    });
  }

  tipsList.innerHTML = "";
  (data.tips || []).forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    tipsList.appendChild(li);
  });

  resultBody.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
