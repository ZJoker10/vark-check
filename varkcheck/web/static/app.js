/* VARK Check v2 — front end.
 *
 * Three steps, one state object, no framework. The only piece of real
 * machinery here is the preview: the report arrives as a complete HTML
 * document, is shown in a script-blocked iframe, and Mermaid is run from *this*
 * page over the iframe's DOM so the resulting SVG can be POSTed back for the
 * PDF/DOCX export. That keeps diagram rendering in the browser instead of
 * dragging Node into a Python app.
 */

const PROFILE_KEY = "varkcheck.profile.v1";
const UPLOAD_KEY = "varkcheck.upload.v1";

const state = {
  upload: null,
  profile: null,
  style: null,
  report: null,
  html: "",
  svgs: {},
  format: "pdf",
  generating: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

/* ------------------------------------------------------------------ helpers */
function show(node, visible = true) {
  node.classList.toggle("hidden", !visible);
}

function status(node, message, tone = "") {
  node.textContent = message;
  node.classList.remove("good", "bad");
  if (tone) node.classList.add(tone);
}

function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function unlock(step) {
  step.classList.remove("locked");
}

async function api(path, body, isForm = false) {
  const options = isForm
    ? { method: "POST", body }
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      };
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

/* -------------------------------------------------------------------- model */
async function loadHealth() {
  const badge = $("#model-badge");
  try {
    const health = await fetch("/api/health").then((r) => r.json());
    badge.textContent = `${health.provider} · ${health.model}`;
    badge.classList.toggle("offline", Boolean(health.offline));
    badge.title = health.offline
      ? "Runs without sending your material anywhere"
      : "Your material is sent to this provider for the report";
  } catch {
    badge.textContent = "server not reachable";
  }
}

/* ------------------------------------------------------------------ quizzes */
let questions = [];

async function loadQuiz() {
  const data = await fetch("/api/quiz").then((r) => r.json());
  questions = data.questions || [];
  const container = $("#quiz-items");
  container.innerHTML = questions
    .map(
      (question) => `
      <div class="quiz-item">
        <p>${escapeHtml(question.text)}</p>
        ${[1, 2, 3, 4, 5]
          .map(
            (value) => `
          <label title="${value} of 5">
            <input type="radio" name="${question.id}" value="${value}">
            <span class="dot">${value}</span>
          </label>`
          )
          .join("")}
      </div>`
    )
    .join("");

  container.addEventListener("change", () => {
    const answered = questions.filter((q) => container.querySelector(`input[name="${q.id}"]:checked`)).length;
    $("#save-profile").disabled = answered < questions.length;
    status($("#quiz-status"), `${answered} of ${questions.length} answered`);
  });
}

function answers() {
  const out = {};
  for (const question of questions) {
    const picked = document.querySelector(`input[name="${question.id}"]:checked`);
    if (!picked) return null;
    out[question.id] = Number(picked.value);
  }
  return out;
}

/* ------------------------------------------------------------------ profile */
function saveProfile(profile) {
  try {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  } catch {
    /* private mode: the profile simply is not remembered */
  }
}

function loadProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/* The upload id lives in sessionStorage: a refresh should not cost you the
   material you just spent a minute uploading, but closing the tab should. */
function saveUpload(result) {
  try {
    sessionStorage.setItem(UPLOAD_KEY, JSON.stringify(result));
  } catch {
    /* private mode: the upload simply is not remembered */
  }
}

function loadUpload() {
  try {
    const raw = sessionStorage.getItem(UPLOAD_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function forgetUpload() {
  state.upload = null;
  try {
    sessionStorage.removeItem(UPLOAD_KEY);
  } catch {
    /* nothing to clean */
  }
  $("#generate-btn").disabled = true;
}

function renderProfile() {
  const profile = state.profile;
  if (!profile) return;

  const ranked = Object.entries(profile.scores).sort((a, b) => b[1] - a[1]);
  const rows = ranked
    .map(
      ([key, value], index) => `
      <div class="bar ${index === 0 ? "top" : ""}">
        <span class="name">${escapeHtml(profile.label && key === profile.dominant ? profile.label : labelOf(key))}</span>
        <span class="track"><span class="fill" style="width:${Math.round(value * 100)}%"></span></span>
        <span class="val">${value.toFixed(2)}</span>
      </div>`
    )
    .join("");

  $("#profile-card").innerHTML = `
    <h3>${
      profile.tied
        ? "No single preference stands out"
        : `${escapeHtml(profile.label || labelOf(profile.dominant))} is your strongest preference`
    }</h3>
    <p class="lede" style="margin:4px 0 0">A starting hypothesis the reports are tuned to, not a diagnosis —
      matching teaching to a learning style has weak evidence behind it.
      ${profile.tied ? "All four shapes are equally reasonable starting points." : ""}</p>
    <div class="bars">${rows}</div>`;

  show($("#quiz-saved"), true);
  show($("#quiz-form"), false);
  unlock($("#step-report"));
  if (!profile.tied) {
    selectStyle(profile.dominant);
  } else if (!$("#styles .style-card.on")) {
    $("#generate-btn").disabled = true;
  }
}

function labelOf(key) {
  return { V: "Visual", A: "Aural", R: "Read / write", K: "Kinesthetic" }[key] || key;
}

/* ------------------------------------------------------------------- upload */
function currentFiles() {
  return $$("#file-input").length ? [...$("#file-input").files] : [];
}

function renderFileList() {
  const files = currentFiles();
  const list = $("#file-list");
  list.innerHTML = files
    .map(
      (file) =>
        `<li><span>${escapeHtml(file.name)}</span><span class="size">${bytes(file.size)}</span></li>`
    )
    .join("");
  $("#read-btn").disabled = files.length === 0 && !$("#notes").value.trim();
}

async function readMaterial() {
  const files = currentFiles();
  const notes = $("#notes").value.trim();
  if (!files.length && !notes) return;

  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  if (notes) form.append("notes", notes);

  const button = $("#read-btn");
  button.disabled = true;
  status($("#upload-status"), "Reading…");

  try {
    const result = await api("/api/extract", form, true);
    state.upload = result;
    saveUpload(result);
    renderUpload(result);
    status($("#upload-status"), result.usable ? "Read." : "Nothing usable was found.", result.usable ? "good" : "bad");
    unlock($("#step-quiz"));
    if (state.profile) {
      unlock($("#step-report"));
      if (state.style) selectStyle(state.style);
    }
  } catch (error) {
    status($("#upload-status"), error.message, "bad");
  } finally {
    button.disabled = false;
  }
}

function renderUpload(result) {
  const card = $("#upload-result");
  const outline = result.outline.slice(0, 8);
  const rest = result.outline.length - outline.length;
  card.innerHTML = `
    <h3>${escapeHtml(result.title)}</h3>
    <ul class="chips">
      <li>${result.files.length} file${result.files.length === 1 ? "" : "s"}</li>
      <li>${result.sections} sections</li>
      <li>${result.chars.toLocaleString()} characters</li>
      ${result.figures ? `<li>${result.figures} image${result.figures === 1 ? "" : "s"}</li>` : ""}
    </ul>
    <ol class="outline">
      ${outline.map((heading) => `<li>${escapeHtml(heading)}</li>`).join("")}
      ${rest > 0 ? `<li class="more">and ${rest} more…</li>` : ""}
    </ol>
    ${
      result.warnings.length
        ? `<div class="warnings"><strong>Worth knowing</strong><ul>${result.warnings
            .map((warning) => `<li>${escapeHtml(warning)}</li>`)
            .join("")}</ul></div>`
        : ""
    }`;
  show(card, true);
}

/* ------------------------------------------------------------------- styles */
function selectStyle(style) {
  state.style = style;
  $$(".style-card").forEach((card) => {
    card.classList.toggle("on", card.dataset.style === style);
  });
  $("#generate-btn").disabled = !state.upload || state.generating;
}

function markMatch() {
  const profile = state.profile;
  const dominant = profile && !profile.tied ? profile.dominant : null;
  $$(".style-card").forEach((card) => {
    show(card.querySelector(".match"), card.dataset.style === dominant);
  });
}

/* ------------------------------------------------------------------- report */
async function generate() {
  if (!state.upload || !state.style || state.generating) return;
  state.generating = true;
  state.report = null;
  state.svgs = {};
  $$("#styles .style-card").forEach((card) => (card.disabled = true));
  $("#generate-btn").disabled = true;
  $("#download-btn").disabled = true;

  const label = $("#generate-status");
  const started = Date.now();
  status(label, "Reading the material one part at a time… 0s");

  const ticker = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    status(label, `Reading the material one part at a time… ${seconds}s`);
  }, 1000);

  try {
    const result = await api("/api/report", {
      upload_id: state.upload.upload_id,
      profile: state.profile,
      style: state.style,
    });
    clearInterval(ticker);
    state.report = result.report;
    state.html = result.html;
    const parts = result.report?.coverage;
    const covered = parts ? ` ${parts.covered}/${parts.parts} parts covered.` : "";
    status(label, `Done in ${Math.round((Date.now() - started) / 1000)}s.${covered}`, "good");
    await showPreview(result.html, result.report);
  } catch (error) {
    clearInterval(ticker);
    if (/expired/i.test(error.message)) {
      forgetUpload();
      status(label, `${error.message}`, "bad");
    } else {
      status(label, error.message, "bad");
    }
  } finally {
    state.generating = false;
    $$("#styles .style-card").forEach((card) => (card.disabled = false));
    $("#generate-btn").disabled = false;
  }
}

async function showPreview(html, report) {
  const frame = $("#preview");
  const container = $("#preview-frame");
  show(container, true);

  const loaded = new Promise((resolve) => {
    frame.onload = () => resolve();
  });
  frame.srcdoc = html;
  await loaded;

  await renderDiagrams(frame);

  const synthesis = report.synthesis || {};
  const coverage = report.coverage || {};
  const meta = [
    `<span><b>${report.sections.length}</b> sections</span>`,
    `<span><b>${report.diagrams.length}</b> diagram${report.diagrams.length === 1 ? "" : "s"}</span>`,
    coverage.parts
      ? `<span>covers <b>${coverage.covered}/${coverage.parts}</b> parts</span>`
      : "",
    `<span><b>${Math.round((synthesis.ratio || 0) * 100)}%</b> source overlap</span>`,
    report.attempts > 1 ? `<span>written <b>${report.attempts}</b> times</span>` : "",
  ]
    .filter(Boolean)
    .join("");
  $("#report-meta").innerHTML = meta;
  show($("#report-meta"), true);

  // Two different things: notes are process facts ("the first draft was
  // rewritten"), warnings are properties of the document you are about to
  // download. The page shows both; the export carries only the second.
  const notices = [...(report.notes || []), ...(report.warnings || [])];
  $("#report-warnings").innerHTML = notices.length
    ? `<strong>Worth knowing</strong><ul>${notices
        .map((notice) => `<li>${escapeHtml(notice)}</li>`)
        .join("")}</ul>`
    : "";
  show($("#report-warnings"), notices.length > 0);

  $("#download-btn").disabled = false;
}

async function renderDiagrams(frame) {
  const doc = frame.contentDocument;
  if (!doc) return;
  const nodes = [...doc.querySelectorAll(".mermaid")];
  state.svgs = {};
  if (!nodes.length) return;

  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "strict",
    // A parse failure must not paint Mermaid's red "Syntax error" banner
    // over the page: the block gets a quiet note of our own instead.
    suppressErrorRendering: true,
    flowchart: { htmlLabels: false, curve: "basis", padding: 14, useMaxWidth: true },
    themeVariables: {
      fontFamily: '"DejaVu Sans", "Segoe UI", Helvetica, Arial, sans-serif',
      fontSize: "15px",
      primaryColor: "#f4f7fa",
      primaryBorderColor: "#7f8fa6",
      primaryTextColor: "#1c1f24",
      lineColor: "#5a616b",
      tertiaryColor: "#ffffff",
    },
  });

  for (const node of nodes) {
    const code = node.textContent;
    const key = node.dataset.diagram;
    try {
      const { svg } = await mermaid.render(`vark-diagram-${key}-${Date.now()}`, code);
      node.innerHTML = svg;
      state.svgs[key] = svg;
    } catch (error) {
      console.warn("A diagram could not be drawn:", error);
      node.classList.add("failed");
      node.textContent =
        "This diagram could not be drawn. The Markdown download still contains its source.";
    }
  }
}

/* ----------------------------------------------------------------- download */
async function download() {
  if (!state.report) return;
  const button = $("#download-btn");
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Preparing…";

  try {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        upload_id: state.upload.upload_id,
        profile: state.profile,
        style: state.style,
        format: state.format,
        svgs: state.svgs,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `${response.status} ${response.statusText}`);
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = match ? match[1] : `study-report.${state.format}`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  } catch (error) {
    status($("#generate-status"), `Download failed: ${error.message}`, "bad");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

/* -------------------------------------------------------------------- wiring */
function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (character) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character];
  });
}

function wire() {
  const drop = $("#drop");
  const input = $("#file-input");

  drop.addEventListener("click", (event) => {
    if (event.target.closest("li")) return;
    input.click();
  });
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", renderFileList);

  ["dragenter", "dragover"].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.remove("over");
    })
  );
  drop.addEventListener("drop", (event) => {
    const dropped = [...(event.dataTransfer?.files || [])];
    if (!dropped.length) return;
    const transfer = new DataTransfer();
    [...input.files, ...dropped].forEach((file) => transfer.items.add(file));
    input.files = transfer.files;
    renderFileList();
  });

  $("#notes").addEventListener("input", renderFileList);
  $("#read-btn").addEventListener("click", readMaterial);

  $("#save-profile").addEventListener("click", async () => {
    const responses = answers();
    if (!responses) return;
    try {
      const result = await api("/api/score", { responses });
      state.profile = result.profile;
      saveProfile(result.profile);
      renderProfile();
      markMatch();
      status($("#quiz-status"), "", "");
    } catch (error) {
      status($("#quiz-status"), error.message, "bad");
    }
  });

  $("#retake-btn").addEventListener("click", () => {
    show($("#quiz-saved"), false);
    show($("#quiz-form"), true);
    $("#save-profile").disabled = true;
    status($("#quiz-status"), "", "");
  });

  $$(".style-card").forEach((card) =>
    card.addEventListener("click", () => selectStyle(card.dataset.style))
  );

  $("#generate-btn").addEventListener("click", generate);
  $("#download-btn").addEventListener("click", download);

  $$("#format-picker button").forEach((button) =>
    button.addEventListener("click", () => {
      state.format = button.dataset.format;
      $$("#format-picker button").forEach((other) => other.classList.toggle("on", other === button));
    })
  );
}

async function init() {
  wire();
  loadHealth();

  try {
    await loadQuiz();
  } catch (error) {
    status($("#quiz-status"), "Could not load the questionnaire.", "bad");
    console.error(error);
  }

  const saved = loadProfile();
  if (saved && saved.scores) {
    state.profile = saved;
    renderProfile();
    markMatch();
  } else {
    show($("#quiz-saved"), false);
    show($("#quiz-form"), true);
  }

  const upload = loadUpload();
  if (upload && upload.upload_id && upload.usable) {
    state.upload = upload;
    renderUpload(upload);
    status($("#upload-status"), "Read (restored from this tab).", "good");
    unlock($("#step-quiz"));
    if (state.profile) unlock($("#step-report"));
  }
}

document.addEventListener("DOMContentLoaded", init);
