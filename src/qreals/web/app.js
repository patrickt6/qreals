
const OPS = __OPS__;
const GROUPS = __GROUPS__;
// ---- persistent store (localStorage) --------------------------------
// One JSON document under APP_KEY. In Gate 0 there is a single implicit
// profile so saved results keep working unchanged; Gate 1 makes it
// profile-aware. Legacy qreals.saved.v1 is migrated once, read-only.
const APP_KEY = "qreals.app.v1";
const LEGACY_SAVED_KEY = "qreals.saved.v1";

// state shape: { version, activeProfileId, profiles: { [id]: Profile } }
// Profile: { id, name, createdAt, theme, fontSize, saved:[], compare:[], lastSession:null }
function _uid(){ return "p" + Math.random().toString(36).slice(2, 9); }

const store = {
  _read(){
    try { return JSON.parse(localStorage.getItem(APP_KEY)) || null; }
    catch(e){ return null; }
  },
  _write(state){ localStorage.setItem(APP_KEY, JSON.stringify(state)); },
  _default(){ return { version: 1, activeProfileId: null, profiles: {} }; },
  state(){
    let s = this._read();
    if (!s){
      s = this._default();
      try {
        const legacy = JSON.parse(localStorage.getItem(LEGACY_SAVED_KEY));
        if (Array.isArray(legacy) && legacy.length){
          const id = _uid();
          s.profiles[id] = { id, name: "My work", createdAt: new Date().toISOString(),
            theme: "light", fontSize: "m", saved: legacy, compare: [], lastSession: null };
          s.activeProfileId = id;
        }
      } catch(e){ /* ignore */ }
      this._write(s);
    }
    return s;
  },
  getSaved(){ const p = this.active(); return p ? (p.saved || []) : []; },
  setSaved(list){ this.updateActive({ saved: list }); },
  profiles(){ return Object.values(this.state().profiles); },
  active(){
    const s = this.state();
    return (s.activeProfileId && s.profiles[s.activeProfileId]) || null;
  },
  createProfile({ name, theme = "light", fontSize = "m", saved = [] }){
    const s = this.state();
    const id = _uid();
    s.profiles[id] = { id, name, createdAt: new Date().toISOString(),
      theme, fontSize, saved, compare: [], lastSession: null };
    s.activeProfileId = id; this._write(s); return id;
  },
  setActive(id){ const s = this.state(); if (s.profiles[id]){ s.activeProfileId = id; this._write(s); } },
  clearActive(){ const s = this.state(); s.activeProfileId = null; this._write(s); },
  deleteProfile(id){ const s = this.state(); delete s.profiles[id];
    if (s.activeProfileId === id) s.activeProfileId = null; this._write(s); },
  updateActive(patch){ const s = this.state(); const p = s.profiles[s.activeProfileId];
    if (p){ Object.assign(p, patch); this._write(s); } },
  getCompare(){ const p = this.active(); return p ? (p.compare || []) : []; },
  setCompare(list){ this.updateActive({ compare: list }); },
  addCompare(item){
    const list = this.getCompare().slice();
    list.push(Object.assign({ id: "c" + Math.random().toString(36).slice(2, 9), note: "" }, item));
    this.setCompare(list); return list[list.length - 1].id;
  },
  removeCompare(id){ this.setCompare(this.getCompare().filter((c) => c.id !== id)); },
  updateCompare(id, patch){
    this.setCompare(this.getCompare().map((c) => c.id === id ? Object.assign({}, c, patch) : c));
  },
  pushHistory(entry){ const s = this.state(); const p = s.profiles[s.activeProfileId]; if (!p) return;
    p.history = (p.history || []);
    p.history.unshift(Object.assign({ when: new Date().toISOString() }, entry));
    if (p.history.length > 200) p.history.length = 200; this._write(s); },
  getHistory(){ const p = this.active(); return p ? (p.history || []) : []; },
  clearHistory(){ this.updateActive({ history: [] }); },
  updateSavedTags(index, tags){ const list = this.getSaved().slice(); if (list[index]){ list[index] = Object.assign({}, list[index], { tags }); this.setSaved(list); } },
  updateSavedNote(index, note){ const list = this.getSaved().slice(); if (list[index]){ list[index] = Object.assign({}, list[index], { note }); this.setSaved(list); } },
};

const $ = (id) => document.getElementById(id);
const home = $("home"), opView = $("opView"), savedView = $("savedView"),
      workspaceView = $("workspaceView");

let currentOp = null;
let lastResult = null;       // {op, input, args, latex, text, rows}
let currentControls = {};    // field name -> { kind, get() }

// ---- MathLive (optional; falls back to plain inputs if the CDN is down) --
let MathfieldElement = null;
try {
  const mod = await import("https://cdn.jsdelivr.net/npm/mathlive@0.109.2/mathlive.min.mjs");
  MathfieldElement = mod.MathfieldElement;
  const MLBASE = "https://cdn.jsdelivr.net/npm/mathlive@0.109.2";
  MathfieldElement.fontsDirectory = MLBASE + "/fonts";
  MathfieldElement.soundsDirectory = MLBASE + "/sounds";
} catch(e){ MathfieldElement = null; }

// ---- LaTeX (from the math editor) -> engine syntax ----------------------
function _readGroup(s, i){
  if (s[i] !== "{") return { body: s[i] || "", end: i + 1 };
  let depth = 0, end = i;
  for (; end < s.length; end++){
    if (s[end] === "{") depth++;
    else if (s[end] === "}"){ depth--; if (depth === 0){ end++; break; } }
  }
  return { body: s.slice(i + 1, end - 1), end };
}
function _expandTex(s){
  s = s.replace(/\\left|\\right/g, "")
       .replace(/\\!|\\,|\\;|\\:|\\ /g, "")
       .replace(/\\cdot|\\times/g, "*")
       .replace(/\\pi/g, "pi")
       .replace(/\\phi|\\varphi/g, "((1+sqrt(5))/2)");
  let out = "", i = 0;
  while (i < s.length){
    if (s.startsWith("\\sqrt", i)){
      i += 5;
      const g = _readGroup(s, i);
      out += "sqrt(" + _expandTex(g.body) + ")";
      i = g.end;
    } else if (s.startsWith("\\frac", i)){
      i += 5;
      const num = _readGroup(s, i); i = num.end;
      const den = _readGroup(s, i); i = den.end;
      out += "(" + _expandTex(num.body) + ")/(" + _expandTex(den.body) + ")";
    } else if (s[i] === "^"){
      i += 1;
      const g = _readGroup(s, i);
      out += "**(" + _expandTex(g.body) + ")";
      i = g.end;
    } else if (s[i] === "{" || s[i] === "}"){
      i += 1;
    } else { out += s[i]; i += 1; }
  }
  return out;
}
function latexToEngine(tex){
  if (!tex) return "";
  return _expandTex(tex).replace(/\s+/g, "");
}

// ---- MathJax helpers ----------------------------------------------------
function typeset(el){
  if (window.MathJax && window.MathJax.typesetPromise){
    window.MathJax.typesetClear && window.MathJax.typesetClear([el]);
    return window.MathJax.typesetPromise([el]);
  }
  return Promise.resolve();
}

// ---- escaping -----------------------------------------------------------
function esc(s){
  return String(s).replace(/[&<>"']/g, (c) => (
    {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]
  ));
}

// ---- toast --------------------------------------------------------------
let toastTimer = null;
function toast(msg){
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
}

// ---- reproducible sharing (Gate 7) -------------------------------------
async function _deflate(str){
  const cs = new CompressionStream("deflate");
  const w = cs.writable.getWriter(); w.write(new TextEncoder().encode(str)); w.close();
  const buf = await new Response(cs.readable).arrayBuffer();
  return btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");
}
async function _inflate(b64u){
  const b64 = b64u.replace(/-/g,"+").replace(/_/g,"/");
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ds = new DecompressionStream("deflate");
  const w = ds.writable.getWriter(); w.write(bytes); w.close();
  return new TextDecoder().decode(await new Response(ds.readable).arrayBuffer());
}
// What the share menu acts on: null = the active profile's compare list (the
// tray/workspace), or an explicit item list when sharing a single result / a
// saved or history entry.
let _shareItems = null;
function _bundleFromCompare(title){
  const items = _shareItems
    ? _shareItems.map((c) => ({ op: c.op, input: c.input, args: c.args || {}, note: c.note || "" }))
    : store.getCompare().map((c) => ({ op: c.op, input: c.input, args: c.args || {}, note: c.note || "" }));
  const dflt = (_shareItems && _shareItems.length === 1)
    ? "[" + (_shareItems[0].input != null ? _shareItems[0].input : "") + "]_q"
    : "qreals comparison";
  return { v: 1, title: title || dflt, items };
}
// Open the share menu over `btn` for an explicit set of items (a single result,
// a saved entry, a history entry). Pass null to share the compare list.
function openShareMenuFor(items, btn){ _shareItems = items; toggleShareMenu(btn); }
async function makeShareLink(bundle){
  const enc = await _deflate(JSON.stringify(bundle));
  if (enc.length > 1800) toast("Large bundle — prefer Download .qreals for email");
  return location.origin + location.pathname + "#s=" + enc;
}
function _download(name, text, type){ const b = new Blob([text], {type:type||"text/plain"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = name; a.click(); URL.revokeObjectURL(a.href); }
function exportQreals(){ _download("comparison.qreals", JSON.stringify(_bundleFromCompare()), "application/json"); }

async function exportTex(){
  const out = await fetch("/export",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(_bundleFromCompare())}).then(r=>r.json());
  _download("qreals.tex", out.tex, "application/x-tex");
}
async function openInOverleaf(){
  const out = await fetch("/export",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(_bundleFromCompare())}).then(r=>r.json());
  const f = document.createElement("form"); f.method="POST"; f.action="https://www.overleaf.com/docs"; f.target="_blank";
  const i = document.createElement("input"); i.type="hidden"; i.name="encoded_snip"; i.value=out.tex; f.appendChild(i);
  document.body.appendChild(f); f.submit(); f.remove();
}
async function exportPdf(){
  // Use the same item source as the other exports: _bundleFromCompare honours
  // _shareItems (a single shared result) and otherwise falls back to the
  // compare list. Reading store.getCompare() directly here meant a single
  // computed result, not yet added to the tray, printed as just its title.
  const bundle = _bundleFromCompare();
  const root = $("printRoot");
  if (!bundle.items.length){ toast("Nothing to export"); return; }
  const parts = await Promise.all(bundle.items.map(async (it) => {
    const res = await computeResult(it.op, it.input, it.args);
    const rows = (res && res.rows && res.rows.length)
      ? '<table class="print-rows">' + res.rows.map((r) =>
          '<tr><td>' + esc(String(r[0])) + '</td><td>' + esc(String(r[1])) + '</td></tr>').join("") + '</table>'
      : "";
    const m = res && res.meta;
    const isPlot = m && (m.roots || m.eigen || m.frieze || m.plot3d || m.points || m.surface);
    const plotNote = isPlot
      ? '<p class="print-note">This result includes an interactive plot, which is not captured in the PDF.</p>'
      : "";
    return '<section class="print-item"><h3>' + esc(tileLabel(it)) + '</h3>' +
      (res ? '\\[' + res.latex + '\\]' : '<i>could not compute</i>') +
      rows + plotNote +
      (it.note ? '<p class="print-note">' + esc(it.note) + '</p>' : '') + '</section>';
  }));
  root.innerHTML = '<h1>' + esc(bundle.title) + '</h1>' + parts.join("");
  await typeset(root);
  setTimeout(() => window.print(), 50);
}
function citeBibtex(){
  const items = store.getCompare();
  const entries = items.map((c, i) => {
    const slug = (c.op + "_" + String(c.input).replace(/[^a-zA-Z0-9]/g, "")).slice(0, 32) || ("item" + i);
    return "@misc{qreals_" + slug + ",\n  title = {[" + c.input + "]_q via qreals},\n" +
      "  author = {qreals engine},\n  year = {2026},\n  note = {op=" + c.op +
      (c.note ? "; " + c.note : "") + "}\n}";
  });
  const text = entries.join("\n\n") || "% (no items in compare)";
  navigator.clipboard.writeText(text).then(() => toast("BibTeX copied"));
}
async function exportHtml(){
  const bundle = _bundleFromCompare();
  const link = await makeShareLink(bundle);
  const secs = await Promise.all(bundle.items.map(async (it) => {
    const res = await computeResult(it.op, it.input, it.args);
    const rows = (res && res.rows && res.rows.length)
      ? '<table>' + res.rows.map((r) => '<tr><td>' + esc(String(r[0])) + '</td><td>' + esc(String(r[1])) + '</td></tr>').join("") + '</table>'
      : "";
    const m = res && res.meta;
    const isPlot = m && (m.roots || m.eigen || m.frieze || m.plot3d || m.points || m.surface);
    const plotNote = isPlot
      ? '<p class="note">This result includes an interactive plot — open the qreals link at the bottom to view it live.</p>'
      : "";
    return '<section><h3>' + esc(tileLabel(it)) + '</h3>' +
      (res ? '\\[' + res.latex + '\\]' : '<em>could not compute</em>') +
      rows + plotNote +
      (it.note ? '<p class="usernote">' + esc(it.note) + '</p>' : "") + '</section>';
  }));
  // NOTE: the MathJax closing tag below is written with a backslash escape so
  // this string (which itself lives inside the page's module script) does not
  // prematurely close it; at runtime the escape collapses to a valid close tag.
  const html = '<!doctype html><html><head><meta charset="utf-8"><title>' + esc(bundle.title) + '</title>' +
    '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"><\/script>' +
    '<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:760px;margin:40px auto;' +
    'padding:0 16px;color:#1f2328;line-height:1.6}h1{font-size:1.5rem}h3{margin-top:28px}' +
    'table{border-collapse:collapse;font-size:.9rem;margin:6px 0}td{border-bottom:1px solid #eee;padding:3px 12px}' +
    '.note{color:#6b757f;font-size:.85rem}.usernote{font-style:italic;color:#475059}a{color:#2456a6}</style></head>' +
    '<body><h1>' + esc(bundle.title) + '</h1>' + secs.join("") +
    '<p class="note">Recomputed locally by the qreals engine — not remembered values. ' +
    'Reproduce live: <a href="' + esc(link) + '">open in qreals</a>.</p></body></html>';
  _download("qreals-comparison.html", html, "text/html");
}

async function emailShare(){
  const bundle = _bundleFromCompare();
  const link = await makeShareLink(bundle);
  const subj = "[qreals] " + bundle.title;
  const items = (_shareItems || store.getCompare());
  const inside = items.map((c) =>
    "  [" + c.input + "]_q" + (c.args && c.args.y ? " vs [" + c.args.y + "]_q" : "") +
    (c.note ? "   (" + c.note + ")" : "")).join("\n");
  const body = [
    "I'm sharing a qreals computation with you. qreals is a tiny app that recomputes",
    "everything locally from the inputs, so nothing here is taken on trust.",
    "",
    "To open it:",
    "",
    "1. If you don't have qreals, install and start it:",
    "      pip install qreals",
    "      qreals serve            (this opens qreals in your browser)",
    "",
    "2. Load this share. Either:",
    "   - click the link below (works if your qreals is on the same port), OR",
    "   - in qreals, click \"Profile\", then \"Open shared link\", and paste the link.",
    "",
    "   " + link,
    "",
    "What's inside:",
    inside,
    "",
    "Sent from qreals.",
  ].join("\n");
  location.href = "mailto:?subject=" + encodeURIComponent(subj) + "&body=" + encodeURIComponent(body);
}

// Show the receive preview for an encoded share payload, then reproduce on
// confirm. Port-independent: only the payload (the part after #s=) matters, so a
// link made on one port reproduces fine in a qreals running on any other port.
async function receiveSharePayload(enc, fromUrl){
  let bundle; try { bundle = JSON.parse(await _inflate(enc)); } catch(e){ toast("Couldn't read that share link"); return; }
  if (!bundle || !Array.isArray(bundle.items)){ toast("That doesn't look like a qreals share link"); return; }
  $("receiveList").innerHTML = bundle.items.map((it) =>
    '<div class="receive-row"><b>' + esc(OPS[it.op] ? OPS[it.op].name : it.op) + '</b>: ' +
    esc(it.input) + (it.args && it.args.y ? ", y=" + esc(it.args.y) : "") +
    (it.note ? ' <span class="meta">— ' + esc(it.note) + '</span>' : '') + '</div>').join("");
  const b = $("receiveBackdrop"); b.classList.remove("hidden"); requestAnimationFrame(() => b.classList.add("show"));
  const close = () => { b.classList.remove("show"); setTimeout(() => b.classList.add("hidden"), 280); if (fromUrl) history.replaceState(null, "", location.pathname); };
  $("receiveReproduce").onclick = () => {
    bundle.items.forEach((it) => store.addCompare({ op: it.op, input: it.input, args: it.args || {}, note: it.note || "" }));
    close(); renderTray(); goWorkspace();
  };
  $("receiveCancel").onclick = close;
}
// Pull the share code out of a full link (any host/port) or a bare fragment/code.
function _extractShareCode(text){
  if (!text) return null;
  const s = String(text).trim();
  const m = s.match(/[#?&]s=([A-Za-z0-9\-_]+)/);
  if (m) return m[1];
  if (/^[A-Za-z0-9\-_]{12,}$/.test(s)) return s;   // they pasted just the code
  return null;
}
// Manual entry point so a share link works regardless of the port it was made
// on: the recipient pastes it into their own running qreals.
function openSharedLink(){
  const t = prompt("Paste a qreals share link or code (works no matter what port it was made on):");
  if (t === null) return;
  const code = _extractShareCode(t);
  if (!code){ toast("No share code found in that text"); return; }
  receiveSharePayload(code, false);
}
async function maybeReceiveShare(){
  const m = location.hash.match(/#s=([^&]+)/); if (!m) return;
  receiveSharePayload(m[1], true);
}

function toggleShareMenu(btn){
  const m = $("shareMenu");
  if (m.classList.contains("hidden")){
    // Anchor the menu directly above the Share button that opened it, so it
    // appears where the user clicked (not stranded in a far corner).
    if (btn){
      const r = btn.getBoundingClientRect();
      m.style.left = Math.round(Math.min(r.left, window.innerWidth - 200)) + "px";
      m.style.right = "auto";
      m.style.bottom = Math.round(window.innerHeight - r.top + 8) + "px";
    }
    m.classList.remove("hidden");
  } else {
    m.classList.add("hidden");
  }
}

// ---- home screen --------------------------------------------------------
function filterCards(q){
  document.querySelectorAll(".op-card").forEach((c) => {
    const hay = (c.textContent || "").toLowerCase();
    c.classList.toggle("hidden", q && !hay.includes(q));
  });
  document.querySelectorAll(".group-title").forEach((h) => {
    const grid = h.nextElementSibling; // the .card-grid
    const any = grid && Array.from(grid.querySelectorAll(".op-card")).some((c) => !c.classList.contains("hidden"));
    h.classList.toggle("hidden", !!q && !any);
    if (grid) grid.classList.toggle("hidden", !!q && !any);
  });
}

function renderHome(){
  const byGroup = {};
  for (const [key, meta] of Object.entries(OPS)){
    (byGroup[meta.group] = byGroup[meta.group] || []).push([key, meta]);
  }
  let html = "";
  for (const g of GROUPS){
    const ops = byGroup[g];
    if (!ops) continue;
    html += '<h2 class="group-title">' + esc(g) + '</h2><div class="card-grid">';
    for (const [key, meta] of ops){
      const sym = meta.tex ? ('\\(' + meta.tex + '\\)') : esc(meta.symbol);
      html += '<button class="op-card" type="button" data-op="' + esc(key) + '">' +
        '<p class="oc-name">' + esc(meta.name) + '</p>' +
        '<span class="oc-sym">' + sym + '</span>' +
        '<p class="oc-blurb">' + esc(meta.blurb) + '</p>' +
      '</button>';
    }
    html += '</div>';
  }
  $("cards").innerHTML = html;
  document.querySelectorAll(".op-card").forEach((c) => {
    c.addEventListener("click", () => openOp(c.dataset.op));
  });
  typeset($("cards"));
  // rebuild group-jump chips (safe to repeat: replaces innerHTML each call)
  $("groupJump").innerHTML = GROUPS.filter((g) => byGroup[g]).map((g) =>
    '<button class="chip" type="button" data-jump="' + esc(g) + '">' + esc(g) + '</button>').join("");
  $("groupJump").querySelectorAll("[data-jump]").forEach((b) => b.addEventListener("click", () => {
    const h = Array.from(document.querySelectorAll(".group-title")).find((t) => t.textContent === b.dataset.jump);
    if (h) h.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  renderSavedInto($("savedHome"), false);
}

// ---- per-operation view -------------------------------------------------
function controlKind(op, f){
  if (f.type === "select") return "select";
  if (f.type === "int") return "int";
  if (OPS[op].input_kind === "sequence") return "text";
  if (f.name === "input" || f.name === "y") return "math";
  return "text";
}
function getVal(name){
  const c = currentControls[name];
  return c ? c.get() : "";
}
function inputValue(){ return getVal("input"); }
function fieldArgs(){
  const args = {};
  for (const f of OPS[currentOp].fields){
    if (f.name === "input") continue;
    args[f.name] = getVal(f.name);
  }
  return args;
}

function buildForm(opKey, preset){
  const meta = OPS[opKey];
  currentControls = {};
  let html = '<h2>' + esc(meta.name) + '</h2>' +
    '<span class="psym" id="psym"></span>' +
    '<p class="pblurb">' + esc(meta.blurb) + '</p>';
  for (const f of meta.fields){
    const kind = controlKind(opKey, f);
    const preVal = (preset && preset[f.name] !== undefined) ? String(preset[f.name]) : null;
    html += '<div class="field"><label for="ctl_' + esc(f.name) + '">' +
      esc(f.label) + '</label>';
    if (kind === "select"){
      const val = (preVal !== null) ? preVal : f.example;
      html += '<select id="ctl_' + esc(f.name) + '">';
      for (const c of f.choices){
        const sel = (String(c.value) === String(val)) ? " selected" : "";
        html += '<option value="' + esc(c.value) + '"' + sel + '>' + esc(c.label) + '</option>';
      }
      html += '</select>';
    } else if (kind === "math" && MathfieldElement){
      html += '<span data-mfslot="' + esc(f.name) + '"></span>' +
        '<div class="fallbackrow">' +
          '<input type="text" class="plain-fallback hidden" id="txt_' + esc(f.name) +
            '" spellcheck="false" autocomplete="off">' +
          '<button type="button" class="linkbtn" data-toggle="' + esc(f.name) +
            '">edit as plain text</button>' +
        '</div>';
    } else if (kind === "int"){
      const val = (preVal !== null) ? preVal : f.example;
      html += '<input id="ctl_' + esc(f.name) + '" type="text" inputmode="numeric" value="' +
        esc(val) + '" spellcheck="false" autocomplete="off">';
    } else {
      const val = (preVal !== null) ? preVal : f.example;
      html += '<input id="ctl_' + esc(f.name) + '" type="text" value="' +
        esc(val) + '" spellcheck="false" autocomplete="off">';
    }
    html += '</div>';
  }
  html += '<div class="preview-box"><span class="pv-label">Reads as</span>' +
    '<span class="pv-math" id="previewMath"></span></div>';
  html += '<button class="go" id="goBtn" type="button">Compute</button>';
  $("formPanel").innerHTML = html;

  // the operation's headline symbol, typeset
  const psym = $("psym");
  if (meta.tex){ psym.innerHTML = '\\(' + meta.tex + '\\)'; typeset(psym); }
  else { psym.textContent = meta.symbol; }

  // wire each control and register a value getter
  for (const f of meta.fields){
    const kind = controlKind(opKey, f);
    const preVal = (preset && preset[f.name] !== undefined) ? String(preset[f.name]) : null;
    if (kind === "math" && MathfieldElement){
      const slot = $("formPanel").querySelector('[data-mfslot="' + f.name + '"]');
      const mf = new MathfieldElement();
      mf.setAttribute("aria-label", f.label);
      mf.value = (preVal !== null) ? preVal : (f.tex || f.example);
      slot.appendChild(mf);
      const txt = $("txt_" + f.name);
      txt.value = latexToEngine(mf.value);
      currentControls[f.name] = {
        kind,
        get: () => (txt.classList.contains("hidden")
          ? latexToEngine(mf.value) : txt.value.trim())
      };
      mf.addEventListener("input", () => { txt.value = latexToEngine(mf.value); schedulePreview(); });
      mf.addEventListener("keydown", (e) => {
        if (e.key === "Enter"){ e.preventDefault(); runCompute(); }
      });
      txt.addEventListener("input", schedulePreview);
      txt.addEventListener("keydown", (e) => {
        if (e.key === "Enter"){ e.preventDefault(); runCompute(); }
      });
      const tog = $("formPanel").querySelector('[data-toggle="' + f.name + '"]');
      tog.addEventListener("click", () => {
        if (txt.classList.contains("hidden")){
          txt.value = latexToEngine(mf.value);
          txt.classList.remove("hidden"); mf.classList.add("hidden");
          tog.textContent = "use the math editor"; txt.focus();
        } else {
          mf.classList.remove("hidden"); txt.classList.add("hidden");
          tog.textContent = "edit as plain text"; mf.focus();
        }
      });
    } else {
      const el = $("ctl_" + f.name);
      currentControls[f.name] = { kind, get: () => (el ? el.value : "") };
      if (el){
        el.addEventListener("input", schedulePreview);
        el.addEventListener("change", schedulePreview);
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter"){ e.preventDefault(); runCompute(); }
        });
      }
    }
  }
  $("goBtn").addEventListener("click", runCompute);
}

let previewTimer = null;
function schedulePreview(){
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updatePreview, 180);
}
async function updatePreview(){
  if (!currentOp) return;
  const body = { op: currentOp, input: inputValue(), args: fieldArgs() };
  let latex = "";
  try {
    const res = await fetch("/preview", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    latex = data.latex || "";
  } catch(e){ latex = ""; }
  const el = $("previewMath");
  if (!el) return;
  el.innerHTML = latex ? ("\\(" + latex + "\\)") : '<span style="color:var(--ink-faint)">&mdash;</span>';
  if (latex) typeset(el);
}

function focusInput(){
  const c = currentControls["input"];
  if (!c) return;
  const mf = $("formPanel").querySelector('[data-mfslot="input"] math-field');
  if (mf && !mf.classList.contains("hidden")){ mf.focus(); return; }
  const el = $("ctl_input") || $("txt_input");
  if (el) el.focus();
}

function openOp(opKey, preset, storedResult){
  currentOp = opKey;
  home.classList.add("hidden");
  savedView.classList.add("hidden");
  workspaceView.classList.add("hidden");
  opView.classList.remove("hidden");
  buildForm(opKey, preset);
  if (storedResult){
    showResult(storedResult);
  } else {
    $("result").innerHTML = '<div class="empty">Fill in the inputs and press Compute.</div>';
  }
  window.scrollTo(0, 0);
  renderTray();   // re-show the compare tray after leaving the workspace
  focusInput();
  updatePreview();
}

async function runCompute(){
  if (!currentOp) return;
  const input = inputValue(), args = fieldArgs();
  const btn = $("goBtn");
  if (btn){ btn.disabled = true; btn.textContent = "Computing..."; }
  $("result").innerHTML = '<div class="result-loading"><span class="spinner"></span>Computing…</div>';
  let data;
  try {
    const res = await fetch("/compute", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ op: currentOp, input, args })
    });
    data = await res.json();
  } catch(e){
    data = { error: "could not reach the server: " + e };
  }
  if (btn){ btn.disabled = false; btn.textContent = "Compute"; }
  if (data.error){
    lastResult = null;
    $("result").innerHTML = '<div class="result-error">error: ' + esc(data.error) + '</div>';
    return;
  }
  lastResult = { op: currentOp, input, args, latex: data.latex,
                 text: data.text, rows: data.rows || [], meta: data.meta || null };
  showResult(lastResult);
  if (store.active()) store.updateActive({ lastSession: { op: currentOp, input, args } });
  store.pushHistory({ op: currentOp, input, args });
}

// Show a result in the main per-operation panel.
function showResult(r){
  renderResultInto($("result"), r, { actions: "main" });
}

// Render a complete result (math, data rows, plain text, and any interactive
// plot) into `root`. Every lookup is scoped to `root` via querySelector on
// classes -- never document-global getElementById -- so the identical renderer
// drives the single result panel and any number of pop-out workspace tiles at
// once without their plots or view toggles colliding.
// opts.actions: "main" (Save / Pop out / Copy LaTeX) or "tile" (Copy LaTeX only).
function renderResultInto(root, r, opts){
  opts = opts || {};
  let html = '';
  const hasPlot = r.meta && Array.isArray(r.meta.roots);
  const hasFrieze = r.meta && r.meta.frieze;
  const hasViz = r.meta && r.meta.plot3d;
  const hasEigen = r.meta && r.meta.eigen && Array.isArray(r.meta.eigen.points);
  if (hasViz){
    html += '<div class="viz-plot">' + vizModeButtons(r.meta.plot3d.kind) +
      '<div class="viz-fig"></div>' +
      '<p class="viz-cap"></p>' +
      '</div>';
  }
  if (hasPlot){
    html += '<div class="roots-plot">' +
      '<div class="roots-modes" role="group" aria-label="plot view">' +
        '<button class="mini" type="button" data-rootmode="2d">2D pole-zero</button>' +
        '<button class="mini" type="button" data-rootmode="phase">2D phase portrait</button>' +
        '<button class="mini" type="button" data-rootmode="3d">3D surface |R(q)|</button>' +
        '<button class="mini" type="button" data-rootmode="3drs">3D |R/S| (phase + radius)</button>' +
      '</div>' +
      '<div class="roots-fig"></div>' +
      '<p class="roots-cap"></p>' +
      '</div>';
  }
  if (hasFrieze){
    html += '<div class="frieze-plot">' +
      '<div class="frieze-modes" role="group" aria-label="frieze view">' +
        '<button class="mini" type="button" data-friezemode="poly">q-polynomial</button>' +
        '<button class="mini" type="button" data-friezemode="int">integer (q = 1)</button>' +
      '</div>' +
      '<div class="frieze-fig"></div>' +
      '<div class="frieze-legend">' +
        '<span class="lg"><span class="sw sw-one"></span>border row of 1s</span>' +
        '<span class="lg"><span class="sw sw-quid"></span>cell value at q = 1</span>' +
        '<span class="lg"><span class="sw sw-int"></span>darker = larger value</span>' +
      '</div>' +
      '<p class="frieze-cap"></p>' +
      '</div>';
  }
  if (hasEigen){
    html += '<div class="roots-plot">' +
      '<div class="roots-fig"></div>' +
      '<p class="roots-cap">Eigenvalues of the transfer matrix ' +
      'K = M_x &otimes; M_y on the complex plane. The dominant (Perron) ' +
      'eigenvalue, ringed, carries the eigenvector the closed form is read from.' +
      '</p></div>';
  }
  html += '<div class="rmath">\\[' + r.latex + '\\]</div>';
  if (r.rows && r.rows.length){
    html += '<dl class="rows">';
    for (const [k, v] of r.rows){
      html += '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>';
    }
    html += '</dl>';
  }
  html += '<div class="rtext-wrap"><div class="rtext-head">' +
    '<p class="rlabel">Plain text</p>' +
    '<button class="mini copy-btn" type="button">Copy</button></div>' +
    '<pre class="rtext">' + esc(r.text) + '</pre></div>';
  if (opts.actions === "main"){
    html += '<div class="result-actions">' +
      '<button class="mini primary save-btn" type="button">Save this result</button>' +
      '<button class="mini addcmp-btn" type="button">Add to compare</button>' +
      '<button class="mini copytex-btn" type="button">Copy LaTeX</button>' +
      '<button class="mini share-result-btn" type="button">Share</button>' +
      '</div>';
  } else {
    html += '<div class="result-actions">' +
      '<button class="mini copytex-btn" type="button">Copy LaTeX</button>' +
      '<button class="mini share-result-btn" type="button">Share</button>' +
      '</div>';
  }
  html += '<details class="derivation"><summary>Show the derivation</summary>' +
    '<div class="derivation-body">loading…</div></details>' +
    '<p class="recompute-note">Computed locally by the qreals engine from your input — not a remembered value. ' +
    '<span class="prov-chip">local, unrecorded</span></p>';
  root.innerHTML = html;
  if (hasViz) setupViz(r.meta.plot3d, root);
  if (hasPlot) setupRootsModes(r.meta, root);
  if (hasFrieze) setupFriezeModes(r.meta.frieze, root);
  if (hasEigen) drawEigen(r.meta.eigen, root);
  const mathEl = root.querySelector(".rmath");
  if (mathEl) typeset(mathEl);
  const copyBtn = root.querySelector(".copy-btn");
  if (copyBtn) copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(r.text).then(() => toast("Plain text copied"));
  });
  const copyTexBtn = root.querySelector(".copytex-btn");
  if (copyTexBtn) copyTexBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(r.latex).then(() => toast("LaTeX copied"));
  });
  const saveBtn = root.querySelector(".save-btn");
  if (saveBtn) saveBtn.addEventListener("click", () => saveResult(r));
  const addCmpBtn = root.querySelector(".addcmp-btn");
  if (addCmpBtn) addCmpBtn.addEventListener("click", () => {
    store.addCompare({ op: r.op, input: r.input, args: r.args || {} });
    renderTray();
    toast("Added to compare");
  });
  const shareResBtn = root.querySelector(".share-result-btn");
  if (shareResBtn) shareResBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    openShareMenuFor([{ op: r.op, input: r.input, args: r.args || {}, note: "" }], e.currentTarget);
  });
  const det = root.querySelector(".derivation");
  if (det) det.addEventListener("toggle", async () => {
    if (!det.open || det._loaded) return;
    det._loaded = true;
    const body = det.querySelector(".derivation-body");
    const data = await fetch("/certificate", { method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ op: r.op, input: r.input, args: r.args || {} }) }).then((x) => x.json()).catch(() => null);
    if (!data || data.error){ body.innerHTML = '<span class="cmp-err">No step-by-step derivation for this tool.</span>'; return; }
    let h = "";
    // headline + structure are PROSE (escaped text) — never math-wrapped, or
    // MathJax collapses the words. Only the real math is typeset, as display math.
    if (data.headline) h += '<p class="deriv-headline">' + esc(data.headline) + '</p>';
    if (data.recursionTex){
      h += '<p class="deriv-line">The even-length continued fraction is folded inside out; ' +
        'each partial \\(R_i\\) is read from position \\(i\\):</p>';
      h += '<div class="deriv-math">\\[' + data.recursionTex + '\\]</div>';
      if (data.gaussNote) h += '<p class="deriv-line">with \\(' + data.gaussNote + '\\) the Gauss \\(q\\)-integer.</p>';
    }
    (data.structure || []).forEach((line) => { if (line) h += '<p class="deriv-line">' + esc(line) + '</p>'; });
    if (data.folds && data.folds.length){
      h += '<table class="fold-table"><thead><tr><th>\\(i\\)</th><th>\\(a_i\\)</th><th>partial \\(R_i\\)</th><th>\\(\\deg N_i\\)</th></tr></thead><tbody>';
      data.folds.forEach((f) => { h += '<tr><td>' + f.pos + '</td><td>' + f.a + '</td><td>\\(' + f.ratio + '\\)</td><td>' + f.degree + '</td></tr>'; });
      h += '</tbody></table>';
    }
    if (data.witness) h += '<p class="witness">Numeric check: ' + esc(data.witness) + '</p>';
    if (data.citations && data.citations.length) h += '<p class="refs">Sources: ' + data.citations.map(esc).join(', ') + '</p>';
    body.innerHTML = h; typeset(body);
    // upgrade the provenance chip if the engine recorded provenance
    if (data.provenance_available){ const chip = root.querySelector(".prov-chip"); if (chip) chip.textContent = "provenance: available"; }
  });
}

// ---- roots-of-R(q) plot (Plotly.js, loaded from the CDN) ------------
// The complex coordinates are display-only; the cyclotomic-vs-core colour of
// each root is the exact class sent by factor.py, never a proximity guess.
const ROOT_CYC = "#2456a6";   // accent: cyclotomic, on the unit circle
const ROOT_CORE = "#b3261e";  // err red: non-cyclotomic core, off the circle
function _fmtComplex(re, im){
  const r = (+re).toFixed(4).replace(/\.?0+$/, "") || "0";
  const i = Math.abs(im).toFixed(4).replace(/\.?0+$/, "") || "0";
  if (Math.abs(im) < 5e-5) return r;
  return r + (im < 0 ? " - " : " + ") + i + "i";
}
function _rootsExtent(roots){
  let extent = 1.0;
  for (const r of roots) extent = Math.max(extent, Math.abs(r.re), Math.abs(r.im));
  return extent * 1.15 + 0.06;
}
// Evaluate |R(q)| at q = x + iy via Horner over the ascending integer coeffs.
function _evalAbsR(coeffs, x, y){
  const [re, im] = _evalCplxR(coeffs, x, y);
  return Math.hypot(re, im);
}
// Evaluate the complex value of a polynomial (ascending coeffs) at q = x + iy.
function _evalCplxR(coeffs, x, y){
  let re = 0, im = 0;
  for (let k = coeffs.length - 1; k >= 0; k--){
    const nr = re * x - im * y + coeffs[k];
    const ni = re * y + im * x;
    re = nr; im = ni;
  }
  return [re, im];
}
// arg(R(q)/S(q)) in (-pi, pi]: the Wegert phase, the hue of the portrait.
function _argRatio(rc, sc, x, y){
  const [ar, ai] = _evalCplxR(rc, x, y);
  const [br, bi] = _evalCplxR(sc, x, y);
  let a = Math.atan2(ai, ar) - Math.atan2(bi, br);
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}
// A cyclic HSV colorscale so phase = -pi and +pi share a hue (the colour
// wheel): a zero cycles it one way, a pole the other, and the order is the
// number of cycles. This is the phase grammar borrowed from phase portraits.
function _hsvToRgb(h){
  const s = 0.72, v = 0.96, c = v * s, hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1)), m = v - c;
  let r = 0, g = 0, b = 0;
  if (hp < 1){ r = c; g = x; } else if (hp < 2){ r = x; g = c; }
  else if (hp < 3){ g = c; b = x; } else if (hp < 4){ g = x; b = c; }
  else if (hp < 5){ r = x; b = c; } else { r = c; b = x; }
  return "rgb(" + Math.round((r + m) * 255) + "," + Math.round((g + m) * 255) +
    "," + Math.round((b + m) * 255) + ")";
}
const PHASE_SCALE = (() => {
  const a = []; const M = 24;
  for (let i = 0; i <= M; i++) a.push([i / M, _hsvToRgb((i / M) * 360)]);
  return a;
})();
// The pole nearest the origin: its modulus is the radius of convergence.
function _nearestPole(meta){
  const poles = meta.poles || [];
  if (meta.radius == null) return null;
  for (const p of poles) if (Math.abs(p.mod - meta.radius) < 1e-9) return p;
  return null;
}
// Shared ROC / pole-zero overlay shapes: real axis, unit circle, the radius
// ring, and a radial line out to the nearest pole. Reused by every roots view.
function _rocShapes(meta, W){
  const shapes = [
    { type: "line", x0: -W, y0: 0, x1: W, y1: 0,
      line: { color: "#b8c0c9", width: 1, dash: "dot" } },        // real axis
    { type: "circle", x0: -1, y0: -1, x1: 1, y1: 1,
      line: { color: "#9aa6b2", width: 1.5 } },                   // unit circle
  ];
  if (meta.radius != null){
    const rho = meta.radius;
    shapes.push({ type: "circle", x0: -rho, y0: -rho, x1: rho, y1: rho,
      line: { color: "#1f8a4c", width: 1.6, dash: "dash" } });    // rho ring
    const np = _nearestPole(meta);
    if (np) shapes.push({ type: "line", x0: 0, y0: 0, x1: np.re, y1: np.im,
      line: { color: "#1f8a4c", width: 1.8 } });                  // radial line
  }
  return shapes;
}
function _rocAnnotations(meta){
  if (meta.radius == null) return [];
  const np = _nearestPole(meta);
  const mx = np ? np.re / 2 : meta.radius / 2;
  const my = np ? np.im / 2 : 0;
  return [{ x: mx, y: my, text: "rho = " + meta.radius.toFixed(4),
    showarrow: false, font: { color: "#1f8a4c", size: 12,
      family: "JetBrains Mono,monospace" },
    bgcolor: "rgba(255,255,255,.75)", bordercolor: "#1f8a4c", borderwidth: 1,
    borderpad: 2, yshift: 12 }];
}
// Open-circle zeros (o) and cross poles (x), the signal-processing convention.
function _zeroTrace(roots, kind, label, colour){
  const pts = roots.filter((r) => r.kind === kind);
  return {
    x: pts.map((r) => r.re), y: pts.map((r) => r.im),
    name: label, type: "scatter", mode: "markers", hoverinfo: "text",
    text: pts.map((r) => "zero: q = " + _fmtComplex(r.re, r.im) +
      "<br>|q| = " + Math.hypot(r.re, r.im).toFixed(4) +
      "<br>factor: " + (r.d != null ? ("Phi_" + r.d) : "core")),
    marker: { color: colour, size: 13, symbol: "circle-open",
      line: { width: 2.4 } },
  };
}
function _poleTrace2d(poles){
  return {
    x: poles.map((p) => p.re), y: poles.map((p) => p.im),
    name: "poles (zeros of S)", type: "scatter", mode: "markers",
    hoverinfo: "text",
    text: poles.map((p) => "pole: q = " + _fmtComplex(p.re, p.im) +
      "<br>|q| = " + p.mod.toFixed(4)),
    marker: { color: ROOT_POLE, size: 12, symbol: "x", line: { width: 2 } },
  };
}
// ---- eigenvalues of the transfer matrix K (quad-arith) on the plane ----
function drawEigen(eigen, root){
  const fig = (root || document).querySelector(".roots-fig");
  if (!fig) return;
  const pts = (eigen && eigen.points) || [];
  if (typeof Plotly === "undefined"){
    fig.innerHTML = '<p class="roots-plot-msg">The plotting library could not ' +
      'load (offline?). The eigenvalues are listed in the result below.</p>';
    return;
  }
  let extent = 0.5;
  for (const p of pts) extent = Math.max(extent, Math.abs(p.re) + 0.3, Math.abs(p.im) + 0.3);
  const sub = pts.filter((p) => !p.dominant);
  const dom = pts.filter((p) => p.dominant);
  const traces = [];
  if (sub.length) traces.push({
    x: sub.map((p) => p.re), y: sub.map((p) => p.im),
    mode: "markers", type: "scatter", name: "eigenvalue",
    marker: { size: 11, color: "#3b6fd4", line: { color: "#1f3d80", width: 1 } },
    text: sub.map((p) => p.label),
    hovertemplate: "%{text}<br>(%{x:.5f}, %{y:.5f} i)<extra></extra>",
  });
  if (dom.length) traces.push({
    x: dom.map((p) => p.re), y: dom.map((p) => p.im),
    mode: "markers", type: "scatter", name: "dominant (Perron)",
    marker: { size: 13, color: "#1b9e77",
      line: { color: "#0f5d45", width: 1 }, symbol: "circle" },
    text: dom.map((p) => p.label),
    hovertemplate: "dominant: %{text}<br>(%{x:.5f}, %{y:.5f} i)<extra></extra>",
  });
  // a faint ring around the dominant eigenvalue to draw the eye
  const shapes = dom.map((p) => ({
    type: "circle", xref: "x", yref: "y",
    x0: p.re - extent * 0.06, x1: p.re + extent * 0.06,
    y0: p.im - extent * 0.06, y1: p.im + extent * 0.06,
    line: { color: "#1b9e77", width: 2 }, fillcolor: "rgba(0,0,0,0)",
  }));
  const layout = {
    height: 420,
    margin: { l: 48, r: 18, t: 14, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 13, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.16, yanchor: "top" },
    xaxis: { title: { text: "Re" }, range: [-extent, extent], zeroline: true,
      zerolinecolor: "#c9d2dc", gridcolor: "#eef1f5", constrain: "domain" },
    yaxis: { title: { text: "Im" }, range: [-extent, extent], zeroline: true,
      zerolinecolor: "#c9d2dc", gridcolor: "#eef1f5",
      scaleanchor: "x", scaleratio: 1 },
    shapes: shapes,
  };
  Plotly.newPlot(fig, traces, layout,
    { displaylogo: false, responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d"] });
}

function drawRoots2d(fig, meta){
  if (!fig) return;
  const roots = meta.roots || [];
  if (typeof Plotly === "undefined"){
    fig.innerHTML = '<p class="roots-plot-msg">The plotting library could not ' +
      'load (offline?). The roots and their classification are listed below.</p>';
    return;
  }
  // a square window holding the unit circle, every zero and pole, and the
  // radius ring, with a small margin
  const poles = meta.poles || [];
  let extent = _rootsExtent(roots);
  for (const p of poles) extent = Math.max(extent, Math.abs(p.re) + 0.1, Math.abs(p.im) + 0.1);
  if (meta.radius != null) extent = Math.max(extent, meta.radius + 0.1);

  const traces = [
    _zeroTrace(roots, "cyclotomic", "cyclotomic zero (on unit circle)", ROOT_CYC),
    _zeroTrace(roots, "core", "non-cyclotomic core zero", ROOT_CORE),
  ].filter((t) => t.x.length);
  if (poles.length) traces.push(_poleTrace2d(poles));

  const layout = {
    height: 460,
    margin: { l: 48, r: 18, t: 14, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 13, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.16, yanchor: "top" },
    xaxis: {
      title: { text: "Re" }, range: [-extent, extent], zeroline: false,
      gridcolor: "#eef1f5", constrain: "domain",
    },
    yaxis: {
      title: { text: "Im" }, range: [-extent, extent], zeroline: false,
      gridcolor: "#eef1f5", scaleanchor: "x", scaleratio: 1,
    },
    shapes: _rocShapes(meta, extent),
    annotations: _rocAnnotations(meta),
  };
  const config = { displaylogo: false, responsive: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d"] };
  Plotly.newPlot(fig, traces, layout, config);
}
// The 2D Wegert phase portrait: hue = arg(R/S) over the complex q-square, read
// top-down. Each zero spins the colour wheel one way and each pole the other,
// so singularities are countable at a glance; the pole-zero markers, unit
// circle, radius ring, radial line and real axis sit on top in the ROC grammar.
function drawRootsPhase(fig, meta){
  if (!fig) return;
  const roots = meta.roots || [], poles = meta.poles || [];
  const rc = meta.coeffs || [], sc = meta.s_coeffs || [];
  if (typeof Plotly === "undefined"){
    fig.innerHTML = '<p class="roots-plot-msg">The plotting library could not ' +
      'load (offline?). The roots and their classification are listed below.</p>';
    return;
  }
  if (!rc.length || !sc.length){
    fig.innerHTML = '<p class="roots-plot-msg">No surface data for this input.</p>';
    return;
  }
  let W = 1.5;
  for (const r of roots) W = Math.max(W, Math.abs(r.re), Math.abs(r.im));
  for (const p of poles) W = Math.max(W, Math.abs(p.re), Math.abs(p.im));
  if (meta.radius != null) W = Math.max(W, meta.radius);
  W = W < 1.5 ? 1.5 : W * 1.06;
  const N = 221;
  const axis = [];
  for (let i = 0; i < N; i++) axis.push(-W + (2 * W * i) / (N - 1));
  const Z = [];
  for (let j = 0; j < N; j++){
    const row = [];
    for (let i = 0; i < N; i++) row.push(_argRatio(rc, sc, axis[i], axis[j]));
    Z.push(row);
  }
  const heat = {
    type: "heatmap", x: axis, y: axis, z: Z, zsmooth: "best",
    colorscale: PHASE_SCALE, zmin: -Math.PI, zmax: Math.PI, hoverinfo: "skip",
    colorbar: { title: { text: "arg(R/S)", side: "right" }, thickness: 12,
      tickvals: [-Math.PI, 0, Math.PI], ticktext: ["-pi", "0", "pi"] },
  };
  const traces = [heat,
    _zeroTrace(roots, "cyclotomic", "cyclotomic zero (on unit circle)", ROOT_CYC),
    _zeroTrace(roots, "core", "non-cyclotomic core zero", ROOT_CORE),
  ].filter((t) => t.type === "heatmap" || t.x.length);
  if (poles.length) traces.push(_poleTrace2d(poles));
  const layout = {
    height: 480, margin: { l: 48, r: 18, t: 14, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 13, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.16, yanchor: "top" },
    xaxis: { title: { text: "Re" }, range: [-W, W], constrain: "domain" },
    yaxis: { title: { text: "Im" }, range: [-W, W], scaleanchor: "x", scaleratio: 1 },
    shapes: _rocShapes(meta, W),
    annotations: _rocAnnotations(meta),
  };
  Plotly.newPlot(fig, traces, layout,
    { displaylogo: false, responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d"] });
}

// The 3D view: the modulus surface z = log10|R(q)| over the complex plane. The
// roots are the wells where the surface plunges to zero, the unit circle is a
// reference ring on the floor, and the cyclotomic-vs-core markers (the exact
// factor.py classes) sit in their wells. The surface is display-only; nothing
// here decides which roots are cyclotomic.
function drawRoots3d(fig, meta){
  if (!fig) return;
  const roots = meta.roots || [];
  const coeffs = meta.coeffs || [];
  if (typeof Plotly === "undefined"){
    fig.innerHTML = '<p class="roots-plot-msg">The plotting library could not ' +
      'load (offline?). The roots and their classification are listed below.</p>';
    return;
  }
  if (!coeffs.length){
    fig.innerHTML = '<p class="roots-plot-msg">No surface data for this input.</p>';
    return;
  }
  const extent = _rootsExtent(roots);
  const N = 81;                 // grid resolution; 81x81 is smooth and fast
  const FLOOR = -3;             // clamp deep wells so they stay readable
  const axis = [];
  for (let i = 0; i < N; i++) axis.push(-extent + (2 * extent * i) / (N - 1));
  const Z = [];
  for (let j = 0; j < N; j++){
    const row = [];
    for (let i = 0; i < N; i++){
      const v = _evalAbsR(coeffs, axis[i], axis[j]);
      row.push(Math.max(Math.log10(v + 1e-12), FLOOR));
    }
    Z.push(row);
  }
  const surface = {
    type: "surface", x: axis, y: axis, z: Z,
    colorscale: "Viridis", showscale: false, opacity: 0.92,
    hoverinfo: "skip",
    contours: { z: { show: true, usecolormap: true, width: 1,
      project: { z: true } } },
    lighting: { ambient: 0.65, diffuse: 0.8, specular: 0.15 },
  };
  // unit-circle reference ring on the floor
  const ring = { type: "scatter3d", mode: "lines", name: "unit circle",
    hoverinfo: "skip", line: { color: "#6b757f", width: 4 },
    x: [], y: [], z: [] };
  for (let t = 0; t <= 96; t++){
    const a = (2 * Math.PI * t) / 96;
    ring.x.push(Math.cos(a)); ring.y.push(Math.sin(a)); ring.z.push(FLOOR);
  }
  const markerTrace = (kind, label, colour) => {
    const pts = roots.filter((r) => r.kind === kind);
    return {
      type: "scatter3d", mode: "markers", name: label, hoverinfo: "text",
      x: pts.map((r) => r.re), y: pts.map((r) => r.im),
      z: pts.map(() => FLOOR),
      text: pts.map((r) => {
        const tag = (r.d != null) ? ("Phi_" + r.d) : "core";
        return "q = " + _fmtComplex(r.re, r.im) +
          "<br>|q| = " + Math.hypot(r.re, r.im).toFixed(4) + "<br>factor: " + tag;
      }),
      marker: { color: colour, size: 5, line: { color: "#fff", width: 1 },
        symbol: "circle" },
    };
  };
  const traces = [surface, ring,
    markerTrace("cyclotomic", "cyclotomic (on unit circle)", ROOT_CYC),
    markerTrace("core", "non-cyclotomic core (off circle)", ROOT_CORE),
  ].filter((t) => t.type === "surface" || (t.x && t.x.length));
  const layout = {
    height: 480,
    margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: 0, yanchor: "top" },
    scene: {
      xaxis: { title: { text: "Re" }, gridcolor: "#e3e7ec" },
      yaxis: { title: { text: "Im" }, gridcolor: "#e3e7ec" },
      zaxis: { title: { text: "log|R(q)|" }, gridcolor: "#e3e7ec" },
      aspectmode: "cube",
      camera: { eye: { x: 1.5, y: 1.5, z: 1.1 } },
    },
  };
  const config = { displaylogo: false, responsive: true };
  Plotly.newPlot(fig, traces, layout, config);
}

const ROOT_POLE = "#c98a00";  // gold: poles (roots of S), the spikes
// The unifying 3D view: the height z = log|R(q)/S(q)| over the complex q-square.
// Zeros of R are dimples on the floor, poles of S (roots of the denominator) are
// spikes, and the nearest spike to the origin is the radius of convergence of
// the power series [a/b]_q. One picture ties together factor, roots, and radius.
function drawRoots3dRS(fig, meta){
  if (!fig) return;
  const roots = meta.roots || [], poles = meta.poles || [];
  const rc = meta.coeffs || [], sc = meta.s_coeffs || [];
  if (typeof Plotly === "undefined"){
    fig.innerHTML = '<p class="roots-plot-msg">The plotting library could not ' +
      'load (offline?). The roots and their classification are listed below.</p>';
    return;
  }
  if (!rc.length || !sc.length){
    fig.innerHTML = '<p class="roots-plot-msg">No surface data for this input.</p>';
    return;
  }
  // the complex q-square [-W, W]^2: 1.5 by default, widened to hold any root or
  // pole that sits further out so every spike and dimple stays in frame
  let W = 1.5;
  for (const r of roots) W = Math.max(W, Math.abs(r.re), Math.abs(r.im));
  for (const p of poles) W = Math.max(W, Math.abs(p.re), Math.abs(p.im));
  W = W < 1.5 ? 1.5 : W * 1.06;
  const N = 91, FLOOR = -3, CEIL = 3;
  const axis = [];
  for (let i = 0; i < N; i++) axis.push(-W + (2 * W * i) / (N - 1));
  // height = log|R/S| (clamped), hue = arg(R/S): magnitude as relief, phase as
  // colour, the full analytic landscape rather than magnitude alone.
  const Z = [], P = [];
  for (let j = 0; j < N; j++){
    const zrow = [], prow = [];
    for (let i = 0; i < N; i++){
      const num = _evalAbsR(rc, axis[i], axis[j]);
      const den = _evalAbsR(sc, axis[i], axis[j]);
      let z = Math.log10(num + 1e-12) - Math.log10(den + 1e-12);
      zrow.push(Math.max(FLOOR, Math.min(CEIL, z)));
      prow.push(_argRatio(rc, sc, axis[i], axis[j]));
    }
    Z.push(zrow); P.push(prow);
  }
  const surface = {
    type: "surface", x: axis, y: axis, z: Z,
    surfacecolor: P, colorscale: PHASE_SCALE, cmin: -Math.PI, cmax: Math.PI,
    showscale: true,
    colorbar: { title: { text: "arg(R/S)", side: "right" }, thickness: 12,
      len: 0.6, tickvals: [-Math.PI, 0, Math.PI], ticktext: ["-pi", "0", "pi"] },
    opacity: 1, hoverinfo: "skip",
    lighting: { ambient: 0.72, diffuse: 0.7, specular: 0.1 },
  };
  const ringAt = (rad, z, colour, dash, name) => {
    const t = { type: "scatter3d", mode: "lines", name, hoverinfo: "skip",
      line: { color: colour, width: 4, dash }, x: [], y: [], z: [] };
    for (let k = 0; k <= 96; k++){
      const a = (2 * Math.PI * k) / 96;
      t.x.push(rad * Math.cos(a)); t.y.push(rad * Math.sin(a)); t.z.push(z);
    }
    return t;
  };
  const markerTrace = (kind, label, colour) => {
    const pts = roots.filter((r) => r.kind === kind);
    return {
      type: "scatter3d", mode: "markers", name: label, hoverinfo: "text",
      x: pts.map((r) => r.re), y: pts.map((r) => r.im), z: pts.map(() => FLOOR),
      text: pts.map((r) => {
        const tag = (r.d != null) ? ("Phi_" + r.d) : "core";
        return "zero of R: q = " + _fmtComplex(r.re, r.im) +
          "<br>|q| = " + Math.hypot(r.re, r.im).toFixed(4) + "<br>factor: " + tag;
      }),
      marker: { color: colour, size: 5, line: { color: "#fff", width: 1 } },
    };
  };
  const poleTrace = {
    type: "scatter3d", mode: "markers", name: "poles (zeros of S)",
    hoverinfo: "text",
    x: poles.map((p) => p.re), y: poles.map((p) => p.im), z: poles.map(() => CEIL),
    text: poles.map((p) => "pole: q = " + _fmtComplex(p.re, p.im) +
      "<br>|q| = " + p.mod.toFixed(4)),
    marker: { color: ROOT_POLE, size: 5, symbol: "diamond",
      line: { color: "#fff", width: 1 } },
  };
  const traces = [surface, ringAt(1, 0, "#6b757f", "solid", "unit circle"),
    markerTrace("cyclotomic", "cyclotomic zero (on unit circle)", ROOT_CYC),
    markerTrace("core", "non-cyclotomic core zero", ROOT_CORE)];
  if (meta.radius != null){
    traces.push(ringAt(meta.radius, 0, "#1f8a4c", "dash",
      "radius of convergence rho = " + meta.radius.toFixed(4)));
    const np = _nearestPole(meta);
    if (np) traces.push({ type: "scatter3d", mode: "lines",
      name: "rho = " + meta.radius.toFixed(4), hoverinfo: "skip",
      line: { color: "#1f8a4c", width: 6 },
      x: [0, np.re], y: [0, np.im], z: [0, 0] });
  }
  if (poles.length) traces.push(poleTrace);
  const layout = {
    height: 480, margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: 0, yanchor: "top" },
    scene: {
      xaxis: { title: { text: "Re" }, gridcolor: "#e3e7ec", range: [-W, W] },
      yaxis: { title: { text: "Im" }, gridcolor: "#e3e7ec", range: [-W, W] },
      zaxis: { title: { text: "log|R/S|" }, gridcolor: "#e3e7ec" },
      aspectmode: "cube",
      camera: { eye: { x: 1.5, y: 1.5, z: 1.1 } },
    },
  };
  Plotly.newPlot(fig, traces, layout, { displaylogo: false, responsive: true });
}

const ROOTS_CAP = {
  "2d": "Pole-zero plot in the z-transform grammar: zeros of R are open " +
    "circles (blue cyclotomic on the unit circle, red core off it), poles of S " +
    "are gold crosses, the green dashed ring and radial line are the radius of " +
    "convergence (the nearest pole), and the dotted line is the real axis " +
    "across which the picture is mirror-symmetric (R has real coefficients).",
  "phase": "Wegert phase portrait: hue = arg(R/S) over the complex q-square. " +
    "Each zero spins the colour wheel one way and each pole the other, so the " +
    "order is how many times the hue cycles. Pole-zero markers, the unit " +
    "circle, the radius ring/line and the real axis sit on top.",
  "3d": "The modulus surface z = log|R(q)|: each root is a well plunging to " +
    "zero, the grey ring marks the unit circle, and the markers carry the " +
    "exact cyclotomic-vs-core class. Drag to orbit, scroll to zoom.",
  "3drs": "The analytic landscape of R/S: height z = log|R(q)/S(q)| with hue = " +
    "arg(R/S). Zeros of R are dimples, poles of S are spikes (height capped so " +
    "they do not run off), and the green ring/line is the radius of convergence " +
    "(the nearest pole). Drag to orbit, scroll to zoom.",
};
function setupRootsModes(meta, root){
  root = root || document;
  const fig = root.querySelector(".roots-fig"), cap = root.querySelector(".roots-cap");
  let mode = "2d";
  const render = () => {
    if (mode === "3d") drawRoots3d(fig, meta);
    else if (mode === "3drs") drawRoots3dRS(fig, meta);
    else if (mode === "phase") drawRootsPhase(fig, meta);
    else drawRoots2d(fig, meta);
    if (cap) cap.textContent = ROOTS_CAP[mode];
    root.querySelectorAll("[data-rootmode]").forEach((b) => {
      b.classList.toggle("primary", b.dataset.rootmode === mode);
    });
  };
  root.querySelectorAll("[data-rootmode]").forEach((b) => {
    b.addEventListener("click", () => { mode = b.dataset.rootmode; render(); });
  });
  render();
}

// ---- 3D visuals: coefficient surface, root sweep, radius grid (Plotly) ----
// Each reuses the exact engine output sent from the server (coefficients from
// q_real_truncated, root classes from factor.py, radii from radius); nothing is
// recomputed here. The geometry is display-only.
const VIZ_NOPLOT = '<p class="roots-plot-msg">The plotting library could not ' +
  'load (offline?). The numeric result is listed below.</p>';
const VIZ_CONFIG = { displaylogo: false, responsive: true };
function _vizLayout3d(scene){
  return {
    height: 500,
    margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: 0, yanchor: "top" },
    scene: Object.assign({
      aspectmode: "cube",
      camera: { eye: { x: 1.6, y: 1.6, z: 1.1 } },
    }, scene),
  };
}
function drawCoeffSurface(fig, p, cap){
  const surface = {
    type: "surface", x: p.n, y: p.x, z: p.z,
    colorscale: "Viridis", showscale: true,
    colorbar: { title: { text: "c_n", side: "right" }, thickness: 12, len: 0.6 },
    contours: { z: { show: true, usecolormap: true, width: 1, project: { z: true } } },
    lighting: { ambient: 0.7, diffuse: 0.8, specular: 0.12 },
    hovertemplate: "n = %{x}<br>x = %{y:.4f}<br>c_n = %{z}<extra></extra>",
  };
  Plotly.newPlot(fig, [surface], _vizLayout3d({
    xaxis: { title: { text: "coefficient index n" }, gridcolor: "#e3e7ec" },
    yaxis: { title: { text: "real x" }, gridcolor: "#e3e7ec" },
    zaxis: { title: { text: "c_n" }, gridcolor: "#e3e7ec" },
  }), VIZ_CONFIG);
  if (cap) cap.textContent = "Taylor coefficients c_n of [x]_q as n and x vary. " +
    "Flat valleys are zero-runs; the surface crossing the floor is a sign flip. " +
    "Drag to orbit, scroll to zoom.";
}
function drawRootSweep(fig, p, cap){
  const mk = (kind, label, colour) => {
    const pts = p.points.filter((r) => r.kind === kind);
    return {
      type: "scatter3d", mode: "markers", name: label, hoverinfo: "text",
      x: pts.map((r) => r.re), y: pts.map((r) => r.im), z: pts.map((r) => r.b),
      text: pts.map((r) => {
        const tag = (r.d != null) ? ("Phi_" + r.d) : "core";
        return "b = " + r.b + "<br>q = " + _fmtComplex(r.re, r.im) +
          "<br>|q| = " + Math.hypot(r.re, r.im).toFixed(4) + "<br>factor: " + tag;
      }),
      marker: { color: colour, size: 4, line: { color: "#fff", width: 0.6 },
        symbol: "circle" },
    };
  };
  // one unit-circle reference ring per swept denominator b
  const bset = Array.from(new Set(p.points.map((r) => r.b))).sort((u, v) => u - v);
  const rings = { type: "scatter3d", mode: "lines", name: "unit circle",
    hoverinfo: "skip", line: { color: "#c2cad3", width: 2 }, x: [], y: [], z: [] };
  for (const b of bset){
    for (let t = 0; t <= 64; t++){
      const ang = (2 * Math.PI * t) / 64;
      rings.x.push(Math.cos(ang)); rings.y.push(Math.sin(ang)); rings.z.push(b);
    }
    rings.x.push(null); rings.y.push(null); rings.z.push(null);
  }
  const traces = [rings,
    mk("cyclotomic", "cyclotomic (on unit circle)", ROOT_CYC),
    mk("core", "non-cyclotomic core (off circle)", ROOT_CORE),
  ].filter((t) => t.name === "unit circle" || (t.x && t.x.length));
  const layout3d = _vizLayout3d({
    xaxis: { title: { text: "Re" }, gridcolor: "#e3e7ec" },
    yaxis: { title: { text: "Im" }, gridcolor: "#e3e7ec" },
    zaxis: { title: { text: "denominator b" }, gridcolor: "#e3e7ec" },
  });
  // camera presets: a top-down preset projects straight onto the Re/Im plane,
  // giving the 2D root-locus read without leaving the 3D view
  layout3d.updatemenus = [{
    type: "buttons", direction: "left", showactive: false,
    x: 0.5, xanchor: "center", y: 1.0, yanchor: "bottom", pad: { b: 4 },
    buttons: [
      { method: "relayout", label: "Top-down",
        args: [{ "scene.camera": { eye: { x: 0, y: 0, z: 2.4 },
          up: { x: 0, y: 1, z: 0 } } }] },
      { method: "relayout", label: "Angled",
        args: [{ "scene.camera": { eye: { x: 1.6, y: 1.6, z: 1.1 },
          up: { x: 0, y: 0, z: 1 } } }] },
    ],
  }];
  Plotly.newPlot(fig, traces, layout3d, VIZ_CONFIG);
  if (cap) cap.textContent = "Roots of R(q) for [a/b]_q stacked by denominator b. " +
    "Blue roots stay pinned to the unit-circle rings (cyclotomic); red roots drift " +
    "off them (non-cyclotomic core). Use Top-down for a flat Re/Im view; drag to " +
    "orbit, scroll to zoom.";
}
// Shared: map a radius to the same colour meaning everywhere (low = dark).
// A small Viridis interpolation so circle fills and tree nodes match the
// scatter colorbars without pulling in a colour library.
const VIRIDIS_STOPS = [
  [0.0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]],
  [0.75, [94, 201, 98]], [1.0, [253, 231, 37]],
];
function _viridis(t){
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < VIRIDIS_STOPS.length; i++){
    if (t <= VIRIDIS_STOPS[i][0]){
      const [t0, c0] = VIRIDIS_STOPS[i - 1], [t1, c1] = VIRIDIS_STOPS[i];
      const f = (t - t0) / (t1 - t0);
      const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
      return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
    }
  }
  return "rgb(253,231,37)";
}
function _radiusRange(pts){
  let lo = Infinity, hi = -Infinity;
  for (const r of pts){ lo = Math.min(lo, r.r); hi = Math.max(hi, r.r); }
  if (!isFinite(lo)){ lo = 0; hi = 1; }
  if (hi - lo < 1e-9) hi = lo + 1;
  return [lo, hi];
}
// The 3D mode (kept, demoted): height = radius, colour = denominator b, so the
// two channels are not redundant. Camera presets include a straight-down-the-
// value-axis view that recovers the 2D number-line read.
function drawRadiusGrid(fig, p, cap){
  const pts = p.points || [];
  const trace = {
    type: "scatter3d", mode: "markers", name: "radius",
    x: pts.map((r) => r.val), y: pts.map((r) => r.b), z: pts.map((r) => r.r),
    text: pts.map((r) => r.a + "/" + r.b + "<br>radius = " + r.r.toFixed(5) +
      "<br>nearest pole q = " + _fmtComplex(r.pole_re, r.pole_im)),
    hoverinfo: "text",
    marker: {
      size: 5, color: pts.map((r) => r.b), colorscale: "Cividis", showscale: true,
      colorbar: { title: { text: "b", side: "right" }, thickness: 12, len: 0.6 },
      line: { color: "#fff", width: 0.5 },
    },
  };
  // stems down to the floor make each height readable
  const stems = { type: "scatter3d", mode: "lines", name: "stems", hoverinfo: "skip",
    line: { color: "#cdd4dc", width: 1 }, x: [], y: [], z: [] };
  for (const r of pts){
    stems.x.push(r.val, r.val, null);
    stems.y.push(r.b, r.b, null);
    stems.z.push(0, r.r, null);
  }
  const layout3d = _vizLayout3d({
    xaxis: { title: { text: "value a/b" }, gridcolor: "#e3e7ec" },
    yaxis: { title: { text: "denominator b" }, gridcolor: "#e3e7ec" },
    zaxis: { title: { text: "radius |q|" }, gridcolor: "#e3e7ec" },
  });
  layout3d.updatemenus = [{
    type: "buttons", direction: "left", showactive: false,
    x: 0.5, xanchor: "center", y: 1.0, yanchor: "bottom", pad: { b: 4 },
    buttons: [
      { method: "relayout", label: "Down value axis",
        args: [{ "scene.camera": { eye: { x: 0, y: 2.4, z: 0.0 },
          up: { x: 0, y: 0, z: 1 } } }] },
      { method: "relayout", label: "Angled",
        args: [{ "scene.camera": { eye: { x: 1.6, y: 1.6, z: 1.1 },
          up: { x: 0, y: 0, z: 1 } } }] },
    ],
  }];
  Plotly.newPlot(fig, [stems, trace], layout3d, VIZ_CONFIG);
  if (cap) cap.textContent = "Radius (height) of [a/b]_q over the Farey grid, " +
    "denominator b as colour. Lower points have a pole closer to the origin. Use " +
    "Down value axis for a flat number-line read; integers (infinite radius) omitted.";
}
// (A, default) the honest low-dimensional form: radius on a number line, one
// marker per fraction, denominator b as colour. Dips below 1 are annotated.
function drawRadiusLine(fig, p, cap){
  const pts = p.points || [];
  const bmin = Math.min(...pts.map((r) => r.b));
  const bmax = Math.max(...pts.map((r) => r.b));
  const main = {
    type: "scatter", mode: "markers", name: "radius", hoverinfo: "text",
    x: pts.map((r) => r.val), y: pts.map((r) => r.r),
    text: pts.map((r) => r.a + "/" + r.b + "<br>radius = " + r.r.toFixed(5) +
      "<br>nearest pole q = " + _fmtComplex(r.pole_re, r.pole_im)),
    marker: {
      size: 10, color: pts.map((r) => r.b), colorscale: "Cividis",
      cmin: bmin, cmax: bmax, showscale: true,
      colorbar: { title: { text: "denominator b", side: "right" },
        thickness: 12, len: 0.7 },
      line: { color: "#ffffff", width: 1 },
    },
  };
  // annotate the fractions whose pole left the unit circle (radius < 1)
  const dips = pts.filter((r) => r.r < 1)
    .sort((u, v) => u.r - v.r).slice(0, 6);
  const ann = dips.map((r) => ({
    x: r.val, y: r.r, text: r.a + "/" + r.b, showarrow: true, arrowhead: 0,
    arrowcolor: "#9aa6b2", ax: 0, ay: -18,
    font: { family: "JetBrains Mono,monospace", size: 10, color: "#475059" },
  }));
  const xs = pts.map((r) => r.val);
  const layout = {
    height: 460, margin: { l: 54, r: 18, t: 14, b: 46 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 13, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: false, annotations: ann,
    xaxis: { title: { text: "value a/b" }, gridcolor: "#eef1f5" },
    yaxis: { title: { text: "radius |q| (nearest pole)" }, gridcolor: "#eef1f5" },
    shapes: [{ type: "line", x0: Math.min(...xs) - 0.2, x1: Math.max(...xs) + 0.2,
      y0: 1, y1: 1, line: { color: "#9aa6b2", width: 1.5, dash: "dash" } }],
  };
  Plotly.newPlot(fig, [main], layout, VIZ_CONFIG);
  if (cap) cap.textContent = "Radius of convergence on the number line, denominator " +
    "b as colour. The dashed line is |q| = 1; fractions dipping below it (labelled) " +
    "are the ones whose nearest pole left the unit circle.";
}
// (B) Ford circles: every reduced a/b gets a circle of radius 1/(2b^2) tangent
// to the number line at a/b. Circle size encodes the denominator; the fill
// colour encodes the radius of convergence. The canonical Farey-grid picture.
function drawFordCircles(fig, p, cap){
  const pts = p.points || [];
  const [lo, hi] = _radiusRange(pts);
  const xs = pts.map((r) => r.val);
  const shapes = pts.map((r) => {
    const rad = 1 / (2 * r.b * r.b);
    return {
      type: "circle", xref: "x", yref: "y",
      x0: r.val - rad, x1: r.val + rad, y0: 0, y1: 2 * rad,
      fillcolor: _viridis((r.r - lo) / (hi - lo)),
      line: { color: "rgba(255,255,255,0.7)", width: 0.5 },
    };
  });
  // a transparent marker at each circle centre carries the hover and the colorbar
  const centres = {
    type: "scatter", mode: "markers", hoverinfo: "text",
    x: pts.map((r) => r.val), y: pts.map((r) => 1 / (2 * r.b * r.b)),
    text: pts.map((r) => r.a + "/" + r.b + "<br>radius = " + r.r.toFixed(5) +
      "<br>circle radius 1/(2b^2) = " + (1 / (2 * r.b * r.b)).toFixed(4)),
    marker: {
      size: 6, color: pts.map((r) => r.r), colorscale: "Viridis",
      cmin: lo, cmax: hi, showscale: true,
      colorbar: { title: { text: "radius |q|", side: "right" },
        thickness: 12, len: 0.7 },
      opacity: 0.0,
    },
  };
  const layout = {
    height: 430, margin: { l: 40, r: 18, t: 14, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: false,
    xaxis: { title: { text: "value a/b" }, gridcolor: "#eef1f5",
      zeroline: false, constrain: "domain" },
    yaxis: { title: { text: "1/(2b^2)" }, gridcolor: "#eef1f5",
      zeroline: true, zerolinecolor: "#9aa6b2",
      scaleanchor: "x", scaleratio: 1, range: [-0.02, 0.6] },
    shapes: shapes.concat([{ type: "line", x0: Math.min(...xs) - 0.2,
      x1: Math.max(...xs) + 0.2, y0: 0, y1: 0,
      line: { color: "#6b757f", width: 1.5 } }]),
  };
  Plotly.newPlot(fig, [centres], layout, VIZ_CONFIG);
  if (cap) cap.textContent = "Ford circles: each reduced a/b sits tangent to the " +
    "number line with radius 1/(2b^2), so circle size is the denominator and " +
    "neighbours are tangent. Fill colour is the radius of convergence.";
}
// (C) the (a, b) grid: colour = radius, with omitted and non-reduced cells left
// blank. Scan a column for "does the radius depend on the denominator?".
function drawRadiusGridAB(fig, p, cap){
  const pts = p.points || [];
  const amax = p.a_max || Math.max(...pts.map((r) => r.a));
  const bmax = p.b_max || Math.max(...pts.map((r) => r.b));
  const lookup = {};
  for (const r of pts) lookup[r.a + "/" + r.b] = r;
  const xa = []; for (let a = 1; a <= amax; a++) xa.push(a);
  const yb = []; for (let b = 1; b <= bmax; b++) yb.push(b);
  const z = [], hover = [];
  for (let b = 1; b <= bmax; b++){
    const zr = [], hr = [];
    for (let a = 1; a <= amax; a++){
      const hit = lookup[a + "/" + b];
      zr.push(hit ? hit.r : null);
      hr.push(hit ? (a + "/" + b + "<br>radius = " + hit.r.toFixed(5))
        : (a + "/" + b + "<br>(omitted: integer or not reduced)"));
    }
    z.push(zr); hover.push(hr);
  }
  const heat = {
    type: "heatmap", x: xa, y: yb, z: z, text: hover, hoverinfo: "text",
    colorscale: "Viridis", showscale: true,
    colorbar: { title: { text: "radius |q|", side: "right" }, thickness: 12, len: 0.75 },
    xgap: 2, ygap: 2,
  };
  const layout = {
    height: 440, margin: { l: 48, r: 18, t: 12, b: 46 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    xaxis: { title: { text: "numerator a" }, dtick: 1, gridcolor: "#eef1f5" },
    yaxis: { title: { text: "denominator b" }, dtick: 1, gridcolor: "#eef1f5" },
  };
  Plotly.newPlot(fig, [heat], layout, VIZ_CONFIG);
  if (cap) cap.textContent = "The (a, b) grid coloured by radius; blank cells are " +
    "integers or non-reduced fractions. Scan down a column to see whether the " +
    "radius tracks the denominator.";
}
// (D) the Stern-Brocot tree: each node is the mediant of its parents, coloured
// by radius. Shows whether the radius follows the recursive mediant structure.
function drawSternBrocot(fig, p, cap){
  const pts = p.points || [];
  const bmax = p.b_max || Math.max(...pts.map((r) => r.b));
  const valmax = (p.a_max || Math.max(...pts.map((r) => r.a)));
  const lookup = {};
  for (const r of pts) lookup[r.a + "/" + r.b] = r;
  const nodes = [], edges = [];
  function rec(la, lb, ra, rb, depth){
    const a = la + ra, b = lb + rb, val = a / b;
    if (b > bmax || val > valmax + 1e-9 || depth > 14) return null;
    const node = { a, b, val, depth };
    nodes.push(node);
    const lc = rec(la, lb, a, b, depth + 1);
    const rc = rec(a, b, ra, rb, depth + 1);
    if (lc) edges.push([val, depth, lc.val, lc.depth]);
    if (rc) edges.push([val, depth, rc.val, rc.depth]);
    return node;
  }
  rec(0, 1, 1, 0, 0);  // mediant of 0/1 and 1/0 is 1/1, the tree root
  const [lo, hi] = _radiusRange(pts);
  const edgeTrace = { type: "scatter", mode: "lines", hoverinfo: "skip",
    line: { color: "#d4dae1", width: 1 }, x: [], y: [] };
  for (const [x0, d0, x1, d1] of edges){
    edgeTrace.x.push(x0, x1, null); edgeTrace.y.push(-d0, -d1, null);
  }
  const known = nodes.filter((n) => lookup[n.a + "/" + n.b]);
  const unknown = nodes.filter((n) => !lookup[n.a + "/" + n.b]);
  const nodeTrace = {
    type: "scatter", mode: "markers", hoverinfo: "text", name: "fractions",
    x: known.map((n) => n.val), y: known.map((n) => -n.depth),
    text: known.map((n) => n.a + "/" + n.b + "<br>radius = " +
      lookup[n.a + "/" + n.b].r.toFixed(5) + "<br>depth " + n.depth),
    marker: {
      size: 11, color: known.map((n) => lookup[n.a + "/" + n.b].r),
      colorscale: "Viridis", cmin: lo, cmax: hi, showscale: true,
      colorbar: { title: { text: "radius |q|", side: "right" },
        thickness: 12, len: 0.75 },
      line: { color: "#ffffff", width: 1 },
    },
  };
  const ghostTrace = {
    type: "scatter", mode: "markers", hoverinfo: "text", name: "outside grid",
    x: unknown.map((n) => n.val), y: unknown.map((n) => -n.depth),
    text: unknown.map((n) => n.a + "/" + n.b + "<br>(outside the a, b grid)"),
    marker: { size: 6, color: "#e3e7ec", line: { color: "#cdd4dc", width: 0.5 } },
  };
  Plotly.newPlot(fig, [edgeTrace, ghostTrace, nodeTrace], {
    height: 470, margin: { l: 40, r: 18, t: 14, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: false,
    xaxis: { title: { text: "value a/b" }, gridcolor: "#eef1f5" },
    yaxis: { title: { text: "tree depth (down)" }, gridcolor: "#eef1f5",
      zeroline: false },
  }, VIZ_CONFIG);
  if (cap) cap.textContent = "The Stern-Brocot tree: each node is the mediant of " +
    "its parents, placed at its value and depth, coloured by radius. Grey nodes " +
    "fall outside the a, b grid. Look for the radius tracking mediant depth.";
}
// The flatter 2D companion to the coefficient surface: the same c_n grid as a
// heat map, easier to read off than the 3D view when the surface is busy.
function drawCoeffHeatmap(fig, p, cap){
  const heat = {
    type: "heatmap", x: p.n, y: p.x, z: p.z,
    colorscale: "RdBu", reversescale: true, zmid: 0,
    colorbar: { title: { text: "c_n", side: "right" }, thickness: 12, len: 0.75 },
    hovertemplate: "n = %{x}<br>x = %{y:.4f}<br>c_n = %{z}<extra></extra>",
  };
  const layout = {
    height: 460,
    margin: { l: 58, r: 18, t: 12, b: 46 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    xaxis: { title: { text: "coefficient index n" }, dtick: 1, gridcolor: "#eef1f5" },
    yaxis: { title: { text: "real x" }, gridcolor: "#eef1f5" },
  };
  Plotly.newPlot(fig, [heat], layout, VIZ_CONFIG);
  if (cap) cap.textContent = "Coefficient c_n of [x]_q as a heat map: red and blue " +
    "are the sign, white is zero (the zero-runs). Read one horizontal line for a " +
    "single x. Drag to zoom, double-click to reset.";
}
// ---- root-sweep companion views: a root-locus problem reads best out of 3D ----
function _sweepExtent(pts){
  let e = 1.0;
  for (const r of pts) e = Math.max(e, Math.abs(r.re), Math.abs(r.im));
  return e * 1.12 + 0.06;
}
// (1) the default: a single Re/Im plane, denominator b encoded as marker colour.
// Cyclotomic roots stay on the unit circle, the core fans inward, no orbiting.
function drawRootLocus2d(fig, p, cap){
  const pts = p.points || [];
  const extent = _sweepExtent(pts);
  const mk = (kind, symbol, label, showscale) => {
    const sel = pts.filter((r) => r.kind === kind);
    return {
      type: "scatter", mode: "markers", name: label, hoverinfo: "text",
      x: sel.map((r) => r.re), y: sel.map((r) => r.im),
      text: sel.map((r) => {
        const tag = (r.d != null) ? ("Phi_" + r.d) : "core";
        return "b = " + r.b + "<br>q = " + _fmtComplex(r.re, r.im) +
          "<br>|q| = " + Math.hypot(r.re, r.im).toFixed(4) + "<br>factor: " + tag;
      }),
      marker: {
        size: 9, symbol: symbol,
        color: sel.map((r) => r.b), colorscale: "Viridis",
        cmin: Math.min(...pts.map((r) => r.b)),
        cmax: Math.max(...pts.map((r) => r.b)),
        showscale: showscale,
        colorbar: showscale
          ? { title: { text: "b", side: "right" }, thickness: 12, len: 0.7 }
          : undefined,
        line: { color: "#ffffff", width: 1 },
      },
    };
  };
  const traces = [
    mk("cyclotomic", "circle", "cyclotomic (circle marker)", true),
    mk("core", "diamond", "core (diamond marker)", false),
  ].filter((t) => t.x.length);
  const layout = {
    height: 470, margin: { l: 48, r: 18, t: 14, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 13, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.16, yanchor: "top" },
    xaxis: { title: { text: "Re" }, range: [-extent, extent], zeroline: true,
      zerolinecolor: "#cdd4dc", gridcolor: "#eef1f5", constrain: "domain" },
    yaxis: { title: { text: "Im" }, range: [-extent, extent], zeroline: true,
      zerolinecolor: "#cdd4dc", gridcolor: "#eef1f5",
      scaleanchor: "x", scaleratio: 1 },
    shapes: [{ type: "circle", xref: "x", yref: "y", x0: -1, y0: -1, x1: 1, y1: 1,
      line: { color: "#9aa6b2", width: 1.5 } }],
  };
  Plotly.newPlot(fig, traces, layout, VIZ_CONFIG);
  if (cap) cap.textContent = "Root locus on one Re/Im plane, denominator b as colour " +
    "(dark to light). Circle markers are cyclotomic (pinned to the unit circle), " +
    "diamonds are the core fanning inward. Drag to zoom, double-click to reset.";
}
// (2) the on/off-circle story as a 1D read: |q| against b. Cyclotomic roots sit
// on the flat |q| = 1 line; core roots peel away from it.
function drawRootModulus(fig, p, cap){
  const pts = p.points || [];
  const mk = (kind, colour, label) => {
    const sel = pts.filter((r) => r.kind === kind);
    return {
      type: "scatter", mode: "markers", name: label, hoverinfo: "text",
      x: sel.map((r) => r.b), y: sel.map((r) => Math.hypot(r.re, r.im)),
      text: sel.map((r) => {
        const tag = (r.d != null) ? ("Phi_" + r.d) : "core";
        return "b = " + r.b + "<br>|q| = " + Math.hypot(r.re, r.im).toFixed(4) +
          "<br>factor: " + tag;
      }),
      marker: { color: colour, size: 8, line: { color: "#fff", width: 0.8 } },
    };
  };
  const bs = pts.map((r) => r.b);
  const bmin = Math.min(...bs, 1), bmax = Math.max(...bs, 1);
  const traces = [
    mk("cyclotomic", ROOT_CYC, "cyclotomic (|q| = 1)"),
    mk("core", ROOT_CORE, "core (|q| != 1)"),
  ].filter((t) => t.x.length);
  const layout = {
    height: 440, margin: { l: 54, r: 18, t: 14, b: 46 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 13, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.18, yanchor: "top" },
    xaxis: { title: { text: "denominator b" }, dtick: 1, gridcolor: "#eef1f5" },
    yaxis: { title: { text: "|q| (root modulus)" }, gridcolor: "#eef1f5" },
    shapes: [{ type: "line", x0: bmin - 0.5, x1: bmax + 0.5, y0: 1, y1: 1,
      line: { color: "#9aa6b2", width: 1.5, dash: "dash" } }],
  };
  Plotly.newPlot(fig, traces, layout, VIZ_CONFIG);
  if (cap) cap.textContent = "Root modulus |q| against denominator b. The dashed " +
    "line is |q| = 1: cyclotomic roots sit on it, core roots peel away. The whole " +
    "on-circle question becomes a single vertical read.";
}
// (3) animate the migration: one frame per b in the Re/Im plane, earlier roots
// ghosted so you watch the trail form as you scrub or play.
function drawRootLocusAnimated(fig, p, cap){
  const pts = p.points || [];
  const bs = Array.from(new Set(pts.map((r) => r.b))).sort((u, v) => u - v);
  const extent = _sweepExtent(pts);
  const colourOf = (r) => (r.kind === "cyclotomic" ? ROOT_CYC : ROOT_CORE);
  const curOf = (b) => {
    const sel = pts.filter((r) => r.b === b);
    return {
      x: sel.map((r) => r.re), y: sel.map((r) => r.im),
      text: sel.map((r) => "b = " + b + "<br>|q| = " +
        Math.hypot(r.re, r.im).toFixed(4)),
      marker: { color: sel.map(colourOf), size: 11,
        line: { color: "#fff", width: 1.2 } },
    };
  };
  const trailOf = (b) => {
    const sel = pts.filter((r) => r.b < b);
    return {
      x: sel.map((r) => r.re), y: sel.map((r) => r.im),
      marker: { color: sel.map(colourOf), size: 5, opacity: 0.22,
        line: { width: 0 } },
    };
  };
  const data = [
    Object.assign({ type: "scatter", mode: "markers", name: "trail",
      hoverinfo: "skip" }, trailOf(bs[0])),
    Object.assign({ type: "scatter", mode: "markers", name: "current b",
      hoverinfo: "text" }, curOf(bs[0])),
  ];
  const frames = bs.map((b) => ({ name: String(b), data: [
    Object.assign({ type: "scatter", mode: "markers", hoverinfo: "skip" }, trailOf(b)),
    Object.assign({ type: "scatter", mode: "markers", hoverinfo: "text" }, curOf(b)),
  ] }));
  const steps = bs.map((b) => ({ method: "animate", label: String(b),
    args: [[String(b)], { mode: "immediate", frame: { duration: 0, redraw: true },
      transition: { duration: 0 } }] }));
  const layout = {
    height: 490, margin: { l: 48, r: 18, t: 14, b: 60 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter,system-ui,sans-serif", size: 12, color: "#475059" },
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: false,
    xaxis: { title: { text: "Re" }, range: [-extent, extent], zeroline: true,
      zerolinecolor: "#cdd4dc", gridcolor: "#eef1f5", constrain: "domain" },
    yaxis: { title: { text: "Im" }, range: [-extent, extent], zeroline: true,
      zerolinecolor: "#cdd4dc", gridcolor: "#eef1f5", scaleanchor: "x", scaleratio: 1 },
    shapes: [{ type: "circle", x0: -1, y0: -1, x1: 1, y1: 1,
      line: { color: "#9aa6b2", width: 1.5 } }],
    updatemenus: [{ type: "buttons", showactive: false, x: 0.02, y: 0,
      xanchor: "left", yanchor: "top", pad: { t: 8 },
      buttons: [
        { method: "animate", label: "Play",
          args: [null, { mode: "immediate", fromcurrent: true,
            frame: { duration: 600, redraw: true },
            transition: { duration: 180 } }] },
        { method: "animate", label: "Pause",
          args: [[null], { mode: "immediate",
            frame: { duration: 0, redraw: false } }] },
      ] }],
    sliders: [{ active: 0, x: 0.14, len: 0.84, xanchor: "left", y: 0, yanchor: "top",
      currentvalue: { prefix: "b = ", font: { size: 13 } }, steps: steps }],
  };
  Plotly.newPlot(fig, data, layout, VIZ_CONFIG).then(() => Plotly.addFrames(fig, frames));
  if (cap) cap.textContent = "Migration animated over b: each frame is one " +
    "denominator's roots, earlier ones ghosted as a trail. Press Play, or drag " +
    "the slider to scrub through the sweep.";
}
function drawViz(fig, p, cap){
  if (!fig || !p) return;
  if (typeof Plotly === "undefined"){ fig.innerHTML = VIZ_NOPLOT; return; }
  if (p.kind === "coeff-surface") return drawCoeffSurface(fig, p, cap);
  if (p.kind === "root-sweep") return drawRootLocus2d(fig, p, cap);
  if (p.kind === "radius-grid") return drawRadiusLine(fig, p, cap);
}
// The per-visual view menu. Each entry is {id, label, fn}; the first is the
// default. Function declarations above are hoisted, so naming them here is safe.
const VIZ_MODES = {
  "coeff-surface": [
    { id: "3d", label: "3D surface", fn: drawCoeffSurface },
    { id: "2d", label: "2D heat map", fn: drawCoeffHeatmap },
  ],
  "root-sweep": [
    { id: "locus", label: "2D locus", fn: drawRootLocus2d },
    { id: "modulus", label: "|q| vs b", fn: drawRootModulus },
    { id: "animate", label: "Animate", fn: drawRootLocusAnimated },
    { id: "stack", label: "3D stack", fn: drawRootSweep },
  ],
  "radius-grid": [
    { id: "line", label: "Number line", fn: drawRadiusLine },
    { id: "ford", label: "Ford circles", fn: drawFordCircles },
    { id: "grid", label: "(a, b) grid", fn: drawRadiusGridAB },
    { id: "tree", label: "Stern-Brocot", fn: drawSternBrocot },
    { id: "3d", label: "3D stems", fn: drawRadiusGrid },
  ],
};
function vizModeButtons(kind){
  const modes = VIZ_MODES[kind];
  if (!modes) return "";
  return '<div class="roots-modes" role="group" aria-label="plot view">' +
    modes.map((m) => '<button class="mini" type="button" data-vizmode="' +
      m.id + '">' + esc(m.label) + '</button>').join("") +
    '</div>';
}
// Wire the view toggle: redraw on click and highlight the active button. Visuals
// with no menu (radius-grid) draw straight through drawViz.
function setupViz(p, root){
  root = root || document;
  const fig = root.querySelector(".viz-fig"), cap = root.querySelector(".viz-cap");
  const modes = VIZ_MODES[p.kind];
  if (!modes){ drawViz(fig, p, cap); return; }
  let mode = modes[0].id;
  const render = () => {
    const m = modes.find((x) => x.id === mode) || modes[0];
    m.fn(fig, p, cap);
    root.querySelectorAll("[data-vizmode]").forEach((b) => {
      b.classList.toggle("primary", b.dataset.vizmode === mode);
    });
  };
  root.querySelectorAll("[data-vizmode]").forEach((b) => {
    b.addEventListener("click", () => { mode = b.dataset.vizmode; render(); });
  });
  render();
}

// ---- Conway-Coxeter frieze with the q-coefficient overlay (Plotly) ----
// The integer triangle and the q-polynomial on every cell come from the
// frieze backend (qfrieze, or the vendored qreals._frieze); this only lays
// them out as a staggered diagonal lattice. Cell colour shades by the integer
// value (the q = 1 specialisation); the label on each cell is its q-polynomial.
const FZ_ACCENT = [36, 86, 166];   // var(--accent) #2456a6
function _fzFill(value, maxv){
  if (value === 1) return "#eef1f5";              // border / unit cells
  const t = maxv > 1 ? (value - 1) / (maxv - 1) : 0;
  const a = (0.10 + 0.5 * t).toFixed(3);
  return "rgba(" + FZ_ACCENT[0] + "," + FZ_ACCENT[1] + "," + FZ_ACCENT[2] + "," + a + ")";
}
const FRIEZE_CAP = {
  poly: "Each staggered cell shows its q-coefficient polynomial; at q = 1 it " +
    "becomes the integer Conway-Coxeter frieze (row 2 is the quiddity [c_i]_q). " +
    "Hover a cell for its value; scroll sideways if the strip runs wide.",
  int: "The classical Conway-Coxeter integer frieze: every cell is the q-cell " +
    "above evaluated at q = 1. Each staggered diamond satisfies ad - bc = 1 and " +
    "the border rows are all 1s. Scroll sideways if the strip runs wide.",
};
// Compact in-cell q-polynomial label (drop the spaces around "+", e.g.
// q^6+2q^5+...), so the lattice stays dense; the hover keeps the spaced form.
function _fzLab(s){ return (s || "").replace(/ \+ /g, "+"); }
function drawFrieze(fig, fz, mode){
  if (!fig) return;
  mode = (mode === "int") ? "int" : "poly";
  const rows = fz.rows || [];
  const D = rows.length;
  const cols = fz.cols || fz.width || (rows[0] ? rows[0].cells.length : 0);
  if (typeof Plotly === "undefined" || !D || !cols){
    fig.innerHTML = '<p class="roots-plot-msg">The plotting library could not ' +
      'load (offline?). The frieze is listed as text below.</p>';
    return;
  }
  // The in-cell label: the q-polynomial, or just its q = 1 integer value.
  const labelOf = (c) => (mode === "int") ? String(c.value) : _fzLab(c.txt);
  // Largest integer entry (colour shading) and the longest cell label (so each
  // cell is wide enough that nothing spills into its neighbour).
  let maxv = 1, maxChars = 1;
  for (const row of rows) for (const c of row.cells){
    maxv = Math.max(maxv, c.value);
    maxChars = Math.max(maxChars, labelOf(c).length);
  }

  // Size each cell to its content: one fixed-pixel lattice, scrolled if wide.
  // Kept compact on purpose so the whole strip reads at a glance; the integer
  // view packs tighter still and fits without scroll.
  const unitW = (mode === "int")
    ? Math.max(30, maxChars * 7 + 10)
    : Math.min(140, Math.max(44, Math.round(maxChars * 4.8 + 12)));
  const unitH = 28;
  const HW = 0.47, HH = 0.42;          // cell half-width / half-height (data units)
  const padL = 0.3, padR = 0.7, padV = 0.45;

  const shapes = [], annotations = [];
  const hx = [], hy = [], htext = [];
  for (let d = 0; d < D; d++){
    const i = d + 1;                   // absolute row (row 0 is the undrawn border)
    const offset = (i % 2) * 0.5;      // Conway-Coxeter half-cell stagger
    const y = (D - 1 - d);             // top drawn row highest, bottom lowest
    const cells = rows[d].cells;
    for (let j = 0; j < cols; j++){
      const c = cells[j];
      const x = j + offset + 0.5;
      shapes.push({
        type: "rect", xref: "x", yref: "y", layer: "below",
        x0: x - HW, x1: x + HW, y0: y - HH, y1: y + HH,
        line: { color: "#cdd4dc", width: 1 },
        fillcolor: _fzFill(c.value, maxv),
      });
      const lab = labelOf(c);
      const fs = (mode === "int") ? 12 : Math.max(7, Math.min(11,
        Math.floor((unitW - 12) / Math.max(1, lab.length * 0.6))));
      annotations.push({
        x, y, xref: "x", yref: "y", showarrow: false,
        text: lab, font: { size: fs, color: "#1f2328",
          family: "JetBrains Mono,monospace" },
      });
      hx.push(x); hy.push(y);
      htext.push("row " + i + ", col " + j + "<br>q-cell: " + (c.txt || "") +
        "<br>at q = 1: " + c.value);
    }
  }

  const hover = {
    x: hx, y: hy, text: htext, type: "scatter", mode: "markers",
    hoverinfo: "text", showlegend: false,
    marker: { size: Math.round(Math.min(unitW, unitH) * 0.85),
      color: "rgba(0,0,0,0)", symbol: "square" },
  };

  const spanX = cols + padL + padR;
  const spanY = D - 1 + 2 * padV;
  const m = { l: 10, r: 10, t: 10, b: 10 };
  const layout = {
    width: Math.round(spanX * unitW) + m.l + m.r,
    height: Math.round(spanY * unitH) + m.t + m.b,
    margin: m,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    hoverlabel: { font: { family: "JetBrains Mono,monospace", size: 12 } },
    showlegend: false,
    xaxis: { visible: false, range: [-padL, cols + padR], fixedrange: true },
    yaxis: { visible: false, range: [-padV, D - 1 + padV], fixedrange: true },
    shapes, annotations,
  };
  // responsive:false keeps the cells at their content width; the .frieze-fig
  // container scrolls horizontally when the strip is wider than the panel.
  const config = { displaylogo: false, responsive: false,
    displayModeBar: false };
  Plotly.newPlot(fig, [hover], layout, config);
}
function setupFriezeModes(fz, root){
  root = root || document;
  const fig = root.querySelector(".frieze-fig"), cap = root.querySelector(".frieze-cap");
  let mode = "poly";
  const render = () => {
    drawFrieze(fig, fz, mode);
    if (cap) cap.textContent = FRIEZE_CAP[mode];
    root.querySelectorAll("[data-friezemode]").forEach((b) => {
      b.classList.toggle("primary", b.dataset.friezemode === mode);
    });
  };
  root.querySelectorAll("[data-friezemode]").forEach((b) => {
    b.addEventListener("click", () => { mode = b.dataset.friezemode; render(); });
  });
  render();
}

// ---- saved list (localStorage) --------------------------------------
function loadSaved(){ return store.getSaved(); }
function writeSaved(list){ store.setSaved(list); updateCount(); }
function updateCount(){
  $("navCount").textContent = loadSaved().length;
}
function saveResult(r){
  const list = loadSaved();
  const label = OPS[r.op] ? OPS[r.op].name : r.op;
  list.unshift({
    op: r.op, input: r.input, args: r.args,
    latex: r.latex, text: r.text, rows: r.rows, meta: r.meta || null,
    label: label, tags: [], note: "", when: new Date().toISOString()
  });
  writeSaved(list);
  toast("Saved");
  // refresh whichever saved list is on screen
  if (!savedView.classList.contains("hidden")) renderSavedInto($("savedFull"), true);
}
function removeSaved(idx){
  const list = loadSaved();
  list.splice(idx, 1);
  writeSaved(list);
  if (!savedView.classList.contains("hidden")) renderSavedInto($("savedFull"), true);
  else renderSavedInto($("savedHome"), false);
}
function fmtWhen(iso){
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {month:"short", day:"numeric",
      hour:"2-digit", minute:"2-digit"});
  } catch(e){ return ""; }
}
function reopenSaved(idx){
  const item = loadSaved()[idx];
  if (!item) return;
  const preset = Object.assign({}, item.args, { input: item.input });
  openOp(item.op, preset, item);
}
function renderSavedInto(container, full){
  const list = loadSaved();
  let html = '<div class="saved-head"><h2>Saved results</h2>' +
    '<span class="count">' + list.length + ' kept in this browser</span>';
  // The full Saved view (not the home preview) carries the multi-select compare.
  if (full && list.length){
    html += '<button class="mini primary cmp-go" type="button">Compare selected</button>';
  }
  html += '</div>';
  // Client-side search over op/input/note/tags (full view only).
  if (full && list.length){
    html += '<input id="savedSearch" class="saved-search" type="text" ' +
      'placeholder="Search saved (op, input, note, tag)…">';
  }
  if (!list.length){
    html += '<div class="saved-empty">No saved results yet. Compute something and ' +
      'press <b>Save this result</b> to keep it here.</div>';
    container.innerHTML = html;
    return;
  }
  const checkHead = full ? '<th class="saved-check"></th>' : '';
  const tagHead = full ? '<th>Tags</th>' : '';
  html += '<table class="saved"><thead><tr>' + checkHead +
    '<th>Operation</th><th>Input</th>' + tagHead +
    '<th>Saved</th><th class="saved-act"></th></tr></thead><tbody>';
  list.forEach((item, i) => {
    const inText = item.input + (item.args && item.args.y ? (", y=" + item.args.y) : "");
    const check = full
      ? '<td class="saved-check"><input type="checkbox" class="cmp-check" data-cmp="' + i + '"></td>'
      : '';
    const tags = item.tags || [];
    let tagCell = "";
    if (full){
      const chips = tags.map((t, ti) =>
        '<span class="tag">' + esc(t) +
        '<button data-tagdel="' + i + '" data-tagi="' + ti + '" type="button" aria-label="remove tag">&times;</button></span>').join("");
      tagCell = '<td class="saved-tags">' + chips +
        '<button class="tag-add" data-tagadd="' + i + '" type="button">+ tag</button></td>';
    }
    const search = ((item.label || item.op) + " " + item.input + " " +
      (item.note || "") + " " + tags.join(" ")).toLowerCase();
    html += '<tr data-idx="' + i + '" data-search="' + esc(search) + '">' + check +
      '<td><span class="saved-op">' + esc(item.label || item.op) + '</span></td>' +
      '<td><span class="saved-in">' + esc(inText) + '</span></td>' + tagCell +
      '<td><span class="saved-when">' + esc(fmtWhen(item.when)) + '</span></td>' +
      '<td class="saved-act">' +
        '<button class="mini" data-open="' + i + '" type="button">Open</button> ' +
        '<button class="mini" data-share="' + i + '" type="button">Share</button> ' +
        '<button class="mini danger" data-del="' + i + '" type="button">Remove</button>' +
      '</td></tr>';
    if (full){
      // A per-result note (B1): inline, click-to-type, persisted to the entry.
      const cols = 1 /*op*/ + 1 /*input*/ + 1 /*tags*/ + 1 /*when*/ + 1 /*act*/ + (check ? 1 : 0);
      html += '<tr class="saved-note-row" data-noterow="' + i + '"><td colspan="' + cols + '">' +
        '<input class="saved-note" data-note="' + i + '" placeholder="Add a note for this result…" value="' +
        esc(item.note || "") + '"></td></tr>';
    }
  });
  html += '</tbody></table>';
  container.innerHTML = html;
  container.querySelectorAll("[data-open]").forEach((b) => {
    b.addEventListener("click", (e) => { e.stopPropagation(); reopenSaved(+b.dataset.open); });
  });
  container.querySelectorAll("[data-del]").forEach((b) => {
    b.addEventListener("click", (e) => { e.stopPropagation(); removeSaved(+b.dataset.del); });
  });
  container.querySelectorAll("[data-share]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      const it = list[+b.dataset.share];
      if (it) openShareMenuFor([{ op: it.op, input: it.input, args: it.args || {}, note: it.note || "" }], e.currentTarget);
    });
  });
  container.querySelectorAll(".saved-note").forEach((inp) => {
    inp.addEventListener("change", (e) => store.updateSavedNote(+inp.dataset.note, e.target.value));
    inp.addEventListener("click", (e) => e.stopPropagation());
  });
  // Row click reopens, except when the click lands on a checkbox, a button, or
  // the note input (which is its own interactive row).
  container.querySelectorAll("tr[data-idx]").forEach((row) => {
    row.addEventListener("click", (e) => {
      if (e.target.closest(".saved-check") || e.target.closest("button") || e.target.closest(".saved-note")) return;
      reopenSaved(+row.dataset.idx);
    });
  });
  const cmpGo = container.querySelector(".cmp-go");
  if (cmpGo) cmpGo.addEventListener("click", () => {
    Array.from(container.querySelectorAll(".cmp-check:checked"))
      .map((c) => list[+c.dataset.cmp]).filter(Boolean)
      .forEach((it) => store.addCompare({ op: it.op, input: it.input, args: it.args || {} }));
    renderTray(); goWorkspace();
  });
  // Tag add/remove (full view only) — re-render this same container after a change.
  container.querySelectorAll("[data-tagadd]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      const i = +b.dataset.tagadd;
      const tag = (prompt("Add a tag") || "").trim();
      if (!tag) return;
      const cur = (list[i] && list[i].tags) || [];
      if (cur.indexOf(tag) === -1) store.updateSavedTags(i, cur.concat(tag));
      renderSavedInto(container, full);
    });
  });
  container.querySelectorAll("[data-tagdel]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      const i = +b.dataset.tagdel, ti = +b.dataset.tagi;
      const cur = (list[i] && list[i].tags) || [];
      store.updateSavedTags(i, cur.filter((_, idx) => idx !== ti));
      renderSavedInto(container, full);
    });
  });
  // Live client-side filter over the already-rendered rows.
  const search = container.querySelector("#savedSearch");
  if (search) search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    container.querySelectorAll("tr[data-idx]").forEach((row) => {
      row.style.display = (!q || (row.dataset.search || "").includes(q)) ? "" : "none";
    });
  });
}

// ---- workspace = expanded editable compare ---------------------------
// The Workspace is a full-screen, tiled, EDITABLE view of the compare items
// held in the store. Each tile owns an op-selector + fields that recompute live
// and write back to the store, plus the full renderResultInto output (math,
// rows, interactive plots). Pressing × removes the item from the compare list.
function tileLabel(r){
  const label = OPS[r.op] ? OPS[r.op].name : r.op;
  let inp = (r.input != null) ? String(r.input) : "";
  if (r.args && r.args.y) inp += ", y = " + r.args.y;
  return label + (inp ? "  ·  " + inp : "");
}
function renderWorkspace(){
  const grid = $("wsGrid");
  grid.innerHTML = "";
  const items = store.getCompare();
  $("wsEmpty").classList.toggle("hidden", items.length > 0);
  items.forEach((c) => addWsTile(c));
  grid.dataset.count = items.length > 4 ? "many" : String(items.length);
}
function addWsTile(item){
  const grid = $("wsGrid");
  const node = document.createElement("div");
  node.className = "ws-tile";
  node.dataset.id = item.id;
  node.innerHTML =
    '<div class="ws-tile-head">' +
      '<select class="cmp-row-op"></select>' +
      '<button class="ws-tile-close" type="button" aria-label="close">&times;</button>' +
    '</div>' +
    '<div class="cmp-row-fields"></div>' +
    '<div class="ws-tile-body"></div>';
  node.querySelector(".cmp-row-op").innerHTML = cmpOpOptions(item.op);
  grid.appendChild(node);
  cmpBuildFields(node, item.op, Object.assign({ input: item.input }, item.args || {}));
  node._compute = async () => {
    const { op, input, args } = cmpRowGet(node);
    store.updateCompare(item.id, { op, input, args });
    const res = await computeResult(op, input, args);
    const body = node.querySelector(".ws-tile-body");
    if (!res){ body.innerHTML = '<div class="result-error">could not compute</div>'; return; }
    renderResultInto(body, res, { actions: "tile" });
    renderTray();
  };
  node.querySelector(".cmp-row-op").addEventListener("change", (e) => {
    const cur = cmpRowGet(node);
    cmpBuildFields(node, e.target.value, { input: cur.input });
    node._compute();
  });
  node.querySelector(".ws-tile-close").addEventListener("click", () => {
    store.removeCompare(item.id); node.remove(); renderTray();
    $("wsGrid").dataset.count = (function(n){ return n > 4 ? "many" : String(n); })($("wsGrid").querySelectorAll(".ws-tile").length);
    $("wsEmpty").classList.toggle("hidden", store.getCompare().length > 0);
    resizeWorkspacePlots();
  });
  node._compute();
}
function resizeWorkspacePlots(){
  if (typeof Plotly === "undefined") return;
  $("wsGrid").querySelectorAll(".js-plotly-plot").forEach((fig) => {
    if (fig.closest(".frieze-fig")) return;   // fixed-size lattice; do not stretch
    try { Plotly.Plots.resize(fig); } catch(e){}
  });
}
function updateWsCount(){
  $("navWsCount").textContent = store.getCompare().length;
}

// ---- compare engine (shared compute + coefficient-diff helpers) -----
// computeResult / cmpOpOptions / cmpBuildFields / cmpRowGet drive both the
// Workspace tiles and the bottom tray. cmpComputeRow + cmpRows + cmpTimers are
// the retired standalone-compare path, kept only as a defensive fallback for
// the generalized cmpBuildFields listener.
let cmpRows = [];   // [{id, op, input, result}] — retired standalone path
const cmpTimers = {};

// Compute one operation and return a result-shaped object (or null on error),
// reusing the same /compute endpoint the per-operation panel calls.
async function computeResult(op, input, args){
  try {
    const res = await fetch("/compute", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ op, input, args: args || {} })
    });
    const data = await res.json();
    if (data.error) return null;
    return { op, input, args: args || {}, latex: data.latex,
             text: data.text, rows: data.rows || [], meta: data.meta || null };
  } catch(e){ return null; }
}
// The operation picker for a row: every tool, grouped, the current op selected.
function cmpOpOptions(selected){
  let html = "";
  for (const g of GROUPS){
    const ops = Object.entries(OPS).filter(([, m]) => m.group === g);
    if (!ops.length) continue;
    html += '<optgroup label="' + esc(g) + '">';
    for (const [key, m] of ops){
      html += '<option value="' + esc(key) + '"' +
        (key === selected ? " selected" : "") + '>' + esc(m.name) + '</option>';
    }
    html += '</optgroup>';
  }
  return html;
}
// Build the per-field controls for a row's chosen op, scoped to that row, and
// wire each one to a debounced recompute.
function cmpBuildFields(row, op, preset){
  const fields = (OPS[op] && OPS[op].fields) || [];
  let html = "";
  for (const f of fields){
    const val = (preset && preset[f.name] != null) ? String(preset[f.name]) : f.example;
    const cls = "cmp-f " + (f.name === "input" ? "cmp-f-input" : "cmp-f-arg");
    if (f.type === "select"){
      html += '<select class="' + cls + '" data-fname="' + esc(f.name) + '">';
      for (const c of f.choices){
        html += '<option value="' + esc(c.value) + '"' +
          (String(c.value) === String(val) ? " selected" : "") + '>' +
          esc(c.label) + '</option>';
      }
      html += '</select>';
    } else {
      const mode = (f.type === "int") ? ' inputmode="numeric"' : '';
      html += '<input class="' + cls + '" data-fname="' + esc(f.name) + '" type="text"' +
        mode + ' spellcheck="false" autocomplete="off" placeholder="' + esc(f.label) +
        '" value="' + esc(val) + '">';
    }
  }
  const box = row.querySelector(".cmp-row-fields");
  box.innerHTML = html;
  box.querySelectorAll("[data-fname]").forEach((el) => {
    const ev = (el.tagName === "SELECT") ? "change" : "input";
    el.addEventListener(ev, () => {
      clearTimeout(row._t);
      row._t = setTimeout(() => (row._compute ? row._compute() : cmpComputeRow(row.dataset.id)), 250);
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter"){ e.preventDefault(); (row._compute ? row._compute() : cmpComputeRow(row.dataset.id)); }
    });
  });
}
// Read {op, input, args} from a row's controls.
function cmpRowGet(row){
  const op = row.querySelector(".cmp-row-op").value;
  let input = "";
  const argv = {};
  row.querySelectorAll("[data-fname]").forEach((el) => {
    if (el.dataset.fname === "input") input = el.value.trim();
    else argv[el.dataset.fname] = el.value;
  });
  return { op, input, args: argv };
}
function cmpScheduleRow(id){
  clearTimeout(cmpTimers[id]);
  cmpTimers[id] = setTimeout(() => cmpComputeRow(id), 250);
}
// Fallback recompute for the generalized cmpBuildFields listener: when a node
// has no per-node `_compute` hook, compute by row id against the (now retired)
// standalone-compare list. Kept as a defensive fallback path.
async function cmpComputeRow(id){
  const row = $("cmpRows") && $("cmpRows").querySelector('[data-id="' + id + '"]');
  if (!row) return;
  const { op, input, args } = cmpRowGet(row);
  const rec = cmpRows.find((r) => r.id === id);
  if (rec){ rec.op = op; rec.input = input; }
  const out = row.querySelector(".cmp-row-out");
  if (!input){
    out.innerHTML = '<span class="cmp-err">enter an input</span>';
    if (rec) rec.result = null; cmpUpdateDiff(); return;
  }
  const res = await computeResult(op, input, args);
  if (!res){
    out.innerHTML = '<span class="cmp-err">could not compute</span>';
    if (rec) rec.result = null; cmpUpdateDiff(); return;
  }
  if (rec) rec.result = res;
  out.innerHTML = '\\[' + res.latex + '\\]';
  typeset(out);
  cmpUpdateDiff();
}

// Pull the integer coefficient list out of a coefficients result so the
// difference [x]_q - [y]_q can be formed in the browser.
function _coeffsOf(res){
  if (!res || res.op !== "coefficients" || !res.rows) return null;
  const row = res.rows.find((kv) => /coefficients/i.test(kv[0]));
  if (!row) return null;
  const xs = String(row[1]).split(",").map((s) => parseInt(s.trim(), 10));
  return xs.length && xs.every((v) => Number.isFinite(v)) ? xs : null;
}
function _laurentLatex(coeffs){
  const terms = [];
  coeffs.forEach((c, k) => {
    if (c === 0) return;
    const mag = Math.abs(c);
    const body = (k === 0) ? String(mag)
      : (mag === 1 ? "" : mag) + (k === 1 ? "q" : "q^{" + k + "}");
    terms.push((c < 0 ? "- " : "+ ") + body);
  });
  if (!terms.length) return "0";
  let s = terms.join(" ");
  if (s.startsWith("+ ")) s = s.slice(2);
  return s;
}
// With exactly two q-real coefficient rows, show [x]_q - [y]_q as a Laurent
// polynomial (exact to the shared truncation order). This is Charles's P(q).
// Rows running other tools simply do not contribute a coefficient list, so the
// difference quietly appears only when it is meaningful.
function cmpUpdateDiff(){
  const box = $("cmpDiff");
  if (cmpRows.length !== 2 || !cmpRows[0].result || !cmpRows[1].result){
    box.innerHTML = ""; return;
  }
  const a = cmpRows[0], b = cmpRows[1];
  const ca = _coeffsOf(a.result), cb = _coeffsOf(b.result);
  if (!ca || !cb){ box.innerHTML = ""; return; }
  const order = Math.min(ca.length, cb.length);
  const d = [];
  for (let k = 0; k < order; k++) d.push(ca[k] - cb[k]);
  const poly = _laurentLatex(d) + " + O(q^{" + order + "})";
  box.innerHTML = '<div class="cmp-card"><p class="cmp-name">[ ' + esc(a.input) +
    ' ]_q  -  [ ' + esc(b.input) + ' ]_q</p>' +
    '<div class="cmp-math">\\[' + poly + '\\]</div></div>';
  typeset(box);
}
// ---- bottom compare tray (per-profile, persistent) -------------------
function renderTray(){
  const tray = $("cmpTray");
  const items = store.getCompare();
  $("cmpTrayCount").textContent = items.length;
  updateWsCount();   // single source of truth: keep the nav "Workspace" badge in sync with the compare list
  if (!items.length){ tray.classList.add("hidden"); $("cmpTrayItems").innerHTML = ""; $("cmpTrayDiff").innerHTML = ""; return; }
  tray.classList.remove("hidden");
  $("cmpTrayItems").innerHTML = items.map((c) =>
    '<div class="cmp-chip" data-id="' + esc(c.id) + '">' +
      '<div class="cmp-chip-head"><span class="cmp-chip-name">' + esc(tileLabel(c)) + '</span>' +
      '<button class="cmp-chip-x" type="button" aria-label="remove">&times;</button></div>' +
      '<div class="cmp-chip-math">computing…</div>' +
      '<input class="cmp-chip-note" placeholder="note…" value="' + esc(c.note || "") + '">' +
    '</div>').join("");
  $("cmpTrayItems").querySelectorAll(".cmp-chip").forEach((chip) => {
    const id = chip.dataset.id;
    const c = items.find((x) => x.id === id);
    chip.querySelector(".cmp-chip-x").addEventListener("click", () => { store.removeCompare(id); renderTray(); });
    chip.querySelector(".cmp-chip-note").addEventListener("change", (e) => store.updateCompare(id, { note: e.target.value }));
    computeResult(c.op, c.input, c.args).then((res) => {
      const m = chip.querySelector(".cmp-chip-math");
      if (!res){ m.innerHTML = '<span style="color:var(--err)">error</span>'; return; }
      m.innerHTML = "\\[" + res.latex + "\\]"; typeset(m);
      chip._res = res; renderTrayDiff();
    });
  });
  renderTrayDiff();
}
function renderTrayDiff(){
  const box = $("cmpTrayDiff");
  const chips = Array.from($("cmpTrayItems").querySelectorAll(".cmp-chip"));
  if (chips.length !== 2 || !chips[0]._res || !chips[1]._res){ box.innerHTML = ""; return; }
  const ca = _coeffsOf(chips[0]._res), cb = _coeffsOf(chips[1]._res);
  if (!ca || !cb){ box.innerHTML = ""; return; }
  const order = Math.min(ca.length, cb.length), d = [];
  for (let k = 0; k < order; k++) d.push(ca[k] - cb[k]);
  box.innerHTML = '<div class="cmp-card"><div class="cmp-chip-math">\\[' +
    _laurentLatex(d) + " + O(q^{" + order + "})" + '\\]</div></div>';
  typeset(box);
}
// ---- navigation ------------------------------------------------------
function hideAllViews(){
  home.classList.add("hidden");
  opView.classList.add("hidden");
  savedView.classList.add("hidden");
  workspaceView.classList.add("hidden");
}
function goHome(){
  hideAllViews();
  home.classList.remove("hidden");
  currentOp = null;
  renderSavedInto($("savedHome"), false);
  renderTray();   // re-show the compare tray after leaving the workspace
  window.scrollTo(0, 0);
}
function goSaved(){
  hideAllViews();
  savedView.classList.remove("hidden");
  renderSavedInto($("savedFull"), true);
  window.scrollTo(0, 0);
}
function goWorkspace(){
  hideAllViews();
  workspaceView.classList.remove("hidden");
  // The workspace IS the expanded compare list, so the compact bottom tray is
  // redundant here — hide it (re-shown by renderTray on home/op views).
  $("cmpTray").classList.add("hidden");
  renderWorkspace();
  // figures sized while their section was hidden read as zero; fix on show
  resizeWorkspacePlots();
  window.scrollTo(0, 0);
}

$("brand").addEventListener("click", goHome);
$("backBtn").addEventListener("click", goHome);
$("backBtn2").addEventListener("click", goHome);
$("backBtn3").addEventListener("click", goHome);
$("navSaved").addEventListener("click", goSaved);
$("navWorkspace").addEventListener("click", goWorkspace);
$("navProfile").addEventListener("click", showStartup);

// ---- history drawer --------------------------------------------------
function renderHistory(){
  const box = $("historyList"); const items = store.getHistory();
  if (!items.length){ box.innerHTML = '<p class="drawer-empty">No history yet.</p>'; return; }
  box.innerHTML = items.map((h, i) =>
    '<div class="hist-row" data-i="' + i + '"><div class="hist-main"><b>' +
    esc(OPS[h.op] ? OPS[h.op].name : h.op) + '</b>: ' + esc(h.input != null ? String(h.input) : "") +
    ' <span class="meta">' + esc(fmtWhen(h.when)) + '</span></div>' +
    '<div class="hist-acts"><button class="mini" data-act="restore" type="button">Restore</button>' +
    '<button class="mini" data-act="save" type="button">Save</button>' +
    '<button class="mini" data-act="cmp" type="button">Compare</button>' +
    '<button class="mini" data-act="share" type="button">Share</button></div></div>').join("");
  box.querySelectorAll(".hist-row").forEach((row) => {
    const h = items[+row.dataset.i];
    row.querySelector('[data-act="restore"]').onclick = () => { closeHistory(); openOp(h.op, Object.assign({ input: h.input }, h.args || {})); };
    row.querySelector('[data-act="cmp"]').onclick = () => { store.addCompare({ op: h.op, input: h.input, args: h.args || {} }); renderTray(); toast("Added to compare"); };
    row.querySelector('[data-act="share"]').onclick = (e) => { e.stopPropagation(); openShareMenuFor([{ op: h.op, input: h.input, args: h.args || {}, note: "" }], e.currentTarget); };
    row.querySelector('[data-act="save"]').onclick = async () => {
      const res = await computeResult(h.op, h.input, h.args || {});
      if (res) saveResult({ op: h.op, input: h.input, args: h.args || {}, latex: res.latex, text: res.text, rows: res.rows, meta: res.meta });
    };
  });
}
// The drawer renders fresh from store.getHistory() every time it opens, so it
// always reflects the latest computes. Toggle so the nav button opens AND closes
// it; clicking anywhere outside also closes it (then re-open to see new entries).
function openHistory(){ renderHistory(); $("historyDrawer").classList.remove("hidden"); }
function closeHistory(){ $("historyDrawer").classList.add("hidden"); }
function toggleHistory(){ if ($("historyDrawer").classList.contains("hidden")) openHistory(); else closeHistory(); }
$("navHistory").addEventListener("click", (e) => { e.stopPropagation(); toggleHistory(); });
$("historyClose").addEventListener("click", closeHistory);
$("historyClear").addEventListener("click", () => { store.clearHistory(); renderHistory(); });
document.addEventListener("click", (e) => {
  const d = $("historyDrawer");
  if (!d.classList.contains("hidden") && !d.contains(e.target) && e.target.id !== "navHistory") closeHistory();
});

// ---- command palette (Cmd/Ctrl-K) -----------------------------------
function paletteActions(){
  const acts = [];
  for (const [k, m] of Object.entries(OPS)) acts.push({ label: "Open: " + m.name, run: () => openOp(k) });
  acts.push({ label: "Go: Workspace", run: goWorkspace });
  acts.push({ label: "Go: Home", run: goHome });
  acts.push({ label: "Switch profile…", run: showStartup });
  acts.push({ label: "Focus input", run: focusInput });
  return acts;
}
let _palMatches = [], _palSel = 0;
function renderPalette(q){
  const ql = q.toLowerCase();
  _palMatches = paletteActions().filter((a) => a.label.toLowerCase().includes(ql));
  if (_palSel >= _palMatches.length) _palSel = 0;
  $("paletteList").innerHTML = _palMatches.map((a, i) =>
    '<div class="palette-item' + (i === _palSel ? ' sel' : '') + '" data-i="' + i + '">' + esc(a.label) + '</div>').join("")
    || '<div class="palette-empty">No matches</div>';
  $("paletteList").querySelectorAll(".palette-item").forEach((el) =>
    el.onclick = () => { const a = _palMatches[+el.dataset.i]; closePalette(); if (a) a.run(); });
}
function openPalette(){ const p = $("palette"); p.classList.remove("hidden"); $("paletteInput").value = ""; _palSel = 0; renderPalette(""); setTimeout(() => $("paletteInput").focus(), 0); }
function closePalette(){ $("palette").classList.add("hidden"); }
$("paletteInput").addEventListener("input", (e) => renderPalette(e.target.value));
$("paletteInput").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown"){ e.preventDefault(); _palSel = Math.min(_palSel + 1, _palMatches.length - 1); renderPalette($("paletteInput").value); }
  else if (e.key === "ArrowUp"){ e.preventDefault(); _palSel = Math.max(_palSel - 1, 0); renderPalette($("paletteInput").value); }
  else if (e.key === "Enter"){ e.preventDefault(); const a = _palMatches[_palSel]; closePalette(); if (a) a.run(); }
  else if (e.key === "Escape"){ closePalette(); }
});
window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k"){ e.preventDefault(); openPalette(); return; }
  if (e.key === "Escape"){
    closePalette(); closeHistory();
    // Close the receive (shared comparison) modal if it is open.
    const rb = $("receiveBackdrop");
    if (rb && !rb.classList.contains("hidden")){
      rb.classList.remove("show"); setTimeout(() => rb.classList.add("hidden"), 280);
    }
    // Close the startup picker only when there's an active profile, so a
    // first-run user isn't left with no profile and no way back to the picker.
    if (store.active()){
      const cf = $("startupCreate");
      if (cf && !cf.classList.contains("hidden")){ hideCreateForm(); }
      else {
        const sb = $("startupBackdrop");
        if (sb && !sb.classList.contains("hidden")) hideStartup();
      }
    }
    return;
  }
  if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName||"")) && !e.target.isContentEditable){
    const p = $("palette"); if (p.classList.contains("hidden")){ e.preventDefault(); focusInput(); }
  }
});
// Segmented preview controls: mark the selected button in a group.
function paintSeg(groupId, attr, value){
  document.querySelectorAll("#" + groupId + " .seg-opt").forEach((b) =>
    b.classList.toggle("sel", b.dataset[attr] === value));
}
function segValue(groupId, attr, fallback){
  const b = document.querySelector("#" + groupId + " .seg-opt.sel");
  return (b && b.dataset[attr]) || fallback;
}
// Active-profile appearance (applies live, persists to the profile).
// The picker's current theme/size selection. Applies live even before a profile
// exists (first run / guest); persists to the active profile when there is one,
// and seeds Guest and the New-profile form so the choice is never a dead end.
let _pickAppearance = { theme: "light", fontSize: "m" };
$("apThemeBtns").addEventListener("click", (e) => {
  const b = e.target.closest("[data-theme]"); if (!b) return;
  _pickAppearance.theme = b.dataset.theme;
  applyAppearance(_pickAppearance);                       // live, with or without a profile
  if (store.active()) store.updateActive({ theme: b.dataset.theme });
  paintSeg("apThemeBtns", "theme", b.dataset.theme);
});
$("apSizeBtns").addEventListener("click", (e) => {
  const b = e.target.closest("[data-size]"); if (!b) return;
  _pickAppearance.fontSize = b.dataset.size;
  applyAppearance(_pickAppearance);
  if (store.active()) store.updateActive({ fontSize: b.dataset.size });
  paintSeg("apSizeBtns", "size", b.dataset.size);
});
function showCreateForm(){
  // Focused create step: hide the picker, the actions, the IO row, and the
  // active-profile appearance row so there is only ONE theme/size control.
  ["profileList", "startupActions", "profileIo", "appearanceRow"].forEach((id) => $(id).classList.add("hidden"));
  $("startupCreate").classList.remove("hidden");
  paintSeg("cpThemeBtns", "theme", _pickAppearance.theme);   // carry over the picker's choice
  paintSeg("cpSizeBtns", "size", _pickAppearance.fontSize);
  applyAppearance(_pickAppearance);                          // live preview matches the picker
  $("cpName").focus();
}
function hideCreateForm(){
  $("startupCreate").classList.add("hidden");
  ["profileList", "startupActions", "profileIo", "appearanceRow"].forEach((id) => $(id).classList.remove("hidden"));
  $("cpName").value = "";
  applyAppearance(store.active() || _pickAppearance);   // revert to the active profile, or the picked look
}
$("newProfileBtn").addEventListener("click", showCreateForm);
$("cpCancel").addEventListener("click", hideCreateForm);
$("cpThemeBtns").addEventListener("click", (e) => {
  const b = e.target.closest("[data-theme]"); if (!b) return;
  paintSeg("cpThemeBtns", "theme", b.dataset.theme);
  document.documentElement.dataset.theme = b.dataset.theme;   // live preview
});
$("cpSizeBtns").addEventListener("click", (e) => {
  const b = e.target.closest("[data-size]"); if (!b) return;
  paintSeg("cpSizeBtns", "size", b.dataset.size);
});
$("cpCreate").addEventListener("click", () => {
  const name = $("cpName").value.trim(); if (!name) { $("cpName").focus(); return; }
  store.createProfile({ name, theme: segValue("cpThemeBtns", "theme", "light"), fontSize: segValue("cpSizeBtns", "size", "m") });
  applyAppearance(store.active()); hideCreateForm(); hideStartup(); updateCount(); renderTray(); renderHome();
});
$("guestBtn").addEventListener("click", enterGuest);

// ---- profile JSON export/import (Gate 7) -----------------------------
$("exportProfileBtn").addEventListener("click", () => {
  const p = store.active(); if (!p){ toast("No active profile to export"); return; }
  _download((p.name || "profile").replace(/[^a-zA-Z0-9]/g,"_") + ".qrealsprofile.json", JSON.stringify(p), "application/json");
});
$("importProfileBtn").addEventListener("click", () => $("importProfileFile").click());
$("openSharedBtn").addEventListener("click", openSharedLink);
$("importProfileFile").addEventListener("change", (e) => {
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    let data; try { data = JSON.parse(reader.result); } catch(err){ toast("Invalid profile file"); return; }
    if (!data || typeof data.name !== "string" || !Array.isArray(data.saved) || !Array.isArray(data.compare)){ toast("Not a qreals profile file"); return; }
    const merge = store.active() && confirm('Merge into the current profile?  OK = merge, Cancel = create a new profile "' + data.name + '".');
    if (merge){
      store.setSaved(store.getSaved().concat(data.saved));
      store.setCompare(store.getCompare().concat(data.compare));
    } else {
      const id = store.createProfile({ name: data.name, theme: data.theme || "light", fontSize: data.fontSize || "m", saved: data.saved.slice() });
      store.setCompare(data.compare.slice());
      applyAppearance(store.active());
    }
    e.target.value = "";
    hideStartup(); updateCount(); renderTray(); renderHome();
    toast("Profile imported");
  };
  reader.readAsText(file);
});

$("cmpTrayClear").addEventListener("click", () => { store.setCompare([]); renderTray(); });
$("cmpTrayExpand").addEventListener("click", (e) => { e.stopPropagation(); goWorkspace(); });

// ---- share menu wiring (Gate 7) --------------------------------------
$("cmpTrayShare").addEventListener("click", (e) => { e.stopPropagation(); openShareMenuFor(null, e.currentTarget); });
$("wsShare").addEventListener("click", (e) => { e.stopPropagation(); openShareMenuFor(null, e.currentTarget); });
document.addEventListener("click", (e) => { const m = $("shareMenu"); if (!m.classList.contains("hidden") && !m.contains(e.target) && e.target.id !== "cmpTrayShare" && e.target.id !== "wsShare") m.classList.add("hidden"); });
$("shareMenu").addEventListener("click", async (e) => {
  const btn = e.target.closest(".share-item"); if (!btn) return;
  $("shareMenu").classList.add("hidden");
  const kind = btn.dataset.share;
  if (kind === "link"){ const link = await makeShareLink(_bundleFromCompare()); await navigator.clipboard.writeText(link); toast("Share link copied"); }
  else if (kind === "email"){ emailShare(); }
  else if (kind === "qreals"){ exportQreals(); }
  else if (kind === "pdf"){ exportPdf(); }
  else if (kind === "tex"){ exportTex(); }
  else if (kind === "overleaf"){ openInOverleaf(); }
  else if (kind === "bibtex"){ citeBibtex(); }
  else if (kind === "html"){ exportHtml(); }
});

// ---- profiles UI -----------------------------------------------------
function applyAppearance(profile){
  const theme = (profile && profile.theme) || "light";
  const size = (profile && profile.fontSize) || "m";
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.setProperty("--ui-scale",
    size === "s" ? "15px" : size === "l" ? "18px" : "16px");
}
function syncAppearanceControls(){
  const p = store.active();
  if (p) _pickAppearance = { theme: p.theme || "light", fontSize: p.fontSize || "m" };
  paintSeg("apThemeBtns", "theme", _pickAppearance.theme);
  paintSeg("apSizeBtns", "size", _pickAppearance.fontSize);
}
function showStartup(){
  const list = $("profileList");
  const profs = store.profiles();
  $("startupTitle").textContent = profs.length ? "Choose a profile" : "Welcome to qreals";
  list.innerHTML = profs.length ? profs.map((p) =>
    '<div class="profile-row" data-pid="' + esc(p.id) + '">' +
    '<span>' + esc(p.name) + '</span>' +
    '<span class="meta">' + (p.saved||[]).length + ' saved · ' + (p.compare||[]).length + ' compare' +
    ' <button class="mini danger profile-del" data-del="' + esc(p.id) + '">Delete</button></span>' +
    '</div>').join("")
    : '<div class="profile-empty">No profiles yet — create one to save your work.</div>';
  list.querySelectorAll("[data-pid]").forEach((row) =>
    row.addEventListener("click", () => enterProfile(row.dataset.pid)));
  list.querySelectorAll(".profile-del").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const id = b.dataset.del, p = store.state().profiles[id];
    if (!confirm("Delete profile \"" + (p ? p.name : id) + "\"? Its saved and compare lists will be lost.")) return;
    store.deleteProfile(id);
    showStartup();   // re-render the picker (deleting the active profile leaves no active profile)
  }));
  const b = $("startupBackdrop");
  b.classList.remove("hidden");
  requestAnimationFrame(() => b.classList.add("show"));
  syncAppearanceControls();
}
function hideStartup(){
  const b = $("startupBackdrop");
  b.classList.remove("show");
  setTimeout(() => b.classList.add("hidden"), 280);
}
function enterProfile(id){
  store.setActive(id);
  applyAppearance(store.active());
  syncAppearanceControls();
  hideStartup();
  updateCount();
  renderHome();
  renderTray();
  const ls = store.active() && store.active().lastSession;
  if (ls && ls.op && OPS[ls.op]){
    toast("Resumed your last computation");
    openOp(ls.op, Object.assign({ input: ls.input }, ls.args || {}));
  }
}
function enterGuest(){
  // ephemeral: deactivate any profile so nothing is persisted this session, but
  // keep the theme/size the user picked on the way in.
  store.clearActive();
  applyAppearance(_pickAppearance);
  hideStartup();
  updateCount();
  renderTray();
}

// ---- start -----------------------------------------------------------
$("toolSearch").addEventListener("input", (e) => filterCards(e.target.value.trim().toLowerCase()));
renderHome();
updateCount();
updateWsCount();
renderTray();
applyAppearance(store.active());
if (!store.active()) showStartup();    // first run / no active profile -> picker
maybeReceiveShare();
