// NASDAQ multi-target deck: main deck (cover + 9 slides) and a separate technical appendix (A1-A5).
// Statistical results only: no trading rules, P&L, costs or option strategies.
// Every number comes from deck_data.json (a copy of reports/multitarget_final/deck_data.json written
// by multitarget_report.py) and, for the appendix, sample_checks.json and timestral_same_sample.json
// (written by tools/sample_checks.py and tools/timestral_same_sample.py); charts are copied too.
// Build (Node.js 18+), from this folder: npm ci   (or npm install), then
//        node build_deck.js            -> NASDAQ_multitarget_report.pptx + NASDAQ_multitarget_appendix.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || "NASDAQ_multitarget_report.pptx";
const OUT_APPX = process.argv[3] || "NASDAQ_multitarget_appendix.pptx";
const J = (f) => JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8"));
const D = J("deck_data.json");
const SC = fs.existsSync(path.join(__dirname, "sample_checks.json")) ? J("sample_checks.json") : null;
const TSS = fs.existsSync(path.join(__dirname, "timestral_same_sample.json")) ? J("timestral_same_sample.json") : null;
const IMG = (f) => path.join(__dirname, f);
const REPO = "github.com/Paciooo/Project_ai_ML_bigdata";

const RED = "8E1728", RED_DARK = "5E0F1B", TINT = "F6E7E9", GOLD = "B8964B", GREEN = "2E6B3F", BLUE = "1F5FA8";
const INK = "1A1A1A", GREY = "7F7F7F", MID = "595959", LINE = "BFBFBF", PALE = "F4F4F4", GREEN_T = "E3F0E6", RED_T = "F6E0E2";
const W = 13.333, L = 0.55, R = W - 0.55, CW = R - L;
const FOOT = "© Alessandro Pacifici · Waseda University";

let pres;
function newPres(title) {
  const p = new pptxgen();
  p.layout = "LAYOUT_WIDE";
  p.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
  p.author = "Alessandro Pacifici";
  p.title = title;
  p.defineSlideMaster({
    title: "CONTENT", background: { color: "FFFFFF" },
    objects: [
      { line: { x: L, y: 1.38, w: CW, h: 0, line: { color: INK, width: 1 } } },
      { line: { x: L, y: 6.98, w: CW, h: 0, line: { color: INK, width: 1 } } },
      { text: { text: FOOT, options: { x: R - 4.5, y: 7.03, w: 4.5, h: 0.3, align: "right", fontSize: 9, bold: true, color: RED, margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: L, y: 0.32, w: CW, h: 0.55, fontSize: 24, bold: true,
                                  color: INK, margin: 0, valign: "bottom", align: "left" }, text: "" } },
      { placeholder: { options: { name: "subtitle", type: "body", x: L, y: 0.88, w: CW, h: 0.45, fontSize: 15, color: GREY,
                                  margin: 0, valign: "top", align: "left" }, text: "" } },
    ],
    slideNumber: { x: L, y: 7.03, w: 0.6, h: 0.3, fontSize: 9, color: GREY, margin: 0 },
  });
  p.defineSlideMaster({ title: "COVER", background: { color: RED } });
  return p;
}
function content(title, subtitle) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  s.addText(subtitle, { placeholder: "subtitle" });
  return s;
}
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: "Arial", color: INK, fontSize: 12,
                                                          margin: 0, valign: "top" }, o));
function header(s, x, y, w, text, color = MID) {
  T(s, text, { x: x + 0.05, y, w: w - 0.05, h: 0.3, fontSize: 12.5, bold: true, color, valign: "bottom" });
  s.addShape(pres.shapes.LINE, { x, y: y + 0.36, w, h: 0, line: { color: GREY, width: 1 } });
}
function bullets(s, items, o) {
  const runs = items.map((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    return parts.map((p, j) => ({ text: p.text, options: Object.assign({
      bullet: j === 0 ? { code: "25AA" } : undefined, color: INK,
      breakLine: j === parts.length - 1 && i < items.length - 1, paraSpaceAfter: 6 }, p.o || {}) }));
  }).flat();
  T(s, runs, Object.assign({ fontSize: 12 }, o));
}
const hdr = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, color: "FFFFFF", fill: { color: RED }, align: "center", valign: "middle" }, o) });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ valign: "middle" }, o) });
const num = (t, o = {}) => cell(t, Object.assign({ align: "center" }, o));
const topts = (x, y, w, colW, extra = {}) => Object.assign({ x, y, w, colW, fontFace: "Arial", fontSize: 11, color: INK,
  border: { type: "solid", pt: 0.75, color: LINE }, margin: [3, 5, 3, 5], rowH: 0.3 }, extra);
const note = (s, text, y, h = 0.42, x = L, w = CW) => T(s, text, { x, y, w, h, fontSize: 9.5, color: GREY });

// ---------------------------------------------------------------- data helpers
const FREQ = { "5min": "5 min", "15min": "15 min", "30min": "30 min", "1h": "1 hour", session: "1 day" };
const FREQS = ["5min", "15min", "30min", "1h", "session"].filter((f) => D.dataset[f]);
const MODEL = { zero: "Zero return", train_mean: "Training mean", majority: "Majority class", persistence: "Persistence",
  rolling_mean_22: "Rolling mean (22)", har_ols: "HAR-OLS", ridge: "Ridge", logistic: "Logistic", random_forest: "Random forest",
  xgboost: "XGBoost", "mlp_64-32-16": "DNN 64-32-16", "mlp_64-32": "DNN 64-32", "mlp_128-64-32": "DNN 128-64-32",
  "mlp_256-128-64-32": "DNN 256-128-64-32", timestral: "Timestral-5M (frozen weights)",
  timestral_calibrated: "Timestral-5M + calibration" };
const TASK = { return: "Return", direction: "Direction", volatility: "Volatility" };
const SEL = D.selected;
const byTask = (t) => SEL.filter((r) => r.task === t);
const f2 = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const pctv = (v, d = 1) => (100 * v).toFixed(d) + "%";
const ciTxt = (lo, hi, d = 2, u = "") => `[${f2(lo, d)}${u}, ${f2(hi, d)}${u}]`;
const sig = (r) => r.ci_lo > 0 ? "interval above zero" : r.ci_hi < 0 ? "interval below zero" : "interval includes zero";
const RET = byTask("return"), DIR = byTask("direction"), VOL = byTask("volatility");
const bestDir = DIR[0], bestRet = RET[0], bestVol = VOL[0];
const testRange = (f) => `${D.dataset[f].test[0]} → ${D.dataset[f].test[1]}`;
const PROF = (f, sp = "test") => D.data_profile.find((p) => p.frequency === f && p.split === sp);
const YEARS = (r) => D.direction_by_year[String(r.id)].filter((y) => y.year !== "all");
const kfmt = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k" : n.toLocaleString("en-US");
const acfRange = (k) => { const v = FREQS.map((f) => PROF(f)[k]); return [Math.min(...v), Math.max(...v)]; };
const CMP = D.comparison, WIN = D.winners, BEST = WIN[0];
const TMAP = (task, f) => D.timeframe_map.find((r) => r.task === task && r.frequency === f);
const DAG = (r) => r.model.startsWith("timestral") ? " *" : "";
const mName = (r) => MODEL[r.model] + DAG(r);
const zTxt = (r) => r.val_z.toFixed(1);
const rmseCut = (r) => 100 * (1 - r.t_rmse_log / r.t_rmse_log_persistence);
const tsNote = (task) => { const t = CMP[task].find((r) => r.model.startsWith("timestral"));
  return t ? `* Timestral: scored on its fixed random sample of ${t.t_n.toLocaleString("en-US")} test candles (its benchmark on the same candles; appendix A4).` : ""; };
const rankNote = "Rows ranked by validation evidence z (validation improvement over the task's naive benchmark ÷ its block-bootstrap standard error); all other columns are test results.";
const EXPLORATORY = "Exploratory: the test period (2021-07 → 2025-05; daily to 2025-01) was examined in earlier phases and the volatility selection score was revised after a first look at test results. Statistical accuracy only: no trading, costs or profitability were evaluated.";
const tmV = (f) => TMAP("volatility", f);

// ================================================================================ MAIN DECK
pres = newPres("What Can Machine Learning Predict in NASDAQ Markets?");

// ---------------------------------------------------------------- 1. Title (no links: repository on the last slide)
{
  const s = pres.addSlide({ masterName: "COVER" });
  T(s, "What Can Machine Learning Predict in NASDAQ Markets?", { x: 0.8, y: 1.75, w: 11.7, h: 1.5, fontSize: 38, bold: true, color: "FFFFFF", valign: "bottom" });
  T(s, "Price, Direction and Volatility Forecasting Across Models and Timeframes", { x: 0.8, y: 3.4, w: 11.7, h: 0.6, fontSize: 20, color: "F3D9DD" });
  s.addShape(pres.shapes.LINE, { x: 0.8, y: 4.25, w: 3.2, h: 0, line: { color: GOLD, width: 2 } });
  T(s, [{ text: "Alessandro Pacifici", options: { bold: true, breakLine: true } },
        { text: "AI, Machine Learning and Big Data for Finance and Management · Waseda University", options: { breakLine: true } },
        { text: "Final report · November 2026 · E-mini Nasdaq-100 futures (NQ), 2006–2025" }],
    { x: 0.8, y: 4.45, w: 11.7, h: 1.2, fontSize: 14, color: "FFFFFF" });
}

// ---------------------------------------------------------------- 2. Executive summary (stands alone)
{
  const s = content("Executive Summary", "Volatility is predictable on every horizon tested; returns and direction show no useful improvement");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: L, y: 1.52, w: CW, h: 0.46, rectRadius: 0.08, fill: { color: PALE }, line: { type: "none" } });
  T(s, [{ text: "Problem  ", options: { bold: true, color: RED } },
        { text: "Markets are hard to forecast, but are all their characteristics equally hard? For the next NQ futures candle we ask what machine learning can predict: the return, the direction or the realized volatility." }],
    { x: L + 0.2, y: 1.52, w: CW - 0.4, h: 0.46, fontSize: 11.5, valign: "middle" });
  const b = BEST, y21 = YEARS(bestDir);
  const early = y21.filter((y) => y.ci_lo_pp > 0).map((y) => y.year), late = y21.filter((y) => y.ci_lo_pp <= 0).map((y) => y.year);
  const har5 = tmV("5min"), har1h = tmV("1h");
  const cards = [
    { task: "Volatility", badge: "PREDICTABLE", bc: GREEN, fill: RED, ink: "FFFFFF", sub: "F3D9DD",
      big: `−${b.t_r2_log_vs_persistence_pct.toFixed(0)}%`,
      cap: `squared log-RV forecast error vs persistence (${MODEL[b.model]}, ${FREQ[b.frequency]})`,
      sec: `vs the HAR-OLS benchmark: −${har5.vs_har.toFixed(1)}% at 5 min, −${har1h.vs_har.toFixed(0)}% at 1 hour; rise or fall right ${pctv(b.t_voldir_acc, 0)} (majority ${pctv(b.t_voldir_majority_acc, 0)})` },
    { task: "Direction", badge: "MARGINAL", bc: GOLD, fill: PALE, ink: INK, sub: MID,
      big: `${f2(bestDir.t_diff_pp)} pp`,
      cap: `accuracy over the majority class: ${pctv(bestDir.t_accuracy, 2)} vs ${pctv(bestDir.t_majority_acc, 2)} (5 min)`,
      sec: `≈ ${Math.round(bestDir.t_diff_pp * 10)} extra right calls per 1,000 candles; seen in ${early[0]}–${early[early.length - 1]}, ${late.length ? "not in " + late.join("–") : "in every year"}; smaller or nil on longer candles` },
    { task: "Return", badge: "NOT DEMONSTRATED", bc: GREY, fill: PALE, ink: INK, sub: MID,
      big: `R² ${f2(bestRet.t_r2_oos_pct, 2)}%`,
      cap: `vs forecasting zero, 95% interval ${ciTxt(bestRet.ci_lo, bestRet.ci_hi, 2, "%")} (5 min)`,
      sec: "Every interval includes zero on all five candle lengths, from 5 minutes to 1 day" },
  ];
  const cw = (CW - 0.5) / 3, cy = 2.1, chh = 2.05;
  cards.forEach((c, i) => {
    const x = L + i * (cw + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: cy, w: cw, h: chh, rectRadius: 0.08, fill: { color: c.fill }, line: { color: i === 0 ? RED : LINE, width: 1 } });
    T(s, c.task, { x: x + 0.2, y: cy + 0.12, w: 1.7, h: 0.32, fontSize: 15, bold: true, color: i === 0 ? "FFFFFF" : RED, valign: "middle" });
    const bw = c.badge.length * 0.085 + 0.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + cw - bw - 0.18, y: cy + 0.15, w: bw, h: 0.27, rectRadius: 0.12, fill: { color: i === 0 ? "FFFFFF" : c.bc }, line: { type: "none" } });
    T(s, c.badge, { x: x + cw - bw - 0.18, y: cy + 0.15, w: bw, h: 0.27, fontSize: 9, bold: true, align: "center", valign: "middle", color: i === 0 ? GREEN : "FFFFFF" });
    T(s, c.big, { x: x + 0.2, y: cy + 0.48, w: cw - 0.4, h: 0.58, fontSize: 30, bold: true, color: c.ink, valign: "middle" });
    T(s, c.cap, { x: x + 0.2, y: cy + 1.06, w: cw - 0.4, h: 0.42, fontSize: 10.5, bold: true, color: c.ink });
    T(s, c.sec, { x: x + 0.2, y: cy + 1.5, w: cw - 0.4, h: 0.52, fontSize: 10, color: c.sub });
  });
  const yb = 4.33, hb = 2.2, wc = 4.55;
  T(s, "Why: volatility clusters, returns do not", { x: L, y: yb - 0.02, w: wc, h: 0.3, fontSize: 12, bold: true, color: RED });
  s.addChart(pres.charts.BAR, [
    { name: "next return", labels: FREQS.map((f) => FREQ[f]), values: FREQS.map((f) => PROF(f).ret_acf1) },
    { name: "log realized variance", labels: FREQS.map((f) => FREQ[f]), values: FREQS.map((f) => PROF(f).lrv_acf1) }],
    { x: L - 0.05, y: yb + 0.28, w: wc + 0.1, h: hb - 0.3, barDir: "col", barGapWidthPct: 60, chartColors: [GREY, RED],
      valAxisMinVal: -0.2, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFontSize: 9, catAxisLabelFontSize: 9,
      valAxisTitle: "lag-1 autocorrelation (test)", showValAxisTitle: true, valAxisTitleFontSize: 9, valAxisLabelFormatCode: "0.0",
      showLegend: true, legendPos: "t", legendFontSize: 9, valGridLine: { color: "E6E6E6", size: 0.5 } });
  const gx = L + wc + 0.3, gw = (CW - wc - 0.3 - 0.2) / 2, gh = (hb - 0.12) / 2;
  [["Data & method", `NQ futures, 1-minute data 2006–2025, ${FREQS.length} candle lengths (5 min–1 day); ${D.n_models_tested} models, benchmarks to deep networks and Mistral's Timestral. Train 2006–17, choose 2017–21, test 2021–25.`],
   ["Originality", "Experimental: one pipeline and test period for three targets and five horizons, compared like for like. Applied: ML and a new foundation model against the classical HAR benchmark."],
   ["Usefulness", "Volatility forecasts are inputs to risk limits and position sizing; the study shows on which horizons ML adds to HAR, and where it adds nothing."],
   ["Implication", "Use ML for realized volatility, not for next-candle returns or direction, with the models, data and horizons tried."]]
    .forEach(([h, t], i) => {
      const x = gx + (i % 2) * (gw + 0.2), y = yb + Math.floor(i / 2) * (gh + 0.12);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: gw, h: gh, fill: { color: i === 3 ? TINT : PALE }, line: { type: "none" } });
      T(s, [{ text: h, options: { bold: true, color: RED, fontSize: 12, breakLine: true } }, { text: t, options: { fontSize: 10 } }],
        { x: x + 0.15, y: y + 0.08, w: gw - 0.3, h: gh - 0.12 });
    });
  note(s, EXPLORATORY + " pp = percentage points, bp = 0.01%, RV = realized variance.", 6.58, 0.38);
}

// ---------------------------------------------------------------- 3. Data & methodology
{
  const s = content("Data & Methodology", "One instrument, five candle lengths, three targets, one chronological split");
  const steps = ["NASDAQ historical data\nNQ 1-min, 2006–2025", "Feature engineering\nlags, volatility, time of day", "Models\nbenchmarks · ML · DNN · Timestral", "Forecasts\nfor candle t+1", "Out-of-sample evaluation\ntest from 2021-07 (exploratory)"];
  const bw = (CW - 4 * 0.3) / 5;
  steps.forEach((t, i) => {
    const x = L + i * (bw + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: bw, h: 0.8, rectRadius: 0.06, fill: { color: i === 4 ? RED : TINT }, line: { color: RED, width: 1 } });
    const [a, b] = t.split("\n");
    T(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b, options: { fontSize: 10 } }],
      { x: x + 0.08, y: 1.6, w: bw - 0.16, h: 0.8, fontSize: 11.5, align: "center", valign: "middle", color: i === 4 ? "FFFFFF" : INK });
    if (i < 4) s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + bw + 0.08, y: 1.9, w: 0.15, h: 0.2, rotate: 90, fill: { color: RED }, line: { type: "none" } });
  });
  header(s, L, 2.6, 6.1, "Three targets for the next candle (t+1), decided at the close of t");
  const tg = [
    ["1  Next return", "r = log(C(t+1)/C(t))", "R² vs the zero forecast"],
    ["2  Next-candle direction", "UP if C(t+1) > O(t+1); doji = miss", "accuracy vs the majority class"],
    ["3  Realized volatility", "RV = Σ (1-min log returns)² in t+1", "log-RV error vs persistence and HAR-OLS"],
  ];
  s.addTable([[hdr("Target"), hdr("Definition"), hdr("Main metric")], ...tg.map(([a, b, c]) => [cell(a, { bold: true }), cell(b), cell(c)])],
    topts(L, 3.07, 6.1, [1.9, 2.25, 1.95], { fontSize: 10.5, rowH: 0.42 }));
  bullets(s, [
    [{ text: "Data: ", o: { bold: true } }, { text: "E-mini Nasdaq-100 futures (NQ, CME), back-adjusted 1-minute series. No other market, no VIX." }],
    [{ text: "Split: ", o: { bold: true } }, { text: `train → ${D.dataset["5min"].train[1]}, validation → ${D.dataset["5min"].validation[1]}, test ${testRange("5min")} (daily test ends ${D.dataset.session.test[1]}, appendix A3).` }],
    [{ text: "No look-ahead: ", o: { bold: true } }, { text: "features use candles ≤ t; scaling fitted on training data; labels never cross a session or a split." }],
    [{ text: "Uncertainty: ", o: { bold: true } }, { text: "95% intervals from a moving-block bootstrap in 20-session blocks (Künsch, 1989)." }],
  ], { x: L, y: 4.95, w: 6.1, h: 1.95, fontSize: 10.5 });
  const x2 = L + 6.45, w2 = CW - 6.45;
  header(s, x2, 2.6, w2, "Data profile, test sample");
  const rows = FREQS.map((f) => { const p = PROF(f);
    return [num(FREQ[f], { bold: true }), num(kfmt(D.dataset[f].test[2])), num(`${pctv(p.share_up, 0)} / ${pctv(p.share_down, 0)}`),
      num(f2(p.ret_acf1, 3), { color: MID }), num(f2(p.lrv_acf1), { bold: true, color: RED })]; });
  s.addTable([[hdr("Candle"), hdr("Test rows"), hdr("UP / DOWN"), hdr("ACF return"), hdr("ACF log RV")], ...rows],
    topts(x2, 3.07, w2, [1.0, 1.1, 1.3, 1.15, w2 - 4.55], { fontSize: 10.5, rowH: 0.36 }));
  const [ra, rb] = acfRange("ret_acf1"), [va, vb] = acfRange("lrv_acf1");
  T(s, [{ text: "Reading: ", options: { bold: true } },
        { text: `candle direction is close to a coin flip; the lag-1 autocorrelation (ACF) of returns is about zero (${f2(ra, 3)} to ${f2(rb, 3)}), while log realized variance is strongly persistent (${va.toFixed(2)} to ${vb.toFixed(2)}).` }],
    { x: x2, y: 5.4, w: w2, h: 0.8, fontSize: 10.5 });
  note(s, `${D.dataset[FREQS[0]].features.length} features: past returns, momentum, rolling volatility, candle shape, RSI, moving-average gaps, volume, time of day, past log RV (1, 5, 22 candles). Timestral sees the last 512 closes.`, 6.25, 0.6, x2, w2);
}

// ---------------------------------------------------------------- 4. Task x timeframe map
{
  const s = content("What Is Predictable, and on Which Horizon?", "Only volatility beats its benchmark on every horizon; its gain over HAR-OLS grows with candle length");
  const fill = (lo, hi) => lo > 0 ? GREEN_T : hi < 0 ? RED_T : PALE;
  const SHORT = (m) => MODEL[m].replace("DNN ", "DNN ");
  const c = (txt, model, lo, hi, d) => cell([{ text: txt, options: { bold: true, breakLine: true, fontSize: 11 } },
      { text: `${ciTxt(lo, hi, d)}`, options: { fontSize: 8.5, color: MID, breakLine: true } },
      { text: SHORT(model), options: { fontSize: 8, color: GREY } }], { align: "center", fill: { color: fill(lo, hi) } });
  const rowsDef = [
    ["Return\nR² vs zero (%)", (f) => { const r = TMAP("return", f); return c(`${f2(r.value, 2)}%`, r.model, r.ci_lo, r.ci_hi, 2); }],
    ["Direction\naccuracy gain vs majority (pp)", (f) => { const r = TMAP("direction", f); return c(`${f2(r.value, 2)} pp`, r.model, r.ci_lo, r.ci_hi, 2); }],
    ["Volatility\nsquared log-RV error cut vs persistence (%)", (f) => { const r = tmV(f); return c(`${f2(r.value, 1)}%`, r.model, r.ci_lo, r.ci_hi, 1); }],
    ["Volatility\nsquared log-RV error cut vs HAR-OLS (%)", (f) => { const r = tmV(f); return c(`${f2(r.vs_har, 1)}%`, r.model, r.vs_har_lo, r.vs_har_hi, 1); }],
  ];
  const body = [[hdr("Task / metric (test)"), ...FREQS.map((f) => hdr(`${FREQ[f]}\n${kfmt(TMAP("return", f).n_test)} candles`))],
    ...rowsDef.map(([lab, fn]) => { const [a, b] = lab.split("\n");
      return [cell([{ text: a, options: { bold: true, color: RED, breakLine: true } }, { text: b, options: { fontSize: 9, color: MID } }]), ...FREQS.map(fn)]; })];
  s.addTable(body, topts(L, 1.52, CW, [2.4, ...FREQS.map(() => (CW - 2.4) / FREQS.length)], { fontSize: 10, rowH: 0.66 }));
  // legend
  [[GREEN_T, "95% interval above zero"], [PALE, "interval includes zero"], [RED_T, "interval below zero"]].forEach(([col, t], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: L + i * 2.6, y: 5.03, w: 0.25, h: 0.18, fill: { color: col }, line: { color: LINE, width: 0.5 } });
    T(s, t, { x: L + i * 2.6 + 0.32, y: 5.0, w: 2.2, h: 0.25, fontSize: 9.5, color: MID });
  });
  const vs = FREQS.map((f) => tmV(f).vs_har);
  s.addChart(pres.charts.BAR, [{ name: "error cut vs HAR-OLS (%)", labels: FREQS.map((f) => FREQ[f]), values: vs }],
    { x: L + 7.7, y: 5.25, w: CW - 7.7, h: 1.7, barDir: "col", chartColors: [RED], showValue: true, dataLabelFormatCode: "0.0",
      dataLabelFontSize: 9, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 9, showLegend: false,
      showTitle: true, title: "Volatility: squared log-RV error cut vs HAR-OLS (%)", titleFontSize: 10, titleColor: MID });
  bullets(s, [
    [{ text: "Volatility: ", o: { bold: true } }, { text: `every horizon beats persistence (squared log-RV error ${Math.min(...FREQS.map((f) => tmV(f).value)).toFixed(0)}% to ${Math.max(...FREQS.map((f) => tmV(f).value)).toFixed(0)}% lower). The extra gain over HAR-OLS rises from ${tmV("5min").vs_har.toFixed(1)}% (5 min) to ${tmV("1h").vs_har.toFixed(1)}% (1 hour); on daily candles the interval includes zero.` }],
    [{ text: "Return and direction: ", o: { bold: true } }, { text: "no interval is above zero except 5-minute direction (+0.69 pp)." }],
  ], { x: L, y: 5.35, w: 7.4, h: 1.55, fontSize: 10.5 });
  note(s, "Each cell: the trained model with the highest validation z for that task and candle length, scored on the test. Daily sample: 642 test sessions (appendix A3).", 6.62, 0.32, L, 7.4);
}

// ---------------------------------------------------------------- 5. All models tested
{
  const s = content("All Models Tested", `${D.n_models_tested} models on ${FREQS.length} timeframes: ${D.n_configs} model–timeframe–task configurations, all run on the same data`);
  const fam = [
    ["Baselines", [["Zero return / training mean", "naive forecast", "R"], ["Majority class", "most frequent training class", "D"], ["Persistence · rolling mean (22)", "RV(t+1) = RV(t) · average past variance", "V"]]],
    ["Statistical", [["HAR-OLS", "regression on past log RV (1, 5, 22 candles) — Corsi (2009)", "V"], ["Ridge / logistic regression", "L2-penalised, strength chosen on validation", "R · D · V"]]],
    ["Machine learning", [["Random forest", "200 trees, depth 8", "R · D · V"], ["XGBoost", "gradient boosting, early stopping", "R · D · V"]]],
    ["Deep learning", [["DNN: 64-32-16 · 64-32 · 128-64-32 · 256-128-64-32", "feed-forward, ReLU, AdamW, early stopping", "R · D · V"]]],
    ["Mistral AI", [["Timestral-5M, frozen weights", "pretrained forecaster of the next close (zero-shot)", "R · D"],
                    ["Timestral-5M + calibration", "forecast spread → log RV, linear map fitted on validation", "V"]]],
  ];
  const body = [[hdr("Family"), hdr("Model"), hdr("Method"), hdr("Tasks")]];
  fam.forEach(([f, ms]) => ms.forEach((m, i) => body.push([
    i === 0 ? cell(f, { bold: true, color: RED, rowspan: ms.length, fill: { color: PALE } }) : null,
    cell(m[0], { bold: true }), cell(m[1], { color: MID }), num(m[2])].filter((c) => c !== null))));
  s.addTable(body, topts(L, 1.55, 8.6, [1.45, 3.0, 3.3, 0.85], { fontSize: 10.5, rowH: 0.42 }));
  const x2 = L + 8.9, w2 = CW - 8.9;
  header(s, x2, 1.5, w2, "Notes");
  bullets(s, [
    [{ text: "R", o: { bold: true } }, { text: " return · " }, { text: "D", o: { bold: true } }, { text: " direction · " }, { text: "V", o: { bold: true } }, { text: " volatility" }],
    [{ text: "Mistral's role: ", o: { bold: true } }, { text: "Timestral is a pretrained time-series model (local weights, no API); it is not a language model and reads no text." }],
    [{ text: "Not run: ", o: { bold: true } }, { text: "LSTM; larger Timestral sizes (28M–209M)." }],
    [{ text: "Every fit and hyper-parameter: ", o: { bold: true } }, { text: "reports/multitarget_final/models.csv." }],
  ], { x: x2, y: 1.98, w: w2, h: 4.9, fontSize: 11 });
}

// ---------------------------------------------------------------- 6. Return forecasting
{
  const C = CMP.return, tf = FREQ[C[0].frequency];
  const s = content("Price / Return Forecasting", `Five models on ${tf} candles: none improves on the zero-return forecast`);
  const rows = C.map((r) => [num(String(r.rank_in_task), { bold: true }), cell(mName(r), { bold: true }), num(zTxt(r)),
    num(r.t_rmse_bps.toFixed(2)), num(`${f2(r.t_r2_oos_pct, 2)}%`, { bold: true, color: r.ci_lo > 0 ? GREEN : r.ci_hi < 0 ? RED : INK }),
    num(ciTxt(r.ci_lo, r.ci_hi, 2)), num(pctv(r.t_dir_acc))]);
  s.addTable([[hdr("#"), hdr("Model"), hdr("Val. z"), hdr("RMSE bp"), hdr("R² vs zero"), hdr("95% interval (%)"), hdr("Sign right")], ...rows],
    topts(L, 1.55, 6.75, [0.3, 2.45, 0.6, 0.75, 0.85, 1.15, 0.65], { fontSize: 10, rowH: 0.36 }));
  note(s, `Benchmark (zero-return forecast): RMSE ${C[0].t_rmse_zero_bps.toFixed(2)} bp on all ${tf} test candles. ${tsNote("return")}`, 3.95, 0.45, L, 6.75);
  s.addImage({ path: IMG(D.charts[String(bestRet.id)]), x: L + 7.05, y: 1.5, w: CW - 7.05, h: (CW - 7.05) * 3.6 / 5.6 });
  bullets(s, [
    [{ text: "Result: ", o: { bold: true } }, { text: `the best R² is ${f2(Math.max(...C.map((r) => r.t_r2_oos_pct)), 2)}%; every interval includes or lies below zero, so no model beats "the next return is zero". The same holds on all five candle lengths (slide 4).` }],
    [{ text: "Predicted vs realized (right, #1): ", o: { bold: true } }, { text: `realized return per forecast decile: forecasts span less than ±0.5 bp while a typical candle moves about ${C[0].t_rmse_zero_bps.toFixed(1)} bp.` }],
    [{ text: "Reading: ", o: { bold: true } }, { text: "linear, tree, deep and foundation models fail in the same way: no usable return signal with the models, data and horizons tried (consistent with, but not proof of, a random walk)." }],
  ], { x: L, y: 4.55, w: 6.75, h: 2.35, fontSize: 11 });
  note(s, `${rankNote} R² interval: model and zero-forecast errors resampled together. Test ${testRange(C[0].frequency)}.`, 5.45, 1.0, L + 7.05, CW - 7.05);
}

// ---------------------------------------------------------------- 7. Direction classification
{
  const C = CMP.direction, c = bestDir, yrs = YEARS(c), tf = FREQ[C[0].frequency];
  const pos = yrs.filter((y) => y.ci_lo_pp > 0).map((y) => y.year), zero = yrs.filter((y) => y.ci_lo_pp <= 0 && y.ci_hi_pp >= 0).map((y) => y.year);
  const nAbove = C.filter((r) => r.ci_lo > 0).length;
  const s = content("Next-Candle Direction Classification", `Five models on ${tf} candles: ${nAbove} of 5 beat the majority class, by less than one percentage point`);
  const rows = C.map((r) => [num(String(r.rank_in_task), { bold: true }), cell(mName(r), { bold: true }), num(zTxt(r)), num(pctv(r.t_accuracy, 2), { bold: true }),
    num(pctv(r.t_majority_acc, 2)), num(`${f2(r.t_diff_pp)} pp`, { bold: true, color: r.ci_lo > 0 ? GREEN : r.ci_hi < 0 ? RED : INK }),
    num(ciTxt(r.ci_lo, r.ci_hi)), num(pctv(r.t_balanced_acc, 2)), num(r.t_f1_up.toFixed(3))]);
  s.addTable([[hdr("#"), hdr("Model"), hdr("Val. z"), hdr("Accuracy"), hdr("Majority"), hdr("Gain"), hdr("95% interval"), hdr("Balanced acc."), hdr("F1 UP")], ...rows],
    topts(L, 1.5, CW, [0.35, 3.0, 0.8, 1.15, 1.15, 1.2, 1.6, 1.4, CW - 10.65], { fontSize: 10.5, rowH: 0.33 }));
  note(s, `${tsNote("direction")} Accuracy counts dojis (open = close) as misses.`, 3.55, 0.27);
  const ty = 3.9;
  header(s, L, ty, 5.4, `#1 ${MODEL[c.model]}: confusion matrix (test)`);
  const n = c.t_tp + c.t_fp + c.t_tn + c.t_fn, sh = (v) => `${v.toLocaleString("en-US")}  (${pctv(v / n)})`;
  s.addTable([[cell(""), hdr("Actual UP"), hdr("Actual DOWN")],
              [hdr("Predicted UP"), num(sh(c.t_tp), { bold: true, fill: { color: TINT } }), num(sh(c.t_fp))],
              [hdr("Predicted DOWN"), num(sh(c.t_fn)), num(sh(c.t_tn), { bold: true, fill: { color: TINT } })]],
    topts(L, ty + 0.48, 5.4, [1.5, 1.95, 1.95], { fontSize: 10.5, rowH: 0.36 }));
  bullets(s, [
    [{ text: "Size: ", o: { bold: true } }, { text: `${f2(c.t_diff_pp)} pp ≈ ${Math.round(c.t_diff_pp * 10)} more right calls per 1,000 candles. Tree, linear and deep models are within a few tenths of each other; Timestral does not beat the majority class.` }],
  ], { x: L, y: ty + 1.68, w: 5.4, h: 1.2, fontSize: 11 });
  const cx = L + 5.75, cw2 = CW - 5.75, chh = 6.9 - ty - 0.6, cww = Math.min(cw2, chh * 6.4 / 3.6);
  s.addImage({ path: IMG(D.charts[String(c.id)]), x: cx + (cw2 - cww) / 2, y: ty - 0.05, w: cww, h: cww * 3.6 / 6.4 });
  T(s, [{ text: "Stability of #1 by test year: ", options: { bold: true } }, { text: `gain above zero in ${pos.join(", ") || "no year"}${zero.length ? `; interval includes zero in ${zero.join(", ")}` : ""}.` }],
    { x: cx, y: 6.42, w: cw2, h: 0.45, fontSize: 11 });
}

// ---------------------------------------------------------------- 8. Volatility
{
  const C = CMP.volatility, b = C[0], tf = FREQ[b.frequency];
  const s = content("Volatility Forecasting", `Five models on ${tf} candles: all beat persistence; tree and deep models also beat the HAR-OLS benchmark`);
  const ci1 = (v, lo, hi, d = 1, u = "%") => `${f2(v, d)}${u} ${ciTxt(lo, hi, d)}`;
  const g = (v, lo) => ({ bold: true, color: lo > 0 ? GREEN : v < 0 ? RED : INK });
  const rows = C.map((r) => [num(String(r.rank_in_task), { bold: true }), cell(mName(r), { bold: true }), num(zTxt(r)),
    num(ci1(r.t_r2_log_vs_persistence_pct, r.ci_lo, r.ci_hi), g(r.t_r2_log_vs_persistence_pct, r.ci_lo)),
    num(ci1(r.t_r2_log_vs_har_pct, r.t_r2_log_vs_har_ci_lo_pct, r.t_r2_log_vs_har_ci_hi_pct), g(r.t_r2_log_vs_har_pct, r.t_r2_log_vs_har_ci_lo_pct)),
    num(r.t_rmse_log.toFixed(3)), num(pctv(r.t_voldir_acc), { bold: true }),
    num(ci1(r.t_voldir_diff_vs_har_pp, r.t_voldir_vs_har_ci_lo_pp, r.t_voldir_vs_har_ci_hi_pp, 2, " pp"), g(r.t_voldir_diff_vs_har_pp, r.t_voldir_vs_har_ci_lo_pp))]);
  s.addTable([[hdr("#"), hdr("Model"), hdr("Val. z"), hdr("Error cut vs persistence [95%]"), hdr("Error cut vs HAR-OLS [95%]"),
               hdr("RMSE log RV"), hdr("RV up/down right"), hdr("Up/down vs HAR-OLS [95%]")], ...rows],
    topts(L, 1.5, CW, [0.3, 2.6, 0.65, 2.15, 2.15, 1.0, 1.05, CW - 9.9], { fontSize: 10, rowH: 0.34 }));
  note(s, `Benchmarks on all ${tf} test candles: persistence (next RV = current RV) RMSE ${b.t_rmse_log_persistence.toFixed(3)}, up/down majority ${pctv(b.t_voldir_majority_acc)}; HAR-OLS RMSE ${b.har_rmse_log.toFixed(3)}, up/down ${pctv(b.t_voldir_acc_har)}. ${tsNote("volatility")}`, 3.62, 0.45);
  const yb = 4.2, chh = 6.45 - yb, cww = chh * 5.6 / 3.6;
  s.addImage({ path: IMG(D.charts[String(bestVol.id)]), x: R - cww, y: yb - 0.05, w: cww, h: chh });
  const tbest = C.reduce((a, r) => r.t_r2_log_vs_persistence_pct > a.t_r2_log_vs_persistence_pct ? r : a);
  bullets(s, [
    [{ text: "Metric: ", o: { bold: true } }, { text: `"error cut" = reduction of the squared log-RV forecast error vs a benchmark (an out-of-sample R²). For #1 it is ${b.t_r2_log_vs_persistence_pct.toFixed(1)}% vs persistence; the RMSE itself falls by ${rmseCut(b).toFixed(1)}% (${b.t_rmse_log.toFixed(3)} vs ${b.t_rmse_log_persistence.toFixed(3)}).` }],
    [{ text: "HAR-OLS is the bar to clear: ", o: { bold: true } }, { text: `it already captures clustering and the time-of-day pattern, so the cut vs HAR-OLS (${b.t_r2_log_vs_har_pct.toFixed(1)}% for #1) is what ML adds.${tbest.model !== b.model ? ` On the test ${MODEL[tbest.model]} (#${tbest.rank_in_task} by validation z) is slightly ahead.` : ""}` }],
    [{ text: "Scope: ", o: { bold: true } }, { text: "realized volatility from 1-minute returns (Andersen & Bollerslev, 1998), not implied volatility; no option prices or option strategies were studied." }],
  ], { x: L, y: yb, w: CW - cww - 0.3, h: 6.45 - yb, fontSize: 10.5 });
  note(s, `${rankNote} Volatility score revised from QLIKE (Patton, 2011) to log-RV error after a first look at test numbers: not blind.`, 6.5, 0.42);
}

// ---------------------------------------------------------------- 9. Strongest evidence + implications
{
  const b = BEST;
  const s = content("Which NASDAQ Market Characteristic Is Most Predictable?", `Strongest evidence of improvement: ${FREQ[b.frequency]} realized volatility, ${MODEL[b.model]}`);
  header(s, L, 1.5, 6.1, "How the configurations were ranked");
  const steps = [
    ["1", "Each model × timeframe is scored on VALIDATION data (2017–2021) by its improvement over its task's naive benchmark."],
    ["2", "z = that improvement ÷ its bootstrap standard error: the strength of evidence against that benchmark, not a common scale of usefulness across tasks."],
    ["3", "The highest z wins within each task (slides 6–8); across tasks z only says where the evidence of beating the naive benchmark is strongest."],
    ["4", "Winners are then scored on the TEST block. The volatility score was revised after a first look at test numbers: the ranking is exploratory."],
  ];
  steps.forEach(([k, t], i) => {
    const y = 1.95 + i * 0.6;
    s.addShape(pres.shapes.OVAL, { x: L, y: y + 0.06, w: 0.36, h: 0.36, fill: { color: RED }, line: { type: "none" } });
    T(s, k, { x: L, y: y + 0.06, w: 0.36, h: 0.36, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    T(s, t, { x: L + 0.5, y, w: 5.6, h: 0.6, fontSize: 10.5, valign: "middle" });
  });
  const res = (r) => r.task === "volatility" ? `${f2(r.t_r2_log_vs_persistence_pct, 1)}% error cut vs persistence` : r.task === "direction" ? `${f2(r.t_diff_pp)} pp vs majority` : `R² ${f2(r.t_r2_oos_pct, 2)}% vs zero`;
  s.addTable([[hdr("Task"), hdr("Task winner (validation)"), hdr("Val. z"), hdr("Test vs naive benchmark")],
    ...WIN.map((r) => [cell(TASK[r.task], { bold: true, color: RED }), cell(`${MODEL[r.model]}, ${FREQ[r.frequency]}`),
      num(r.val_z.toFixed(1), { bold: true }), cell(`${res(r)} ${ciTxt(r.ci_lo, r.ci_hi, r.task === "volatility" ? 1 : 2)}`, { color: r.ci_lo > 0 ? GREEN : INK })])],
    topts(L, 4.45, 6.1, [0.95, 2.0, 0.6, 2.55], { fontSize: 9.5, rowH: 0.34 }));
  const x2 = L + 6.45, w2 = CW - 6.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x2, y: 1.5, w: w2, h: 4.35, rectRadius: 0.08, fill: { color: RED }, line: { type: "none" } });
  T(s, "STRONGEST EVIDENCE OF IMPROVEMENT (HIGHEST VALIDATION z)", { x: x2 + 0.3, y: 1.62, w: w2 - 0.6, h: 0.3, fontSize: 10.5, bold: true, color: "F3D9DD" });
  T(s, `${MODEL[b.model]} · ${FREQ[b.frequency]} realized volatility`, { x: x2 + 0.3, y: 1.92, w: w2 - 0.6, h: 0.45, fontSize: 18, bold: true, color: "FFFFFF" });
  T(s, `−${b.t_r2_log_vs_persistence_pct.toFixed(1)}%`, { x: x2 + 0.3, y: 2.42, w: 2.6, h: 0.8, fontSize: 38, bold: true, color: "FFFFFF", valign: "middle" });
  T(s, `squared log-RV forecast error vs persistence on the test, 95% interval ${ciTxt(-b.ci_hi, -b.ci_lo, 1, "%")}; validation z ${b.val_z.toFixed(1)}`,
    { x: x2 + 2.95, y: 2.45, w: w2 - 3.25, h: 0.8, fontSize: 10.5, color: "F3D9DD", valign: "middle" });
  const kpi = [[`−${b.t_r2_log_vs_har_pct.toFixed(1)}%`, `squared error vs HAR-OLS ${ciTxt(-b.t_r2_log_vs_har_ci_hi_pct, -b.t_r2_log_vs_har_ci_lo_pct, 1)}`],
               [`−${rmseCut(b).toFixed(1)}%`, `RMSE of log RV vs persistence (${b.t_rmse_log.toFixed(3)} vs ${b.t_rmse_log_persistence.toFixed(3)})`],
               [pctv(b.t_voldir_acc), `RV up/down right (majority ${pctv(b.t_voldir_majority_acc)}, HAR-OLS ${pctv(b.t_voldir_acc_har)})`]];
  const kw = (w2 - 0.6 - 0.3) / 3;
  kpi.forEach(([v, t], i) => {
    const x = x2 + 0.3 + i * (kw + 0.15);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.4, w: kw, h: 1.2, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { type: "none" } });
    T(s, v, { x: x + 0.1, y: 3.45, w: kw - 0.2, h: 0.45, fontSize: 18, bold: true, color: RED, align: "center", valign: "middle" });
    T(s, t, { x: x + 0.1, y: 3.92, w: kw - 0.2, h: 0.65, fontSize: 9, color: INK, align: "center" });
  });
  const tb = CMP.volatility.reduce((a, r) => r.t_r2_log_vs_persistence_pct > a.t_r2_log_vs_persistence_pct ? r : a);
  T(s, (tb.model !== b.model ? `On the test ${MODEL[tb.model]} (#2 by validation z) cuts the error slightly more (−${tb.t_r2_log_vs_persistence_pct.toFixed(1)}%); the ranking is kept as computed on validation, not re-chosen on the test. ` : "")
      + `Longer candles show larger gains over HAR-OLS (up to −${tmV("1h").vs_har.toFixed(1)}% at 1 hour, slide 4).`,
    { x: x2 + 0.3, y: 4.72, w: w2 - 0.6, h: 1.05, fontSize: 10, color: "FFFFFF", valign: "middle" });
  T(s, [{ text: "Implications: ", options: { bold: true, color: RED_DARK } },
        { text: "realized volatility is the most predictable characteristic here, useful as an input to risk limits and sizing; for returns no improvement was demonstrated, and the direction gain is small and fades after 2023. " },
        { text: "Limits: ", options: { bold: true, color: RED_DARK } },
        { text: "exploratory test; one market; statistical accuracy only — commissions, spread and latency were not modelled, so economic value (e.g. of a +0.69 pp direction gain) is unknown." }],
    { x: L, y: 6.05, w: CW, h: 0.85, fontSize: 10.5 });
}

// ---------------------------------------------------------------- 10. Reproducibility & references
{
  const s = content("Reproducibility & References", "Every table, chart and slide is regenerated from code in the repository; the licensed NQ data are not published");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: L, y: 1.55, w: CW, h: 0.72, rectRadius: 0.08, fill: { color: RED }, line: { type: "none" } });
  T(s, "Code, results and this deck", { x: L + 0.3, y: 1.55, w: 3.2, h: 0.72, fontSize: 13, bold: true, color: "F3D9DD", valign: "middle" });
  T(s, [{ text: `https://${REPO}`, options: { hyperlink: { url: `https://${REPO}` }, color: "FFFFFF", bold: true } },
        { text: "   ·   branch main", options: { color: "F3D9DD" } }],
    { x: L + 3.4, y: 1.55, w: CW - 3.7, h: 0.72, fontSize: 18, valign: "middle" });
  const w1 = 6.6;
  header(s, L, 2.45, w1, "How to reproduce (details: appendix A1)");
  const steps = [["Environment", "Python 3.12 + requirements-lock.txt, Node.js 18+"],
                 ["Data", "NQ 1-minute bars 2006–2025 (licensed, not in Git): set FUTURES_DATA_DIR"],
                 ["Run", "multitarget.py → multitarget_report.py → verify_multitarget.py"],
                 ["Deck", "presentation: npm ci; node build_deck.js"]];
  s.addTable(steps.map(([a, b]) => [cell(a, { bold: true, color: RED }), cell(b)]), topts(L, 2.92, w1, [1.4, w1 - 1.4], { fontSize: 10.5, rowH: 0.36 }));
  bullets(s, [
    [{ text: "Checked: ", o: { bold: true } }, { text: "verify_multitarget.py recomputes every reported point estimate, the rankings and the deck data from the saved forecasts (0 problems); unit tests cover targets, timing, bootstrap and the Timestral calibration." }],
    [{ text: "Not re-checked: ", o: { bold: true } }, { text: "the raw data, the training (not re-run) and the bootstrap intervals (order only)." }],
  ], { x: L, y: 4.72, w: w1, h: 2.2, fontSize: 10.5 });
  const x2 = L + 6.95, w2 = CW - 6.95;
  header(s, x2, 2.45, w2, "References");
  const refs = [
    "Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. International Economic Review, 39(4), 885–905.",
    "Corsi, F. (2009). A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics, 7(2), 174–196.",
    "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics, 17(3), 1217–1241.",
    "Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246–256.",
    "Mistral AI (2026). Timestral: time series foundation models. Announcement mistral.ai/news/timestral; code github.com/mistralai/timestral (commit a534bdb); weights huggingface.co/mistralai/Timestral-5M.",
  ];
  T(s, refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 7 } })), { x: x2, y: 2.92, w: w2, h: 3.95, fontSize: 9.5 });
}

pres.writeFile({ fileName: path.join(__dirname, OUT) }).then((f) => console.log("wrote " + f));

// ================================================================================ APPENDIX
pres = newPres("Technical appendix: What Can Machine Learning Predict in NASDAQ Markets?");
let An = 0;
const appx = (title, subtitle) => { An += 1; return content(`A${An}  ${title}`, subtitle); };

// ---------------------------------------------------------------- A1. Reproduction and verification
{
  const s = appx("Reproduction and Verification", "Commands, environment and what the automatic checks do and do not cover");
  const steps = [
    ["1  Environment", "Python 3.12.3, CPU; exact run versions in requirements-lock.txt and run.json", "pip install -r requirements-lock.txt -r requirements-timestral.txt"],
    ["2  Data (not in Git)", "NQ 1-minute bars 2006–2025 (TradeStation, licensed); Timestral-5M weights in the local Hugging Face cache", "$env:FUTURES_DATA_DIR = \"C:\\path\\to\\futures-data\""],
    ["3  Train and forecast", "all models, five timeframes", "python multitarget.py --tag final"],
    ["4  Tables and charts", "results, rankings, timeframe map, charts, deck_data.json", "python multitarget_report.py <run> reports\\multitarget_final"],
    ["5  Sample and Timestral checks", "appendix A3 and A4 (read the 1-minute data / saved forecasts)", "python tools\\sample_checks.py <run> reports\\multitarget_final\npython tools\\timestral_same_sample.py <run> reports\\multitarget_final"],
    ["6  Copy, verify, build", "copy deck_data.json, charts and the two JSON files to presentation\\", "python verify_multitarget.py <run> reports\\multitarget_final presentation\ncd presentation; npm ci; node build_deck.js"],
  ];
  s.addTable([[hdr("Step"), hdr("What"), hdr("Command (PowerShell, repository root)")],
    ...steps.map(([a, b, c]) => [cell(a, { bold: true, color: RED }), cell(b, { color: MID }), cell(c, { fontFace: "Consolas", fontSize: 9 })])],
    topts(L, 1.55, 8.3, [1.75, 2.75, 3.8], { fontSize: 10, rowH: 0.56 }));
  const x2 = L + 8.6, w2 = CW - 8.6;
  header(s, x2, 1.5, w2, "Verified");
  bullets(s, ["every point estimate in the results tables, recomputed independently from the saved forecasts",
    "the ranking rules, the five-model tables, the timeframe map, the data profile and the deck data",
    "past-to-future Timestral calibration: no calibration target after the forecasts it is used for",
    "unit tests (pytest) for targets, timing, splits, bootstrap and calibration"], { x: x2, y: 1.95, w: w2, h: 2.3, fontSize: 10 });
  header(s, x2, 4.3, w2, "Not verified");
  bullets(s, ["the raw 1-minute data and the candle building (checked separately, A3)",
    "the training: models were not re-fitted",
    "bootstrap intervals: only their order is checked",
    `the run used an uncommitted version of multitarget.py (hash in run.json); later changes: majority-class fix applied to saved outputs, docstrings`],
    { x: x2, y: 4.75, w: w2, h: 2.15, fontSize: 10 });
}

// ---------------------------------------------------------------- A2. Selection and statistics
{
  const s = appx("Selection and Statistics", "How models were tuned, ranked and given uncertainty intervals");
  bullets(s, [
    [{ text: "Tuning: ", o: { bold: true } }, { text: "hyper-parameters (Ridge α, logistic C) and early stopping (XGBoost, networks) use the validation block (2017-08-25 → 2021-07-14) only." }],
    [{ text: "Evidence score z: ", o: { bold: true } }, { text: "paired validation improvement over the task's naive benchmark (squared-error reduction vs zero; accuracy gain vs majority; squared log-RV error reduction vs persistence) divided by its moving-block bootstrap standard error (20-session blocks, 2,000 replicates)." }],
    [{ text: "Revision (not blind): ", o: { bold: true } }, { text: "the first volatility score was the QLIKE reduction (Patton, 2011). On validation a few limit-locked March 2020 candles with almost zero variance dominated it, so after a first report had printed test numbers it was replaced by the log-RV squared-error reduction." }],
    [{ text: "Five-model tables (slides 6–8): ", o: { bold: true } }, { text: "one model per family on each task's best validation timeframe, ranked by z. Timeframe map (slide 4): the highest-z trained model per task and timeframe." }],
    [{ text: "Intervals: ", o: { bold: true } }, { text: "differences (accuracy gains) use the bootstrap of paired per-row differences; ratios (R²-type error cuts) resample model and benchmark losses together and recompute the ratio in each replicate (Künsch, 1989)." }],
    [{ text: "Majority-class fix: ", o: { bold: true } }, { text: "the first run counted dojis as \"not up\"; the majority class was recomputed from non-doji training candles without refitting any model (tools/mt_fix_majority.py)." }],
  ], { x: L, y: 1.6, w: CW, h: 5.3, fontSize: 12 });
}

// ---------------------------------------------------------------- A3. Sample checks
if (SC) {
  const s = appx("Sample Checks", "Why the daily test ends on 2025-01-30, and whether incomplete candles matter");
  const d = SC.daily_sample, lr = d.lost_reasons;
  header(s, L, 1.5, 6.0, "A. Daily sample end (measured, tools/sample_checks.py)");
  const fl = d.flagged_minutes.filter((m) => m.session >= "2021-07-15");
  bullets(s, [
    [{ text: "Cause: ", o: { bold: true } }, { text: `all ${d.of_which_not_in_data_set} sessions from 2025-01-31 to 2025-05-30 are dropped. ${lr["flagged_minute_in_last_50_sessions (study.build)"]} fall within 50 sessions after a session with a flagged 1-minute bar (|1-min log return| > 2%); 2025-01-31 itself is dropped because its label session (2025-02-03) contains one.` }],
    [{ text: "Flagged bars look like genuine moves, not bad prints: ", o: { bold: true } }, { text: `e.g. ${fl.slice(-3).map((m) => `${m.session} (${f2(m.log_return_pct, 1)}%)`).join(", ")}. ${d.flagged_after_gap_over_60min} of the ${d.flagged_minutes.length} flags in 2006–2025 come right after a gap of more than an hour (weekend, holiday or daily break); the 2022 flags fall at 8:30 ET, when US macro data are released.` }],
    [{ text: "Impact: ", o: { bold: true } }, { text: `${d.daily_rows_in_50_session_window_by_split.test} of ${d.daily_sessions_by_split.test} daily test sessions (train ${d.daily_rows_in_50_session_window_by_split.train}, validation ${d.daily_rows_in_50_session_window_by_split.val}) lie in such windows, so the daily sample under-represents turbulent periods. Intraday the window is 50 candles, a negligible loss.` }],
    [{ text: "Not corrected here: ", o: { bold: true } }, { text: "a fix (e.g. not flagging opening gaps) changes the sample, so it needs a new daily run; the saved results are kept as they are." }],
  ], { x: L, y: 1.95, w: 6.0, h: 4.95, fontSize: 10.5 });
  const x2 = L + 6.35, w2 = CW - 6.35;
  header(s, x2, 1.5, w2, "B. Full-length candles with missing 1-minute bars");
  const IC = SC.incomplete_candles, fr = ["5min", "15min", "30min", "1h"];
  const rows = fr.map((f) => { const r = IC[f], sh = r.share_full_candles_with_missing_minutes_pct, a = r.test_rescored.all_rows, c = r.test_rescored.complete_label_candles_only;
    const k = Object.keys(a).find((x) => x.includes("mlp_128"));
    return [num(FREQ[f], { bold: true }), num(`${sh.all_years.toFixed(1)}%`), num(`${sh.test.toFixed(2)}%`), num(`${sh["test_08:30-15:00_CT"].toFixed(3)}%`),
      num(r.test_rows_label_candle_incomplete.toLocaleString("en-US")), num(`${a[k].toFixed(2)} → ${c[k].toFixed(2)}`)]; });
  s.addTable([[hdr("Candle"), hdr("All years"), hdr("Test"), hdr("Test, 8:30–15:00 CT"), hdr("Test labels affected"), hdr("DNN 128 error cut (%) all → complete")], ...rows],
    topts(x2, 1.95, w2, [0.75, 0.8, 0.7, 1.05, 1.05, w2 - 4.35], { fontSize: 9.5, rowH: 0.42 }));
  bullets(s, [
    "A full-length candle with 4 of its 5 minutes is accepted as complete; the count is stored (n_minutes) but not used (unit test documents it).",
    "Missing minutes are mostly overnight in 2006–2017, consistent with thin trading (no bar is written for a minute without trades), and almost absent in regular hours. In the test, a few sessions with long gaps coincide with quarterly contract-roll days (e.g. 2021-09-09, 2022-09-08): dataset gaps.",
    "Rescoring the saved test forecasts without incomplete label candles moves the rescored metrics (HAR-OLS, DNN, XGBoost; return, direction, volatility) by at most 0.4 points: no change to the conclusions.",
  ], { x: x2, y: 4.25, w: w2, h: 2.65, fontSize: 10 });
}

// ---------------------------------------------------------------- A4. Timestral on the same sample
if (TSS) {
  const s = appx("Timestral on the Same Sample", "Mistral's Timestral-5M vs the trained models on exactly the same random test candles (saved forecasts)");
  const g = (f, task, m, tgt) => TSS.find((r) => r.frequency === f && r.task === task && r.model === m && (!tgt || r.target === tgt));
  const dnnR = CMP.return.find((r) => r.model.startsWith("mlp_")).model, dnnD = CMP.direction.find((r) => r.model.startsWith("mlp_")).model,
        dnnV = CMP.volatility.find((r) => r.model.startsWith("mlp_")).model;
  const BODY = "body C(t+1) vs O(t+1)", CC = "close-to-close C(t+1) vs C(t)";
  const v = (r, d = 2, u = "") => r ? `${f2(r.value, d)}${u}` : "–";
  const rows = FREQS.map((f) => [num(FREQ[f], { bold: true }), num(g(f, "return", "timestral").n.toLocaleString("en-US")),
    num(v(g(f, "return", "timestral"), 2, "%")), num(v(g(f, "return", dnnR), 2, "%")),
    num(v(g(f, "direction", "timestral", BODY))), num(v(g(f, "direction", "timestral", CC))), num(v(g(f, "direction", dnnD, BODY))),
    num(v(g(f, "volatility", "timestral (spread + calibration)"), 1, "%")), num(v(g(f, "volatility", dnnV), 1, "%")),
    num(`${g(f, "volatility", "timestral raw spread (uncalibrated)").value.toFixed(2)} / ${g(f, "volatility", "persistence (current log RV)").value.toFixed(2)}`)]);
  s.addTable([[hdr("Candle"), hdr("Sample"), hdr("Return R²: Timestral"), hdr("Return R²: DNN"), hdr("Direction pp: Timestral, body"),
               hdr("Direction pp: Timestral, close-to-close"), hdr("Direction pp: DNN, body"), hdr("Vol. error cut: Timestral"),
               hdr("Vol. error cut: DNN"), hdr("Corr. with next log RV: spread / current RV")], ...rows],
    topts(L, 1.5, CW, [0.85, 0.85, 1.15, 1.1, 1.25, 1.35, 1.15, 1.25, 1.1, CW - 10.05], { fontSize: 9.5, rowH: 0.4 }));
  bullets(s, [
    [{ text: "Target mismatch is not the reason: ", o: { bold: true } }, { text: "Timestral forecasts the next close, the study scores the candle body; scored on the close-to-close direction it does no better (columns 5–6)." }],
    [{ text: "Same rows: ", o: { bold: true } }, { text: "the trained models keep their advantage on Timestral's own sample, so the gap is not a sampling artefact." }],
    [{ text: "Volatility is indirect: ", o: { bold: true } }, { text: "Timestral's volatility comes from the spread of its PRICE forecast (q10–q90), mapped to log RV by a linear calibration; the trained models forecast log RV directly from past RV. The raw spread correlates with next log RV slightly less than current RV does, and the calibration fails beyond 5 minutes." }],
    [{ text: "Scope: ", o: { bold: true } }, { text: "one small model (5M), frozen weights, 512-close context, one market: not a verdict on foundation models in general." }],
    [{ text: "Proposed next experiment (not run): ", o: { bold: true } }, { text: "aligned targets — forecast open and close of t+1 (body direction) and the quantiles of the next candle's 1-minute returns, then derive RV from Timestral's own path; compare 28M–209M on the same sample." }],
  ], { x: L, y: 4.4, w: CW, h: 2.5, fontSize: 10.5 });
}

// ---------------------------------------------------------------- A5. Data profile, validation and test
{
  const s = appx("Data Profile", "Descriptive statistics of the three targets, validation and test blocks (data_profile.csv)");
  const rows = [];
  ["val", "test"].forEach((sp) => FREQS.forEach((f) => { const p = PROF(f, sp);
    rows.push([num(sp === "val" ? "Validation" : "Test", { bold: true, color: RED }), num(FREQ[f], { bold: true }), num(p.n.toLocaleString("en-US")),
      num(`${p.first_session} → ${p.last_session}`), num(`${pctv(p.share_up)} / ${pctv(p.share_down)} / ${pctv(p.share_doji)}`),
      num(p.ret_sd_bps.toFixed(1)), num(f2(p.ret_acf1, 3)), num(p.lrv_mean.toFixed(2)), num(pctv(p.share_rv_up)), num(f2(p.lrv_acf1))]); }));
  s.addTable([[hdr("Block"), hdr("Candle"), hdr("Rows"), hdr("Sessions"), hdr("UP / DOWN / doji"), hdr("Return s.d. (bp)"),
               hdr("ACF return"), hdr("Mean log RV"), hdr("RV rises"), hdr("ACF log RV")], ...rows],
    topts(L, 1.5, CW, [1.05, 0.85, 1.05, 2.4, 2.1, 1.0, 0.95, 0.95, 0.9, CW - 11.25], { fontSize: 9.5, rowH: 0.4 }));
  note(s, "ACF = lag-1 autocorrelation (intraday: pairs within a session; daily: consecutive sessions). RV = sum of squared 1-minute log returns inside the candle.", 6.4, 0.4);
}

pres.writeFile({ fileName: path.join(__dirname, OUT_APPX) }).then((f) => console.log("wrote " + f));
